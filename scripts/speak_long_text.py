"""Synthesize long text by splitting sentences, generating segments, and joining.

Automates:
1. Splitting long text into sentence-level segments (~50-80 chars).
2. Loading tokenizers (fast torch-free path) and ONNX sessions once in a single process.
3. Precomputing shared caption and speaker conditions once.
4. Synthesizing each segment and joining with a natural pause gap (default 0.4s).
"""
import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_onnx import (  # noqa: E402
    SR,
    encode_caption_state,
    load_sessions,
    prepare_speaker_condition,
    synthesize_waveform,
)
from join_wavs import join_audio_arrays  # noqa: E402
from tokenize_text import load_tokenizers, tokenize_segment  # noqa: E402

DEFAULT_CAPTION = "若い女性の声。明るく親しみやすく話す。"
DEFAULT_SEED = 42
DEFAULT_GAP = 0.4
DEFAULT_THREADS = min(6, max(2, (os.cpu_count() or 2) // 2))


def _resolve_default_dir(env_var: str, candidates: tuple[Path, ...]) -> str:
    """Resolve directory from env var, first existing candidate, or final fallback."""
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    return str(candidates[-1])


def _resolve_tokenizer_dir(explicit: str | None, models_dir: Path) -> str | None:
    """Resolve tokenizer dir from arg/env, local models/<variant>/tokenizer, or HF cache."""
    if explicit:
        return explicit
    local_tok = models_dir / "int8" / "tokenizer"
    if local_tok.is_dir():
        return str(local_tok)
    f32_tok = models_dir / "f32" / "tokenizer"
    if f32_tok.is_dir():
        return str(f32_tok)
    return None


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
    """Split text into sentence-level segments (~50-80 chars max)."""
    # Clean whitespace
    text = text.strip()
    if not text:
        return []

    # First split by explicit line breaks or punctuation (。, ！, ？, !, ?)
    raw_sentences = re.split(r'(?<=[。！？!?\n])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    refined = []
    for s in sentences:
        if len(s) <= max_chars:
            refined.append(s)
        else:
            # Sub-split long sentence by comma (、, ,)
            sub_parts = re.split(r'(?<=[、,])', s)
            current = ""
            for p in sub_parts:
                p = p.strip()
                if not p:
                    continue
                if len(current) + len(p) <= max_chars:
                    current += p
                else:
                    if current:
                        refined.append(current)
                    current = p
            if current:
                refined.append(current)

    return refined


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    default_models_dir = _resolve_default_dir(
        "IRODORI_MODELS_DIR",
        (Path("Irodori-TTS/models"), Path("models"), SKILL_ROOT / "models"),
    )
    default_irodori_dir = _resolve_default_dir(
        "IRODORI_TTS_DIR",
        (Path("Irodori-TTS/official"), Path("official"), Path("Irodori-TTS"), SKILL_ROOT / "official"),
    )

    ap = argparse.ArgumentParser(description="Split long text and synthesize in a single process")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to speak")
    group.add_argument("--text-file", help="Path to text file to speak")

    ap.add_argument("--out", required=True, help="Output wav file path")
    ap.add_argument(
        "--caption",
        default=None,
        help="Voice caption (default: empty when --ref-embed/--ref-latents is set, else neutral sample)",
    )
    ap.add_argument(
        "--ref-latents",
        default=os.environ.get("IRODORI_REF_LATENTS"),
        help="Optional reference latent npz (default: IRODORI_REF_LATENTS env if set)",
    )
    ap.add_argument(
        "--ref-embed",
        default=os.environ.get("IRODORI_REF_EMBED"),
        help="Optional Speaker Inversion .speaker.safetensors (default: IRODORI_REF_EMBED env if set)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--variant", choices=("int8", "f32"), default="int8", help="ONNX model variant")
    ap.add_argument("--truncation-factor", type=float, default=0.85, help="Noise truncation factor")
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP, help="Silence gap between sentences (seconds)")
    ap.add_argument("--models-dir", default=default_models_dir, help="Models directory")
    ap.add_argument("--irodori-dir", default=default_irodori_dir, help="Official Irodori-TTS directory")
    ap.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("IRODORI_TOKENIZER_DIR"),
        help="Tokenizer directory (default: IRODORI_TOKENIZER_DIR, <models-dir>/int8/tokenizer, or HF cache)",
    )
    ap.add_argument("--max-chars", type=int, default=80, help="Max characters per segment")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="ONNX Runtime intra-op threads")
    ap.add_argument("--keep-parts", action="store_true", help="Keep part wav files")

    args = ap.parse_args()

    if args.ref_latents and args.ref_embed:
        ap.error("--ref-latents and --ref-embed are mutually exclusive")

    caption = args.caption
    if caption is None:
        caption = "" if (args.ref_latents or args.ref_embed) else DEFAULT_CAPTION
    has_caption = bool(caption.strip())

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text

    segments = split_text_into_sentences(text, max_chars=args.max_chars)
    if not segments:
        print("Error: No valid text segments found.")
        sys.exit(1)

    print(f"Total {len(segments)} segments to synthesize:")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i}/{len(segments)}] ({len(seg)} chars): {seg}")

    t_start = time.perf_counter()

    # 1. Load tokenizers once and tokenize all segments
    models_dir = Path(args.models_dir).resolve()
    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer_dir, models_dir)
    t0 = time.perf_counter()
    tok_bundle = load_tokenizers(
        Path(args.irodori_dir).resolve(),
        str(Path(tokenizer_dir).resolve()) if tokenizer_dir else None,
    )
    pre_caption_ids = tok_bundle.encode_caption(caption)
    tokenized_segments = [
        tokenize_segment(
            tok_bundle,
            seg,
            caption,
            precomputed_caption=pre_caption_ids,
            include_duration_features=False,
        )
        for seg in segments
    ]
    t_tok = time.perf_counter() - t0

    # 2. Load ONNX sessions once and precompute caption + speaker conditions
    sessions, load_timings = load_sessions(models_dir, variant=args.variant, threads=args.threads)

    cap_ids_list, cap_len = pre_caption_ids
    precomputed_caption_state = encode_caption_state(
        sessions["enc"], cap_ids_list, cap_len, has_caption=has_caption
    )

    ref_latents_path = None
    if args.ref_latents:
        candidate = Path(args.ref_latents).resolve()
        if candidate.exists():
            ref_latents_path = candidate
        else:
            print(f"Warning: ref-latents {candidate} not found. Running without ref.")

    ref_embed_path = None
    if args.ref_embed:
        candidate = Path(args.ref_embed).resolve()
        if candidate.exists():
            ref_embed_path = candidate
        else:
            print(f"Warning: ref-embed {candidate} not found. Running without ref.")

    speaker_cond = prepare_speaker_condition(
        sessions,
        ref_latents=ref_latents_path,
        ref_embed=ref_embed_path,
    )

    # 3. Synthesize each segment in-process
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="irodori_parts_")) if args.keep_parts else None

    part_wavs = []
    t_synth_start = time.perf_counter()
    for i, tok in enumerate(tokenized_segments, 1):
        wav, seg_timings, _ = synthesize_waveform(
            sessions,
            tok,
            speaker_cond,
            seed=args.seed,
            truncation_factor=args.truncation_factor,
            precomputed_caption=precomputed_caption_state,
        )
        part_wavs.append(wav)
        if temp_dir is not None:
            part_path = temp_dir / f"part_{i:03d}.wav"
            sf.write(part_path, wav, SR, subtype="PCM_16")
        print(f"  Generated segment {i}/{len(segments)} ({seg_timings['generation_total']:.2f}s)")
    t_synth = time.perf_counter() - t_synth_start

    # 4. Join wavs and write final output
    full_wav = part_wavs[0] if len(part_wavs) == 1 else join_audio_arrays(part_wavs, SR, gap_sec=args.gap)
    sf.write(out_path, full_wav, SR, subtype="PCM_16")

    if temp_dir is not None:
        print(f"Preserved part wavs in: {temp_dir}")

    t_total = time.perf_counter() - t_start
    print(
        f"\nSuccessfully generated full audio: {out_path} "
        f"(total {t_total:.2f}s: tok {t_tok:.2f}s, load {load_timings['load_total']:.2f}s, synth {t_synth:.2f}s)"
    )


if __name__ == "__main__":
    main()
