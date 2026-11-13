"""Download ONNX models for Irodori-TTS into models/int8 and models/f32.

OS-independent Python script to download required ONNX models.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Fix NO_PROXY issue where IPv6 notation breaks httpx/urllib behind some proxies
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"



def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping existing: {dest}")
        return

    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="", flush=True)
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory to save models")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    files = [
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/text_encoder_int8.onnx", models_dir / "int8" / "text_encoder_int8.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/dit_step_int8.onnx", models_dir / "int8" / "dit_step_int8.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/duration.onnx", models_dir / "int8" / "duration.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/context_kv.onnx", models_dir / "int8" / "context_kv.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/speaker_encoder.onnx", models_dir / "int8" / "speaker_encoder.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/dacvae_decode.onnx", models_dir / "int8" / "dacvae_decode.onnx"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/config.json", models_dir / "int8" / "config.json"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/manifest.json", models_dir / "int8" / "manifest.json"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/tokenizer/tokenizer.json", models_dir / "int8" / "tokenizer" / "tokenizer.json"),
        ("https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main/tokenizer/tokenizer_config.json", models_dir / "int8" / "tokenizer" / "tokenizer_config.json"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/text_encoder.onnx", models_dir / "f32" / "text_encoder.onnx"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/dit_step.onnx", models_dir / "f32" / "dit_step.onnx"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/dacvae_encode.onnx", models_dir / "f32" / "dacvae_encode.onnx"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/config.json", models_dir / "f32" / "config.json"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/tokenizer/tokenizer.json", models_dir / "f32" / "tokenizer" / "tokenizer.json"),
        ("https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf/tokenizer/tokenizer_config.json", models_dir / "f32" / "tokenizer" / "tokenizer_config.json"),
    ]

    print(f"Starting ONNX models download to {models_dir}...")
    for url, dest in files:
        download_file(url, dest)
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    main()

