---
name: irodori-voice-dash
description: >-
  Fast local Japanese text-to-speech skill using Irodori-TTS v4.1-Small-MF ONNX
  int8 in a single process. Generates 48kHz/16bit PCM WAV files with optional
  fixed-voice reference latents (.npz) and automatic long-text sentence
  splitting and joining. Use when asked to speak, narrate text, generate voice
  notes/audio summaries, or design and fix a custom TTS voice.
license: MIT
compatibility: >-
  Requires Python 3.12, onnxruntime, numpy, soundfile, pyloudnorm, scipy,
  tokenizers, plus local Irodori-TTS ONNX models and official repo checkout.
  Optional: safetensors (only for experimental --ref-embed). Supports Windows and Linux.
---

# Irodori Voice Dash

Irodori-TTS v4.1-Small-MF（ONNX int8 / CPU）を単一プロセスで高速実行し、48kHz / 16bit PCM WAV を生成する音声合成スキル。

## パスと環境解決のルール

- このスキル内のファイル（`scripts/`・`references/`・`docs/`）は、すべて **この `SKILL.md` があるフォルダ（`<skill-dir>`）基準の相対パス** で解決すること。
- ワークスペースのランタイム構成（`.venv`、`models/`、`official/`、および固定声 `.npz` やシード等のローカル設定）は、プロジェクトの `AGENTS.md` / `CLAUDE.md` または環境変数を確認して適用すること：
  - `IRODORI_MODELS_DIR`: ONNX モデル格納先（`int8/` と `f32/` を含むディレクトリ）
  - `IRODORI_TTS_DIR`: 公式 [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のチェックアウト先
  - `IRODORI_TOKENIZER_DIR`: トークナイザディレクトリ（未指定時は `<models-dir>/int8/tokenizer` または HF キャッシュを使用）
  - `IRODORI_REF_LATENTS`: 固定参照潜在（`.npz`）の既定パス（任意）

## 音声の生成手順（短文・長文共通）

テキストが渡されたら [`scripts/speak_long_text.py`](scripts/speak_long_text.py) を実行する。
内部で **単一プロセス起動のまま、文分割（句点・感嘆符・改行、最大80字）→ `torch` レス高速トークナイズ → ONNX セッション＆共通条件（キャプション・参照潜在）を初回1回のみ読込・計算 → 連続 `int8` 推論 → 0.4秒ギャップ結合** まで一括実行される。

- **Linux / macOS**:
  ```bash
  <venv>/bin/python <skill-dir>/scripts/speak_long_text.py \
    --text "<読み上げたいテキスト>" \
    --out out.wav
  ```
- **Windows (PowerShell)**:
  ```powershell
  <venv>\Scripts\python.exe <skill-dir>\scripts\speak_long_text.py `
    --text "<読み上げたいテキスト>" `
    --out out.wav
  ```

よく使うオプション：
- `--text-file <path>`: 長文テキストファイルを読み込む
- `--ref-latents <path.npz>`: 声質を固定する参照潜在を指定（未指定時は `IRODORI_REF_LATENTS` を参照）
- `--ref-embed <path.speaker.safetensors>`: （上級者向け裏技）公式 Speaker Inversion 埋め込みを直接読む（通常の音声ガチャ向けではなく `safetensors` と `speaker_dim=768` 互換・配布規約遵守が必要。詳細は [README.md](README.md) / [references/voice-design.md](references/voice-design.md)）
- `--caption "<caption text>"` / `--seed <int>`: キャプション・シードを指定（固定声使用時もプロジェクト設定に合わせて指定）
- `--models-dir` / `--irodori-dir` / `--tokenizer-dir`: ランタイムの配置先を明示（既定では環境変数・`Irodori-TTS/*`・カレント配下を自動探索）
- `--threads <int>`: ONNX Runtime スレッド数（既定は `min(6, max(2, cpu_count // 2))`）

生成後は出力 WAV ファイルの存在を確認し、ユーザーがそのまま再生できるパスやリンクを報告する。

## 同梱スクリプト一覧

- [`scripts/speak_long_text.py`](scripts/speak_long_text.py) — 長文分割＋単一プロセス連続推論＋0.4秒結合のメイン CLI
- [`scripts/tokenize_text.py`](scripts/tokenize_text.py) — 公式トークナイザ互換の単体トークナイズ（`--no-duration-features` で `torch` 不要）
- [`scripts/infer_onnx.py`](scripts/infer_onnx.py) — 単発 ONNX 推論（声ガチャ選定や `f32` 検証用）
- [`scripts/make_ref_latents.py`](scripts/make_ref_latents.py) — 参照 WAV から固定声 `.npz`（DACVAE 8回平均）を作成
- [`scripts/join_wavs.py`](scripts/join_wavs.py) — 複数 WAV を指定ギャップで結合
- [`scripts/download_models.py`](scripts/download_models.py) / [`scripts/download_models.ps1`](scripts/download_models.ps1) / [`scripts/download_models.sh`](scripts/download_models.sh) — ONNX モデル取得

## 利用上の注意・禁止事項

- **声優・キャラクター・実在人物の声の模倣禁止**：シードのランダム性を活かして好みの声を「ガチャ感覚で選ぶ」のが趣旨。固有名詞（声優名・キャラクター名等）での声寄せ要求は受け付けず、声の高さ・若々しさ・抑揚・息づかい等の特徴表現のみで設計すること（詳細は [references/voice-design.md](references/voice-design.md)）。
- **モデル重み・`.npz` / `.safetensors` の扱い**：モデル重みや個人の参照潜在（`.npz`）・外部 Speaker Inversion データ（`.speaker.safetensors`）は再配布・公開リポジトリへ含めない。Irodori-TTS 本体および各データ配布元のライセンス・倫理的制限（同意のない他者のなりすまし禁止等）を遵守すること。
- **透かしなし**：この ONNX 経路では透かし（SilentCipher）は入らないため、生成音声を公開する場合は AI 音声である旨の明示を推奨する。

## 詳細リファレンス（必要時のみ参照）

- 環境の構築・モデルダウンロード・float 検証: [references/setup.md](references/setup.md)
- 新しい声の設計・5案ガチャ選定・参照潜在の固定: [references/voice-design.md](references/voice-design.md)
- パラメータ根拠・トラブルシューティング・倫理規約: [references/troubleshooting.md](references/troubleshooting.md)
- ベンチマーク・設計知見の詳細: [docs/findings.md](docs/findings.md)
