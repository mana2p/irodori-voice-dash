# セットアップ・環境構築・移植検証手順

環境を新規構築・再構築する場合や、モデルのダウンロード・トラブル調査を行う場合のみ参照すること。

> コマンド例中の `<skill-dir>` は本スキル（`irodori-voice-dash`）を配置したディレクトリ、`<runtime-dir>` はモデルや仮想環境を置くディレクトリ（例: `./Irodori-TTS` や `./`）を指す。

## 前提環境

- Windows 10/11 (PowerShell 7 / pwsh 推奨) または Linux x86_64
- Python 3.12 (`uv venv --python 3.12` 推奨。Python 3.15 等では `onnxruntime` 等の wheel が未提供の場合あり)
- 推論ランタイム: `onnxruntime==1.31.1`, `numpy`, `soundfile`, `pyloudnorm`, `scipy`
- トークナイズ用: 公式 [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のチェックアウトと `tokenizers`（`PretrainedTextTokenizer` フォールバックおよび単体 CLI の `duration_features` 出力時は `torch`, `transformers` も使用）

---

## 手順 1: モデルと推論環境の準備

### 1. ONNX モデルのダウンロード

スクリプトを実行して ONNX グラフ一式を `<runtime-dir>/models/{int8,f32}/` に配置する。
（重みは本スキルに含まず、各配布元から直接取得する）

- **OS 共通 (Python)**:
  ```bash
  python <skill-dir>/scripts/download_models.py --models-dir <runtime-dir>/models
  ```
- **Windows (PowerShell)**:
  ```powershell
  pwsh <skill-dir>/scripts/download_models.ps1 -ModelsDir <runtime-dir>\models
  ```
  ※ `--ssl-no-revoke` オプションにより閉域・プロキシ環境での Schannel 失効確認エラー (`CRYPT_E_NO_REVOCATION_CHECK`) を回避。
- **Linux / macOS (Bash)**:
  ```bash
  bash <skill-dir>/scripts/download_models.sh
  ```
  ※ `download_models.sh` は `<skill-dir>/models` に保存するため、別ディレクトリに置く場合は Python 版 `--models-dir` を推奨。

※ **罠対策**: `NO_PROXY=localhost,127.0.0.1` を設定すること（IPv6 表記による `Invalid port: ':1]'` エラー防止）。各スクリプト内で自動設定済み。
※ **容量削減**: `int8` 運用と固定潜在の確立後は、`models/f32/text_encoder.onnx`・`dit_step.onnx`・`tokenizer/` は削除可能（計約 2.43 GB 節約）。ただし将来別の声を `.npz` 化する場合は `models/f32/dacvae_encode.onnx`（約 104 MB）のみ残すこと。

### 2. 推論＆トークナイズ用仮想環境 (.venv) の構築

- **Windows**:
  ```powershell
  uv venv --python 3.12 <runtime-dir>\.venv
  uv pip install --python <runtime-dir>\.venv\Scripts\python.exe onnxruntime==1.31.1 numpy soundfile pyloudnorm scipy tokenizers
  ```
- **Linux**:
  ```bash
  uv venv --python 3.12 <runtime-dir>/.venv
  uv pip install --python <runtime-dir>/.venv/bin/python onnxruntime==1.31.1 numpy soundfile pyloudnorm scipy tokenizers
  ```
※ `PretrainedTextTokenizer` フォールバックや `tokenize_text.py` 単体で `duration_features` まで出したい場合のみ `torch` と `transformers` を追加する（`speak_long_text.py` および `tokenize_text.py --no-duration-features` では `tokenizers` のみで動作可能）。

### 3. 公式 Irodori-TTS の準備 (正規化・形態素処理モジュール用)

正規化・形態素解析モジュール (`irodori_tts`) として公式リポジトリを配置する（`models/int8/tokenizer` を使用すれば HF の巨大キャッシュは不要）。

```bash
git clone https://github.com/Aratako/Irodori-TTS.git <runtime-dir>/official
```

---

## 手順 2: float で移植を検証する (新規環境時)

`int8` をいきなり信用しない。検証時は順番を遵守すること：
1. `scripts/infer_onnx.py --variant f32` で1文生成し、公式 PyTorch 版と同条件で比較する。
2. 公式ランタイムの中間テンソル (`text_state`、尺予測 `log_frames` 等) と照合する（`text_state` 最大差 1e-5 オーダー、尺予測は完全一致）。
3. 一致を確認して `--variant int8` に切り替える。

---

## 手順 3: トークナイズ (重要仕様)

`scripts/tokenize_text.py` を使用すること（自作スクリプトでの代用禁止）。
- **本文はレアトークン分割 ON、キャプションは OFF**（公式ランタイム準拠）。
- キャプションに分割を適用すると尺予測がずれる（実測 189 → 178 frames）。
- `tokenizer.json` および `tokenizer_config.json` が存在する場合は `importlib` 経由で `text_normalization.py` / `token_split.py` を直接読み込み、`tokenizers` のみで動作する `torch` レス高速経路が自動選択される（`speak_long_text.py` からは `duration_features` 生成も省略し、単体 CLI でも `--no-duration-features` または `torch` 未導入時に自動スキップする）。
