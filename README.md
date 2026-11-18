# irodori-voice-dash

**Irodori-TTS v4.1-Small-MF** の **ONNX int8 推論**をローカル CPU（Windows / Linux）で手軽・高速に動かすためのクロスエージェント対応スキル（[Agent Skills](https://agentskills.io/) 仕様準拠）です。

Antigravity・Claude Code・OpenAI Codex・Cursor・VS Code（GitHub Copilot）など主要 AI コーディングエージェント／エディタで共通利用できます。

> エージェント向け手順書は [SKILL.md](SKILL.md)、詳細リファレンスは [references/setup.md](references/setup.md) / [references/voice-design.md](references/voice-design.md) / [references/EMOJI_ANNOTATIONS.md](references/EMOJI_ANNOTATIONS.md) / [references/troubleshooting.md](references/troubleshooting.md)、実験記録は [docs/findings.md](docs/findings.md) を参照してください。

## 特徴

- **単一プロセス・高速バッチ生成（`scripts/speak_long_text.py`）**：
  - 長文を句点・感嘆符・最大80字単位で自動分割し、1プロセス内で **ONNX セッションと条件（キャプション・参照潜在）を初回1回のみ読込・計算** して連続合成 → `0.4` 秒間隔で結合します。
  - `tokenizer.json` がある環境では `import torch` を踏まず `tokenizers` 単体でエンコードする高速パスを自動選択します。
- **ONNX Runtime (CPU) のみで動作**：Windows / Linux 双方で動作（PyTorch 公式実装の推論フローを分割 ONNX グラフで再現し、float 経路の一致を検証済み）。
- **声の固定（`.npz`）と感情リーク対策**：混合文で選んだ勝ちクリップを `make_ref_latents.py`（DACVAE 8回平均近似）で参照潜在化し、安定した同一声質で読み上げられます。

## 対応エディタ・配置パス一覧

[skills CLI](https://skills.sh/) を使って主要エージェント／エディタへ一括導入できます：

```bash
npx skills add mana2p/irodori-voice-dash
```

手動で配置する場合は、このスキルフォルダ `irodori-voice-dash/` を各ツールの標準ディレクトリへコピー（またはシンボリックリンク）してください。

| ツール / エージェント | プロジェクト配置先 | グローバル配置先 |
|---|---|---|
| **共通標準（Antigravity / Codex / Cursor）** | `.agents/skills/irodori-voice-dash` | `~/.agents/skills/irodori-voice-dash` |
| **VS Code / GitHub Copilot** | `.github/skills/irodori-voice-dash` | `~/.copilot/skills/irodori-voice-dash` |
| **Claude Code** | `.claude/skills/irodori-voice-dash` | `~/.claude/skills/irodori-voice-dash` |
| **Cursor（専用パス）** | `.cursor/skills/irodori-voice-dash` | `~/.cursor/skills/irodori-voice-dash` |

## セットアップ（概要）

詳細は [references/setup.md](references/setup.md) を参照してください。

```bash
# 1. ONNX モデルを取得 (重みは本スキルには含まれません)
python scripts/download_models.py --models-dir ./models

# 2. 推論＆高速トークナイズ用 venv (Python 3.12 推奨)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python onnxruntime==1.31.1 numpy soundfile pyloudnorm scipy tokenizers

# 3. 正規化・形態素解析モジュール用に公式 repo を取得
git clone https://github.com/Aratako/Irodori-TTS.git ./official
```

※ Windows PowerShell の場合は `.venv\Scripts\python.exe` と `scripts/download_models.ps1` を使用します。

## 使い方

### 1. サクッと1コマンドで話す（短文・長文共通）

```bash
.venv/bin/python scripts/speak_long_text.py \
  --models-dir ./models \
  --irodori-dir ./official \
  --text "こんにちは。長文も自動で文ごとに分割して結合します。" \
  --out out.wav
```

固定声（`.npz`）を使う場合は `--ref-latents`（または環境変数 `IRODORI_REF_LATENTS`）を指定します：

※ 感情・話し方・効果音を制御したい場合は、対応絵文字を **文頭または文中** に挿入します（同じ絵文字を重ねると強調、自動文分割との干渉を防ぐため「。」の直後は避けてください。一覧は [references/EMOJI_ANNOTATIONS.md](references/EMOJI_ANNOTATIONS.md) を参照）。

```bash
.venv/bin/python scripts/speak_long_text.py \
  --models-dir ./models \
  --irodori-dir ./official \
  --ref-latents ref/my-voice.npz \
  --seed 9999 \
  --caption "大人の女性の声。明るく澄んだトーンで丁寧に話す。" \
  --text-file script.txt \
  --out narration.wav
```

### 2. 声をデザインして固定する

詳細は [references/voice-design.md](references/voice-design.md) を参照してください。

```bash
# 1. 混合文をトークナイズ
.venv/bin/python scripts/tokenize_text.py \
  --irodori-dir ./official \
  --tokenizer-dir ./models/int8/tokenizer \
  --no-duration-features \
  --text "最新の調査結果では深刻な課題も指摘されていますが、確かな解決の兆しも見えています。" \
  --caption "若い女性の声。明るく親しみやすく話す。" \
  --out tokens.json

# 2. シードを変えて 5 案生成して試聴
.venv/bin/python scripts/infer_onnx.py \
  --models-dir ./models --variant int8 --tokens tokens.json \
  --truncation-factor 0.85 --seed 101 --out take_101.wav

# 3. 勝ちクリップを .npz として固定（8回平均近似）
.venv/bin/python scripts/make_ref_latents.py \
  --models-dir ./models --clips take_101.wav --out ref/my-voice.npz
```

### 3. 【上級者向け・裏技】Speaker Inversion（`.speaker.safetensors`）を使う場合

> ⚠️ **注意（本来の想定外の機能です）**：
> 本スキルの本来の目的は、上の **「音声ガチャで好みの声を引き当て、`make_ref_latents.py` で `.npz` として固定する」** 使い方であり、一般の音声ガチャ向け機能ではありません。
> 公式または事前学習済みの **Speaker Inversion 埋め込みファイル（`.speaker.safetensors`）** をすでに手元に持っている場合の **裏技的な応用ルート** として、`--ref-embed`（環境変数 `IRODORI_REF_EMBED`）による読み込みにも対応しています。利用時は以下に注意してください：
>
> - 追加で `safetensors` パッケージが必要です（`uv pip install safetensors`）。
> - `Irodori-TTS-v4.1-Small-MF`（`speaker_dim=768`）向けに作成された埋め込みのみ使用できます（**v3 RF 版など別世代・別方式のモデルで作った `.speaker.safetensors` を流用すると次元一致しても音声が破綻します**）。
> - `--caption` を省略すると純粋に埋め込み側の声だけで合成されます（`caption_mask=0`）。
> - 第三者が配布している `.speaker.safetensors` 等を扱う場合は、**配布元のライセンス・利用規約および「同意のない他者のなりすまし・特定人物の模倣禁止」の規約を必ず遵守** してください（無断利用・公開は厳禁、`.speaker.safetensors` も公開リポジトリへ含めないでください）。

```bash
.venv/bin/python scripts/speak_long_text.py \
  --models-dir ./models \
  --irodori-dir ./official \
  --ref-embed ref/custom.speaker.safetensors \
  --text "サンプルボイスのテスト音声です。" \
  --out sample.wav
```

### 主要パラメータ決定値

| パラメータ | 値 | 根拠 |
|---|---|---|
| `seed` | 任意（固定推奨） | 再現性は同一テキスト・同一条件の範囲 |
| `truncation_factor` | `0.85` | ノイズ激減と表現力のバランス（`0.7` は感情文が淡々に） |
| 参照の正規化 | `-16.0 LUFS` | 公式 `ref_normalize_db` 準拠 |
| 参照エンコード平均回数 | `8` 回 | サンプリング式エンコーダの事後平均近似 |
| 長文の分割 | 1区間 70〜80 字 | 145 字一発では語の脱落/挿入が発生 |
| 結合の間隔 | `0.4` 秒 | `scripts/join_wavs.py --gap 0.4` |

## モデルの入手先

重みは再配布しません。各配布元から取得してください（`scripts/download_models.*` が自動化）：

- int8 + 共有 float グラフ：[Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8](https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8)
- float グラフ + 参照エンコーダ：[raratu/Onsei-iOS-Models](https://huggingface.co/raratu/Onsei-iOS-Models) (`v4.1-small-mf/`)
- 公式実装・元モデル：[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) / `Aratako/Irodori-TTS-v4.1-Small-MF`

## 利用上のルール: 声優・キャラクターの声について

本スキルの声の選定は、Irodori-TTS のシードごとの違い・ランダム性を活かして、
ユーザー好みの声を「ガチャ感覚で選ぶ」ことが趣旨です。
実在の人物やキャラクターの声を模倣するための仕組みではありません。

そのため、次を禁じます：

- 本スキル（声の選定フロー）を使い、特定の声優の声や特定のキャラクターの
  声に寄せようとする行為。
- それを意図したプロンプトで AI エージェントを誘導する行為、
  および、そうして作られた音声そのものをアップロード・公開する行為。

声の希望は、固有名詞（声優名・キャラクター名）ではなく、その声の雰囲気や
特徴を表す言葉で伝えてください（例：高めで若々しい、抑揚が大きい、
落ち着いた、息づかいが感じられる、など）。

## ライセンス / クレジット

- 本 repo の**コード**は MIT License（[LICENSE](LICENSE) 参照）。
- モデル重みのライセンスは各配布元に従います。Irodori-TTS 本体は MIT ですが
  倫理的制限（同意のない他者のなりすまし等の禁止）が付随します。必ず確認してください。
- この ONNX 経路では透かし（SilentCipher）は入りません。生成音声を公開する場合は
  AI 音声である旨の明示を推奨します。
- 基盤となる構成・実験知見は [kgmkm/muse-irodori-voice](https://github.com/kgmkm/muse-irodori-voice) をもとに、単一プロセス高速化・Windows/Linux 両対応・Agent Skills 標準化を行ったものです。
- 謝辞：Irodori-TTS 作者の Aratako 氏、ONNX 変換を公開された Ryoppi 氏・raratu 氏、元キット作者の kgmkm 氏および検証環境である Muse に感謝します。
