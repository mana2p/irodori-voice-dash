# 声のデザイン・ガチャ選定・参照潜在固定手順

新しい声（キャプション）を設計したり、ガチャを回して参照潜在（`.npz`）を作り直したりする場合のみ参照すること。

> コマンド例中の `<python>` は仮想環境の Python（Windows: `<runtime-dir>\.venv\Scripts\python.exe`、Linux: `<runtime-dir>/.venv/bin/python`）、`<skill-dir>` は本スキルディレクトリを指す。

---

## 声優・キャラクター名での指定に関する規約（選定の前提ルール）

- 本スキルの趣旨は、シードのランダム性を活かして好みの声を「ガチャ感覚で選ぶ」ことであり、実在の人物や特定キャラクターの声の模倣ではない。
- ユーザーが特定の声優名やキャラクター名を挙げて「○○の声に寄せてほしい」と求めた場合は、**その固有名詞による指定を受け付けない**（模倣は本スキルの趣旨外であることを簡潔に伝える）。
- 代わりに、その声のどんなところが好きなのかを、固有名詞を使わずに雰囲気や特徴の言葉で表してもらうよう丁寧に要請する（例：高さ、若々しさ、抑揚の大きさ、落ち着き、元気さ、息づかい、など）。
- 得られた特徴表現だけを使ってキャプション設計と選定に進む。**固有名詞そのものをキャプションやプロンプトに入力しないこと。**

---

## 手順 4: 声を選ぶ (混合文で選定する)

声はキャプションの文字列だけでは決まらない。**引き当てた声を参照で固定する**のが本筋であり、この手順は「良い引きを効率よく引く」ための選定フロー。

1. **キャプションを書いて固定する** (選定中は変えない):
   - 数値 (Hz・オクターブ) は書いても効かない。相対表現で記述する。
   - 先頭に性別と声の高さ。属性は 2〜4 個。ブレスは語彙で明示する。
   - 例：「大人の女性の声。高さは中くらいからやや高めで、明るく澄んだトーン。明瞭で聞き取りやすく、知的なナレーション調で丁寧に話す。」
2. **選定文には「短い混合文」を使う**:
   - 悲しい内容と楽しい内容を 1 つの文群に両方含める (目安 60 字前後)。
   - 勝ちクリップ自体が両方の感情の演技を含むため、後で参照固定した際の感情リーク（シリアスな文が楽しげに引っ張られる現象）を防止できる。
   - 例：「最新の調査結果では深刻な課題も指摘されていますが、確かな解決の兆しも見えています。私たちがサポートします。」
3. **5案をシード出しして耳で選ぶ**:
   - まず `scripts/tokenize_text.py` でトークン JSON を作成する：
     ```bash
     <python> <skill-dir>/scripts/tokenize_text.py \
       --irodori-dir <runtime-dir>/official \
       --tokenizer-dir <runtime-dir>/models/int8/tokenizer \
       --no-duration-features \
       --text "<選定文>" \
       --caption "<キャプション>" \
       --out tokens.json
     ```
   - 大きく離したランダムシード（例: 42, 1234, 2026, 7777, 9999）で 5 案生成する：
     ```bash
     <python> <skill-dir>/scripts/infer_onnx.py \
       --models-dir <runtime-dir>/models --variant int8 --tokens tokens.json \
       --truncation-factor 0.85 --seed 101 --out take_101.wav
     ```
   - 気に入ったものがなければ別のシードでさらに 5 案出す。
4. **声の特徴メモを残す** ＆ **推奨パラメータ**:
   - `--truncation-factor 0.85`（ノイズ感が大幅に低減する黄金値。0.7 まで絞ると抑揚が失われ淡々になる）。

---

## 手順 5: 声を固定する (参照潜在: ref-latents)

※ 実行には `<runtime-dir>/models/f32/dacvae_encode.onnx` (約 104 MB) が必要。

1. 手順 4 の **勝ちクリップ 1 本を単一参照** にする。
2. **参照潜在 (`.npz`) を一度だけ作成して保存**:
   ```bash
   <python> <skill-dir>/scripts/make_ref_latents.py \
     --models-dir <runtime-dir>/models --clips winner.wav --out ref/my-voice.npz
   ```
   ※ `dacvae_encode.onnx` はサンプリング式エンコーダのため、内部で **8 回エンコードして平均**を取り、事後平均 (posterior mean) に近似する。1 回エンコードで済ませないこと。
3. 検証としてポジティブな文とシリアスな文を `scripts/speak_long_text.py --ref-latents ref/my-voice.npz` で生成し、声の同一性と感情の切り替わりを確認する。

---

## 補足（上級者・裏技ルート）：`.speaker.safetensors` を直接使う場合

※ 本来の「音声ガチャ → `.npz` 固定」フローとは別枠の応用オプションです（通常時は使わなくて OK）。

- 公式の Speaker Inversion 形式で作成した `*.speaker.safetensors`（キー `speaker_embedding`, shape `(tokens, 768)`）がある場合、`speak_long_text.py` / `infer_onnx.py` の `--ref-embed <path.speaker.safetensors>` で `speaker_encoder.onnx` をバイパスして直接合成できます（別途 `safetensors` パッケージが必要）。
- `--caption` 未指定時は純粋な埋め込み声質（`caption` 無効）で生成されます。
- **利用上の注意**：本スキル標準のモデル（`Irodori-TTS-v4.1-Small-MF` / `speaker_dim=768`）向けに作成したもののみ利用可能（v3 RF 版など別世代・別方式のモデルで作ったデータは音質が破綻します）。外部配布データを使う場合は配布元の利用規約および実在人物・キャラクター模倣禁止・なりすまし禁止のルールを必ず守ること（公開リポジトリへのコミットも厳禁）。

