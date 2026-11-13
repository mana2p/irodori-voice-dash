# 検証済みパラメータ・トラブルシューティング・規約

詳細な実験根拠やトラブルシュートが必要な場合のみ参照すること。
さらなる実験記録の詳細は [docs/findings.md](../docs/findings.md) を参照。

---

## 検証済み パラメータ決定値

| パラメータ | 決定値 | 根拠・技術仕様 |
|---|---|---|
| 推論 variant | `int8` | CPU 環境でのメモリ消費 2.67GB、生成時間大幅短縮 |
| MeanFlow ステップ数 | 4 | モデル既定 (蒸留済み CFG なし) |
| truncation_factor | `0.85` | 初期ガウスノイズのスケーリング。0.85 でノイズ激減・表現力維持 |
| 参照正規化 (LUFS) | `-16.0 LUFS` | 公式 `ref_normalize_db` (ITU-R BS.1770) 準拠 + ピーク 1.0 クリップ防止 |
| 参照エンコード平均回数 | 8 回 | サンプリング式 DACVAE エンコーダの事後平均近似 |
| 参照オーディオ総量 | 混合文 1 本 (~10〜12秒) | 感情リーク防止と音色固定の両立 |
| 長文分割単位 | 1文 / 70〜80 字以内 | アテンション破綻・語脱落防止 |
| 文間ポーズ (gap) | 0.4 秒 | 自然な発話リズム |
| 出力仕様 | 48kHz, 16bit PCM | リニア PCM WAV |

---

## トラブルシューティング

| 症状 | 原因と解決策 |
|---|---|
| モデル DL 時に `Invalid port` エラー | `NO_PROXY` に IPv6 表記が含まれている。`NO_PROXY=localhost,127.0.0.1` に設定する。 |
| Windows curl で `CRYPT_E_NO_REVOCATION_CHECK` | Schannel 証明書失効チェック失敗。`curl.exe --ssl-no-revoke` を使用する（`download_models.ps1` 対応済み）。 |
| Windows で `No module named 'resource'` | Unix 専用モジュール。`infer_onnx.py` は Windows 用フォールバック (ctypes) 済み。 |
| `pip install onnxruntime` で `No matching distribution` | Python 3.15 等では wheel 未提供。`uv venv --python 3.12` で Python 3.12 環境を構築する。 |
| 尺予測 (duration) が公式とずれる | キャプションにレアトークン分割がかかっている。キャプション側は split OFF にする。 |
| 参照を渡しても声質が実行ごとに揺らぐ | `dacvae_encode` のサンプリング揺らぎ。`make_ref_latents.py` で 8 回平均を取った npz を固定利用する。 |
| シリアスな文が明るく聞こえてしまう | 感情リーク。参照クリップを「悲しみ + 喜び」の両方を含む混合文で再選定する。 |
| 長文で語が飛ぶ・同じ単語を繰り返す | モデルのワンショット限界。`speak_long_text.py` で 1 文 / 80 字単位に自動分割生成して結合する。 |

---

## 規約と倫理的利用

- 生成された音声モデル・音声ファイルは、実在人物のなりすましや同意のない模倣に使用しないこと。
- 生成音声を一般公開する場合は、AI 音声（Irodori-TTS）であることを明記すること。
- モデル重みファイルは各著作者・配布元のライセンスに従い、無断再配布を行わないこと。

