# 引き継ぎ: Gemma 4 31B 高速化を RTX PRO 6000 実機で試す

このファイルは、クラウドセッション（2026-10-24〜25）で行った調査を実機のローカルセッションに引き継ぐためのもの。詳細はすべて `gemma4-31b-rtx-pro-6000-speedup-report.md` にある。ローカルセッションの最初に「`HANDOFF-gemma4-speedup.md` と `gemma4-31b-rtx-pro-6000-speedup-report.md` を読んで、§1 の計測から始めて」と指示すればよい。

## 0. 前提（確定事項）
- 目的: Gemma 4 31B（dense）の日本語・北摂方言の品質を**一切落とさず**に、llama-server の単一ストリームと複数スロットの合計スループットを上げる。モデル切り替えは対象外（レポート付録 C）。
- GPU: RTX PRO 6000 Blackwell（96 GB、1,792 GB/s、SM120）。CUDA 12.8/12.9 でビルド（13.x は遅くなる報告あり、未確認）。
- 現状: llama-server、`--parallel N` で複数スロット、温度サンプリング、長いペルソナ。
- 公開実測の現状値: llama.cpp Q4 で 51〜61 tok/s（上限 104、現実的天井 90〜96）。

## 1. 最初に測ること（半日。順番どおりに）
すべて `llama-server`／`llama-bench`／`llama-batched-bench` だけで済む。結果はこのファイルの末尾「計測ログ」に追記する。

### 1-1. ビルド確認
```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j
./build/bin/llama-bench -m <QAT_Q4_0.gguf> -fa 1 -p 0 -n 128 -r 5
```
期待: Q4_0 で 55〜65 tok/s。50 未満なら CUDA ツールキット版と `GGML_CUDA_FORCE_CUBLAS` が付いていないかを疑う。

### 1-2. スロット歯抜けテスト（最重要。レポート §3.1(A)）
`-np 8`（`--kv-unified` なし）でサーバを起き、`id_slot` を指定して同時生成:
1. スロット {0,1} → 2. {0,2} → 3. {0,2,5}。合計 tok/s を比較。
- 2〜3 倍落ちれば「非 unified KV の ubatch 分割」に該当。`--kv-unified` を付けて再測定（同時 1〜8、`--spec-type` なし）。
- `GGML_CUDA_GRAPH_DEBUG=1` でウォームアップ再設定が毎ステップ出るかも確認。

### 1-3. 同時トークン数に対するステップ時間
```
./build/bin/llama-batched-bench -m <gguf> -c 70656 -b 2048 -ub 512 -fa 1 -npp 2048 -ntg 128 -npl 1,2,4,5,6,8,9,12,16,32
```
`-kvu` あり／なし、`-ub 256` でも実行。T_TG(npl=8)/T_TG(npl=1) が 1.6 超なら MMVQ→MMQ の崖あり。Q4_K_M は 6 列、Q4_0 は 9 列で切り替わる。

### 1-4. MTP の受理率とドラフト 1 位置のコスト d
```
llama-server -m <QAT_Q4_0.gguf> -md <mtp-gemma-4-31B-it.gguf> --spec-type draft-mtp \
  --spec-draft-n-max K -fa on -ctk f16 -ctv f16 --jinja -np 1
```
K=1..4、自分の日本語ログ 50 プロンプト、T=0 と本番設定（T≈0.9、min-p）で、`/metrics` の `llamacpp:spec_decode_num_accepted_tokens_per_pos_total` を読む。`T_step = a + K·d` を当てはめて d を求める。
- `--spec-draft-sampling greedy` と `probabilistic` を比較（温度ありでは probabilistic が有利な見込み）。
- d ≥ 2 ms なら MTP ループの overhead 削減（レポート §8 案 6／§8.3）が単一ストリームの最大レバー。

### 1-5. `-bs`（GPU サンプリング）のパリティと効果
c=8、K=2 で `-bs` あり／なしの tok/s と CPU 使用率。注意: 投機併用時に `repeat_penalty` が 1.0 以外だと GPU サンプリングが無効になる（ソース確認済み）。DRY/XTC/typical/grammar/n_probs も無効化要因。ログの fallback 警告を確認。

### 1-6. 方言コーパスの KL ハーネス（量子化判定用。レポート付録 C.2）
BF16 GGUF を基準に `llama-perplexity --kl-divergence-base` → 候補（QAT Q4_0、Q6_K、Q8_0）を `--kl-divergence`。コーパスは北摂方言対話・地方文化 QA・混在文 1,000〜2,000 サンプル。基準 logits は約 1 MiB/トークンなのでディスクに注意。

## 2. 判断の目安
- 歯抜けテストで 1.5 倍以上落ちる → `--kv-unified` を暫定採用（ただし Q 行数 8 超で KV 多重走査の副作用、要実測）。根本修正は改造案 16。
- GPU 稼働率 < 0.85（nsys、batch 1） → `-bs`、グラフのバケット化、MTP ループ改善。
- 方言コーパスで Q4_0 に劣化 → Q6_K ＋ MTP（Q4_0 ＋ 投機なしより速い見込み）。
- 投機の K: 同時 1〜2 で 3、4 で 2、8 で 1〜2、16 で 0〜1 から始める。

## 3. 推奨コマンド（出発点。レポート §3.3）
```
# 同時 1〜4 人、MTP あり
llama-server -hf google/gemma-4-31B-it-qat-q4_0-gguf \
  -md mtp-gemma-4-31B-it.gguf --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-p-min 0.3 --spec-draft-sampling probabilistic \
  -fa on -ctk f16 -ctv f16 --jinja \
  --parallel 4 -c 65536 -b 512 -ub 512 \
  --ctx-checkpoints 32 --checkpoint-min-step 1024 --cache-ram 32768 --cache-idle-slots \
  --threads-http 8 -bs
# 同時 8〜16 人、合計重視
llama-server -hf google/gemma-4-31B-it-qat-q4_0-gguf \
  -fa on -ctk f16 -ctv f16 --jinja \
  --parallel 16 -c 131072 -b 512 -ub 256 \
  --ctx-checkpoints 32 --checkpoint-min-step 1024 --cache-ram 65536 --cache-idle-slots \
  --threads-http 16 -bs
```
注意: `--cache-reuse` は Gemma 4 では `--swa-full` なしだと無効。`--swa-full` は 16×8K で f16 105 GB になるので使わない。KV は f16 のまま。

## 4. その後の改造候補（レポート §8.3、審査済み順）
1. ubatch 分割の根本修正（FA に系列→ストリーム添字配列、`split_equal(sequential=false)`）
2. GPU 常駐の棄却サンプラ（ペナルティ付きでも GPU 化）
3. CUDA グラフのバケット化（デコード ubatch を 8/16/32/64 にパディング）
4. MTP ループの K 段グラフ展開（ホスト往復排除）
5. 北摂方言を含む自前ログで MTP ドラフタを追加学習（無損失、受理率向上）
6. K==V グローバル層の KV 圧縮（640/512、Q 側に fold、厳密）
7. MMA FA カーネルへの q8_0 KV ネイティブ対応（複数スロットの KV 読み出し削減）

## 5. 未確認（実機で埋める）
歯抜け分割の実効果、日本語（温度あり）の MTP 受理率、d の実測値、`-bs` のパリティ、Q4_0 の方言劣化、`--kv-unified` の副作用、CUDA 13.x ビルドの速度差。

## 計測ログ
（ローカルセッションで追記）
