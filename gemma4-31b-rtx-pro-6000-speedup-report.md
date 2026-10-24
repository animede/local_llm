# Gemma 4 31B（dense）を RTX PRO 6000 Blackwell 1枚で llama.cpp より速く動かす — 調査レポート

作成日: 2026-10-24。対象: `google/gemma-4-31B-it`、RTX PRO 6000 Blackwell（96 GB、1,792 GB/s、SM120）、llama-server 複数スロット運用、日本語 AI キャラクターチャット（温度サンプリング、固定の長いペルソナ、同時 2〜16 ユーザー）。

凡例: 【確認】＝ソースコードまたは一次資料で確認、【実測】＝第三者の公開実測（条件付き）、【推定】＝計算モデルや推論に基づく見積もり、【未確認】＝裏が取れていない。

---

## 0. 結論（先に要点）

1. **現状の llama.cpp は上限の 40〜65% しか出ていない。** 4bit 重み 17.3 GB を毎トークン読む帯域律速で理論上限は約 100 tok/s（現実的に 90〜96）。公開実測は llama.cpp Q4 で 51〜61 tok/s、vLLM NVFP4 で 41〜51 tok/s。つまり「エンジンを替える」前に「使い方を変える」余地が大きい。
2. **llama.cpp のまま今週できることが 3 つある（ソース確認済み）。**
   - `--parallel N` を明示すると既定で非 unified KV になり、生成中スロットの番号が飛び飛び（例 {0,3,5}）だと 1 ステップが複数の ubatch に分割され、**重みを複数回読む**。しかも形状が毎ステップ変わるので CUDA グラフが効かない。8 スロットで 3〜4 人同時の典型状況では合計スループットが 2〜2.5 倍落ちている可能性がある【確認（コード）／効果量は要実測】。
   - 262,144 語彙のサンプリングが既定で CPU・逐次実行（約 0.4〜0.5 ms/行）。投機的デコードでは行数が「受理数＋1」倍になり、同時 8〜16 ではステップ時間の 20〜30% を占める【確認（コード）＋ RTX 5090・32 スロットで +48% の実測】。複数スロットで投機の利得が消えるのは、この CPU サンプリング、検証行の計算、ドラフタのループ overhead の 3 つが同程度に積み上がるため【推定】。`--backend-sampling`（`-bs`）で GPU 化できるが、**投機併用時は繰り返しペナルティが GPU を拒否し**、DRY/XTC/typical も非対応なので CPU に戻る制約がある。
   - Google 公式の MTP ドラフタ（`gemma-4-31B-it-assistant`、2026-06-07 に llama.cpp 本体へマージ）を `--spec-type draft-mtp` で使う。英語 greedy で 1.4〜2 倍、**日本語長文で 1.9 倍（受理率 55%）**の実測がある。温度ありなら `--spec-draft-sampling probabilistic` が必須級。
3. **同時 8〜16 ユーザー以上なら vLLM が合計スループットで 3〜5 倍**（16 クライアントで vLLM 1,025 対 llama.cpp 279 tok/s の第三者実測）。ただし SM120 では `--attention-backend FLASHINFER` などの明示設定が要り、vLLM 側も版ごとの退行がある。SGLang は同 GPU で vLLM より遅い報告が多く、TensorRT-LLM は Gemma 4 で KV cache reuse が使えないためチャット向きでない。
4. **単一ストリームの最高値は「vLLM ＋ NVFP4 ＋ DFlash/DSpark ドラフタ」**で、この GPU で 42 → 172 tok/s（greedy）、温度 1 で 152 tok/s の実測がある（vLLM 0.24 ＋ パッチ、K=15、英語）。日本語での受理率は未計測。
5. **改造案で筋が良いもの（審査上位）**: ① llama-server の ubatch 分割問題の根本修正（非連続ストリーム対応）、② GPU 常駐の検証・棄却サンプラ ＋ CUDA グラフのバケット化、③ 自分の日本語ログでドラフタを追加学習（無損失で受理率を直接上げる唯一の手段）、④ Gemma 4 固有の「K==V グローバル層の KV 圧縮（厳密、−37.5%）」、⑤ MMA アテンションカーネルへの q8_0 KV ネイティブ対応（複数スロットで KV 読み出しが重みを上回るため）。
6. **やらない方がよいもの**: 活性化スパース化、自己投機、深さ刈り込み、2:4 スパース、L2 常駐、SageAttention3、31B の拡散化、ペルソナのソフトトークン化、exllamav3 への全面移行、NVIDIA 公式 NVFP4 チェックポイント（32.7 GB、アテンションは BF16 のままで FP8 級の速度）。

---

## 1. 前提と理論上限

### 1.1 アーキテクチャ（transformers / vLLM / llama.cpp のソースで確認）
- 60 層 ＝ スライディング窓（1,024）50 層 ＋ グローバル 10 層（5:1、最終層はグローバル）。hidden 5,376、FFN 21,504（GeGLU）、32 ヘッド。
- ローカル層 KV: 16 ヘッド × 256。グローバル層 KV: **4 ヘッド × 512、K と V が同一射影**（`attention_k_eq_v`、v_proj なし。K は重み付き RMSNorm ＋ 部分 RoPE 128 次元、V は重みなし RMSNorm）。どのエンジンも KV は K・V 別々に保存している。
- 語彙 262,144、埋め込みは tied（lm_head だけで 1.41B パラメータ、毎トークン全読み）。最終 logit softcap 30。アテンション softcap なし。
- パラメータ: ローカル層 478.9M × 50 ＋ グローバル層 534.0M × 10 ＝ 29.29B ＋ 1.41B ＝ 30.70B（テキスト部）。視覚塔 約 550M は分離可能（llama.cpp は mmproj を付けなければ 0）。
- トークナイザは Gemma 3 と同一の 262k。日本語は **1 文字 ≈ 0.6〜0.7 トークン**（会話文 1.65 字/トークン、小説 1.43〜1.53）。200 字の返答 ≈ 120〜140 デコードステップ。英語の 2.7〜3 倍のステップ数が要る。

### 1.2 1 トークンあたりの読み出し量と帯域上限（1,792 GB/s）
| 重み形式 | GB/token | 理論上限 tok/s | 現実的到達（75%） |
|---|---|---|---|
| BF16 | 61.4 | 29 | 22 |
| FP8 / Q8_0 | 31〜33 | 55〜58 | 42 |
| Q6_K | 25.2 | 71 | 53 |
| Q4_K_M（実ファイル 18.3〜19.6 GB） | 18.3〜19.6 | 91〜98 | 70 |
| NVFP4 全層 / Q4_0 / Google QAT Q4_0（4.5 bpw） | 17.3 | 104 | 78 |
| IQ4_XS | 16.3 | 110 | 82 |
| EXL3 3.0 bpw（head 8 bit） | 12.4 | 145 | 108 |

DRAM の実効上限は仕様の約 93%（RTX 5090 の単一カーネル実験で 1,674 GB/s）なので、4bit の現実的な天井は **90〜96 tok/s**。

### 1.3 KV キャッシュ
- ローカル層: 16 KiB/token（f16）× 50 層。窓 1,024 で頭打ち → 理想 0.8 GiB/系列。**llama.cpp は `n_swa + n_ubatch` セル（`-ub 512` で 1,536 セル）を確保し、リングが回るので毎ステップ 1.1〜1.2 GB を読む**【確認】。`-ub 256` で 1,280 セルに縮む。
- グローバル層: 80 KiB/token（f16）。8K で 0.64 GiB、32K で 2.5 GiB、128K で 10 GiB／系列。
- **同時 N 系列のデコード 1 ステップ**: 重み 17.3 GB は 1 回、KV は系列ごと（8K で約 1.5〜1.8 GB）。N=16 なら KV 読み出し 24〜29 GB ＞ 重み。同時数が多いほど KV の精度・レイアウトが合計スループットを決める。
- 容量（4bit 重み、6 GiB 予備）: 8K コンテキストで約 50 系列（f16 KV）、32K で約 22 系列。FP8 KV なら倍。

### 1.4 公開実測（RTX PRO 6000 単体、投機なし）
| スタック | 単一ストリーム | 並列 | 出典・条件 |
|---|---|---|---|
| llama.cpp Q4 | 51〜61 tok/s（4K ctx）、34（256K） | Q8_0 で 16 並列 279、32 並列 344 | hardware-corner、llamaperf、truespar（ベンダ計測） |
| Ollama Q4_K_M | 58〜64 | 4 並列で合計 58 | allenkuo、petronella |
| vLLM NVFP4 | 40.7（2026-03、Marlin/Triton フォールバック期）、51（SM120 パッチ）、42.3（0.24） | 4 並列 126 | MillstoneAI、lna-lab、hikarioyama |
| vLLM FP8 | 36.4 | 4 並列 129、16 並列 1,025、32 並列 1,615、ピーク 1.17k | MillstoneAI、truespar、HexGrid |
| SGLang FP8 | 45 | 16 並列 498 | truespar |
| vLLM NVFP4 ＋ DFlash/DSpark | **171.7（greedy）／151.6（T=1）** | — | hikarioyama、vLLM 0.24＋パッチ、K=15、bf16 KV、`nvidia/Gemma-4-31B-IT-NVFP4` |
| llama.cpp ＋ MTP | Q8: 約 40 → 約 83；B200 Q4_K_M: 70 → 101（受理 0.78） | — | dev.to、unsloth |

どのスタックも上限の 40〜65%。(a) カーネル／サーバ効率で 1.3〜1.5 倍、(b) 投機的デコードで 1.5〜2 倍、(c) 重みのビット削減で 1.1〜1.2 倍の余地が独立に存在する。

---

## 2. SM120（RTX PRO 6000）特有の落とし穴

RTX PRO 6000 は「Blackwell」でもデータセンター向け B200（SM100）とは別物（SM120、RTX 5090 と同系）。

- **SM120 に無いもの**: tcgen05/UMMA、TMEM、2-CTA MMA、TMA マルチキャスト、wgmma。共有メモリ 99 KB/ブロック（B200 は 227 KB）。FP4/FP8 は warp 単位の `mma.sync`（`sm_120a` ターゲット必須、CUDA 12.8 以上）。
- **SM120 で動かないもの**: FlashAttention-3/4（vLLM は FA4 を SM100 限定、FA2 は head_dim 256 まで）、TensorRT-LLM の trtllm-gen カーネル（「SM120 対応予定なし」2026-06）、CUTLASS SM100 NVFP4 GEMM、Machete、DeepGEMM、FlashMLA。
- **Gemma 4 固有**: グローバル層が head_dim 512 なので FA2 系が使えない。vLLM は Gemma 4 に対し **バックエンド未指定のときだけ** TRITON_ATTN（最遅）を強制する（`Gemma4Config.verify_and_update_config`）。`--attention-backend FLASHINFER`（テキスト専用、FA2 prefill ＋ XQA decode、head 512 対応、FP8/NVFP4 KV 対応）または `TRITON_FLASHINFER`（画像入力あり）を明示する【確認】。ただし SM12x では head 512 のグループがあると FlashInfer の CUDA グラフが単一トークンデコード限定になり、投機の検証ステップに税がかかる【確認】。
- **NVFP4 GEMM**: vLLM 既定は FlashInfer CUTLASS。SM120 専用 b12x カーネル（小バッチで 1.2〜1.6 倍、M=9 で HBM 帯域 63 → 89%）は `--linear-backend flashinfer_b12x` で明示。MXFP4 は今も Marlin W4A16 にフォールバック。
- **TensorRT-LLM**: SM120 は 0.20 から単 GPU 対応だが、Gemma 4 は KV cache reuse 非対応、Gemma NVFP4 on SM120 は「未対応」（2026-07-31）、NVFP4 KV 未提供、RC のみ。チャット用途では主候補にならない。NIM は Server Edition 用 NVFP4 プロファイルのみ、ライセンス要確認。
- **CUDA ツールキット**: 12.8/12.9 推奨。llama.cpp を CUDA 13.x でビルドして大幅に遅くなった／MMQ が落ちた報告あり【未確認（コミュニティ報告のみ）】。
- 2026 年 3〜5 月の vLLM 実測（NVFP4 40 tok/s 台）はフォールバック時代の数字。現行 vLLM 0.32 ＋ FlashInfer 0.7 での 31B 単体の新しい実測は見つからなかった。

---

## 3. llama.cpp のまま速くする（今週）

### 3.1 ソースで確認した 3 つの問題

**(A) 非 unified KV の ubatch 分割**【確認】
- `common.h` で `kv_unified = false` が既定。`server.cpp` は `n_parallel < 0`（auto）のときだけ `kv_unified = true` にする。つまり `--parallel 8` と明示すると非 unified。
- 非 unified では `llama_kv_cache_iswa::init_batch` が simple split をスキップし `split_equal(n_ubatch, sequential=true)` を使う。`split_equal` は **連続した seq id しか同じ ubatch に入れない**（`batch.seq_id[i][0] == last_seq_id + 1`）。理由は K/V のビューがストリームの連続範囲 [s0, s1] だから。
- サーバはスロットを LCP 類似度（既定 0.1 で有効）→ 空き → LRU で割り当てるので、生成中スロットが {0,3,5} のような歯抜けになるのが普通。その場合 1 ステップ ＝ 3 ubatch ＝ 重み 17 GB を 3 回読む forward。
- さらに CUDA グラフはノード形状が 2 回連続で一致しないとキャプチャされず、ubatch の形状が交互に変わると永遠にウォームアップ中（毎回 eager 起動、約 900〜1,000 カーネル）。
- 8 スロット・3〜4 人同時で期待される連続ランは 2.25〜2.5。合計スループットが本来の 40〜50% に落ちている可能性【推定】。
- 対策: 短期は `--kv-unified`（`-kvu`）。ただし unified では FA が統合バッファ全体のマスクを走査し、Q の行数が 8 を超える（同時 9 人以上、または MTP で c×(K+1) > 8）とグローバル層の KV を Q タイルごとに多重に読む副作用がある【確認】。**両方を実測して決める**（手順は §9）。根本修正は改造案（§8）。

**(B) CPU サンプリング**【確認】
- `post_decode()` でスロットごとに逐次、262,144 要素の logits を D2H コピーしてソフトマックス・top-p 等を CPU で実行。1 行あたり約 0.45 ms（RTX 5090・32 スロット・`-bs` で 706 → 1,046 tok/s、issue #27050 からの逆算）。
- 投機的デコードでは行数が c×τ（τ ＝ 受理数＋1、最初の不一致で打ち切り）。c=16・τ=2 なら 32 行 × 0.4 ms ≈ 13 ms が直列に乗る（計算モデルでは検証行の MMQ 計算 8〜26 ms、ドラフタループ K·d 3〜18 ms と同程度）。
- `-bs`（`--backend-sampling`）で GPU 側へ。対応サンプラは greedy / dist / top_k / top_p / min_p / temp / penalties / logit_bias。**typical、XTC、DRY、mirostat、top-n-sigma、grammar は非対応で、チェーンの先頭から「対応サンプラが続く範囲」だけが GPU 化される**（途中に非対応があると以降は全部 CPU）。`n_probs > 0`（post_sampling_probs なし）でも無効。
- **投機（MTP 等）併用時、`repeat_penalty` などのペナルティは `n_outputs_max_per_seq > 1` だと GPU を拒否する**（`llama_sampler_penalties_backend_init`）。ペナルティはチェーン先頭なので、投機 ＋ ペナルティ ＝ 実質 CPU サンプリングに戻る。`repeat_penalty 1.0`（無効）にすればペナルティサンプラは no-op になり GPU 化できる。キャラチャットで DRY/XTC を使っているなら、ここはトレードオフ。

**(C) 量子化 KV は毎ステップ f16 に変換される**【確認】
- CUDA の FlashAttention で量子化 KV（q8_0 等）をネイティブに読めるのはベクトルカーネルだけ。ベクトルカーネルは head_dim ≤ 256 かつ Q 行数 ≤ 2（ストリームあたり）のときだけ選ばれる。Gemma 4 のグローバル層（head 512）は常に MMA カーネル、複数行（unified の複数スロット、MTP の検証）も MMA/tile → `need_f16_K/V` で **K・V 全体を毎回 f16 に変換**（`fattn-common.cuh` の `launch_fattn`）。
- 96 GB ある環境で量子化 KV を使う理由はない。`-ctk f16 -ctv f16`（既定）のままにする。q8_0 KV で MTP の受理率が 0 になった報告もある。
- 例外: 非 unified で 1 行デコードならローカル層（head 256）は q8_0 をネイティブに読める。`--cache-type-k-swa` のようなキャッシュ別指定は未実装（改造案）。

### 3.2 その他の確認事項
- **`--cache-reuse` は Gemma 3/4 では `--swa-full` を付けない限り無効**（iSWA キャッシュの `get_can_shift()` が両キャッシュ同サイズを要求）。`--swa-full` は 50 層分の KV が窓制限なしになり 16 スロット × 8K で f16 105 GB → 非現実的。代わりにコンテキストチェックポイント（`--ctx-checkpoints`、`--checkpoint-min-step`）で復元される。同一スロットの会話継続は安い。ペルソナ境界（約 2K）での再利用を狙うなら `--checkpoint-min-step 512〜1024`【推定】。
- **プロンプト処理が全スロットのデコードを止める**: 1 ステップ ＝ 各生成スロット 1 トークン ＋ 新規プロンプトの `-b`（既定 2,048）チャンク。新規ユーザー参加のたびに他ユーザーが 0.45 秒止まる。`-b 512` で 0.12 秒に（prefill 効率はほぼ不変、ubatch は 512 のまま）。
- MMVQ（小バッチのデコードカーネル）→ MMQ（テンソルコア）の切り替え点は CC 12.0 向けに調整済み: **Q4_K/Q3_K は 5 列、Q5_K 6、Q6_K 7、Q4_0/NVFP4/Q8_0/IQ4_XS は 8 列**。Q4_K_M だと同時 6 トークンで MMQ 側に落ちる。Q4_0 系はこの点でも有利。
- NVFP4/MXFP4 GGUF は Blackwell では **既定で W4A4**（活性化も FP4）の MMQ 経路（prefill +20〜68%）。9 行以上の検証バッチも W4A4 になるので品質が揺れる。`GGML_CUDA_MMQ_PREC=q8` で W4A8 に戻せる。
- CUDA グラフと PDL（Programmatic Dependent Launch）は既に有効（`GGML_CUDA_PDL=0` で無効化可）。MMQ と サンプリング系カーネルには PDL が入っていない。gate/up＋GELU の融合は列数 1 のときだけ。
- 異なる LoRA を使うリクエストは同じバッチに入らない。キャラごとに LoRA を分けると並列性が落ちる。
- llama.cpp にしか無いもの: DRY／XTC サンプラ（vLLM・SGLang・TRT-LLM に無い。TabbyAPI にはある）、`/slots` の save/restore、マルチモデルルータ。

### 3.3 推奨構成（出発点。すべて要実測）

重み: Google 公式 QAT Q4_0（`google/gemma-4-31B-it-qat-q4_0-gguf`、17.7 GB）または Unsloth QAT UD-Q4_K_XL。Q4_0 は CUDA で最も軽い 4bit 型で、直接比較では Q4_K_M より 10〜30% 速く、QAT で品質を回復している。日本語での文字化け無しの報告あり。

```
# 同時 1〜4 人を速く（MTP あり）
llama-server -hf google/gemma-4-31B-it-qat-q4_0-gguf \
  -md mtp-gemma-4-31B-it.gguf --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-p-min 0.3 --spec-draft-sampling probabilistic \
  -fa on -ctk f16 -ctv f16 --jinja \
  --parallel 4 -c 65536 -b 512 -ub 512 \
  --ctx-checkpoints 32 --checkpoint-min-step 1024 --cache-ram 32768 --cache-idle-slots \
  --threads-http 8 -bs
# -bs を活かすなら repeat_penalty=1.0（DRY/XTC/typical も不可）。ペナルティが必要なら -bs を外す。
# QAT リポジトリには MTP GGUF が無いので unsloth 等のドラフタ GGUF（Q8_0、515 MB）を -md で指定。

# 同時 8〜16 人の合計スループット重視
llama-server -hf google/gemma-4-31B-it-qat-q4_0-gguf \
  -fa on -ctk f16 -ctv f16 --jinja \
  --parallel 16 -c 131072 -b 512 -ub 256 \
  --ctx-checkpoints 32 --checkpoint-min-step 1024 --cache-ram 65536 --cache-idle-slots \
  --threads-http 16 -bs
# --kv-unified の有無、--spec-type draft-mtp --spec-draft-n-max 1 の有無は §9 の手順で実測して決める。
```

ビルド: CUDA 12.8/12.9、`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`（自動で 120a）。

既知の不具合: `--parallel 8` ＋ MTP ＋ tensor split で 2 回目のプロンプトでクラッシュ（#25522）、`--kv-unified` ＋ MTP ＋ 複数スロットで失敗（#30368、別モデル）、SWA チェックポイント＋FA のメモリ違反（#22527）、unified の prompt cache 周り（#30961、#30252）。複数スロットで MTP を使う場合は最新 master で再現確認。

---

## 4. エンジン別比較と移行判断

| エンジン | Gemma 4 対応 | SM120 のアテンション | 量子化 | 投機 | 複数ユーザー／prefix cache | 実測（単一／合計） | 移行コスト |
|---|---|---|---|---|---|---|---|
| llama.cpp | ○（MTP／EAGLE-3／DFlash、複数スロット MTP に未解決 issue） | 自前 FA（head 512 対応） | Q4_K_M、QAT Q4_0、NVFP4/MXFP4（W4A4 prefill） | スロット単位 | スロット、prefill が全員を止める、非 unified の分割問題 | Q4 51〜61／Q8 16 並列 279 | なし |
| vLLM 0.29〜0.32 | ○（Day-0、MTP は CUDA graph 対応。版ごとの退行あり） | 既定 TRITON_ATTN → FLASHINFER を明示 | FP8、NVFP4（CUTLASS/b12x）、AWQ/GPTQ-Marlin、QAT W4A16、FP8/NVFP4 KV | MTP、EAGLE-3、DFlash、DSpark、同時数別ドラフト長 | 最良。ハイブリッド KV マネージャ、prefix cache（多セッションでヒット率崩壊の報告 #48435） | NVFP4 41〜51、FP8 36／16 並列 1,025、ピーク 1.17k。5090＋MTP 160〜215 | 中（cu130 イメージ、FlashInfer JIT、版固定） |
| SGLang | ○（NEXTN、31B の LoRA 故障中） | flashinfer（FA2）／triton | FP8、modelopt_fp4、AWQ | NEXTN、EAGLE | RadixAttention。既定 `--swa-full-tokens-ratio 0.8` は窓 1,024 に過大 → 0.2 に | FP8 45／16 並列 498 | 中〜高 |
| TensorRT-LLM 1.3rc〜1.4 | ○（RC のみ、**KV reuse 非対応**、ハング報告） | FlashInfer FA2 に退避 | FP8、NVFP4（数値不正 issue open） | MTP（draft 3） | prefix cache 無し | Gemma 4 の SM120 実測なし | 高 |
| exllamav3 1.6 ＋ TabbyAPI | ○（EXL3 2〜6 bpw） | 自前 Triton | EXL3（KLD 最良） | ドラフトモデル | 動的バッチ、prefill 弱い | Gemma 4 の Blackwell 実測なし。Qwen3.8-27B で vLLM 比 C1 86%、C8 55% | 中 |
| Paddock（Rust、新興） | ○（ベンダ公称） | 自前 | FP8/NVFP4/GGUF | 不明 | radix cache | 1 並列 100.8、16 並列 1,011（投機込みの可能性） | 未成熟 |
| Ollama | ○ | GGML | GGUF | MTP は MLX 先行 | 弱い | Q4 58〜64、4 並列で合計 58 | 利点なし |
| LMDeploy／MLC／Aphrodite／mistral.rs | ×／×／×／△ | — | — | — | — | — | 対象外 |

判断基準:
- **同時 ≤ 4 で単一ストリームの速さ**: llama.cpp（§3 の是正 ＋ MTP）で 100 tok/s 前後【推定】まで行ける見込み。vLLM ＋ DFlash/DSpark は 150〜170 tok/s の実測があるが日本語受理率は未知。
- **同時 8〜16 の合計スループット**: vLLM（prefix cache、連続バッチ、GPU サンプリング、FP8 KV）。llama.cpp は §3 の是正後でも、KV 読み出し（f16 のみ）と MMQ 経路の効率で不利。
- **サンプラ互換**: DRY/XTC/typical が必要なら llama.cpp か TabbyAPI しか無い。

vLLM の出発点（テキスト専用）:
```
vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
  --attention-backend FLASHINFER --language-model-only \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --enable-prefix-caching --async-scheduling \
  --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}'
# NVFP4 重みなら --linear-backend flashinfer_b12x、KV は --kv-cache-dtype fp8（日本語 A/B 後）。
# 同時数別ドラフト長: num_speculative_tokens_per_batch_size [[1,4,3],[5,8,2],[9,16,1]]
# 既定の max_num_batched_tokens 8192 は ITL が荒れるので 2048 に。
```

---

## 5. 量子化の選び方（速度優先・日本語チャット）

1. **Google QAT Q4_0 GGUF**（17.7 GB）／Unsloth QAT UD-Q4_K_XL — Q4_0 は CUDA の MMVQ で最速クラス、bytes/token は NVFP4 と同じ、MMVQ の切替点も 8 列。QAT で品質回復。
2. **NVFP4 GGUF**（`CISCai/gemma-4-31B-it-NVFP4-turbo-GGUF`、19.3 GB）— 元は NVIDIA の NVFP4（MLP のみ FP4）にアテンションを RTN で FP4 化したもの。埋め込みが BF16（2.8 GB）のまま残るため **デコードは Q4_0 より遅い**。利点は W4A4 の prefill（+35〜68%）と 9 行以上の検証バッチ。アテンション FP4 の日本語品質データ無し。llama-quantize には NVFP4 プリセットが無く、imatrix も無視される。
3. **vLLM ＋ QAT W4A16**（`google/gemma-4-31B-it-qat-w4a16-ct`、19.8 GB）または AWQ — Marlin W4A16 のデコードは同 GPU で NVFP4/FP8 より速い報告（Qwen3.6-27B: AWQ 69.0、NVFP4 55.6、FP8 48.8 tok/s、vLLM 0.22）。
4. **NVIDIA 公式 `nvidia/Gemma-4-31B-IT-NVFP4`（32.7 GB）はアテンションが BF16 のまま**。デコード性能は FP8 級（上限約 55 tok/s）。「4bit」と思って使うと期待外れ。
5. EXL3 3.0〜3.5 bpw は理論上限が最も高い（約 145 tok/s）が、実測デコードは llama.cpp に劣り、SWA メモリ問題・SM120 不具合報告があり、現時点では研究用途。

品質の目安（31B、MMLU-Pro の BF16 比低下）: FP8 −0.7、AWQ −1.0、NVFP4 −1.6（beri.net）。NVIDIA 版 NVFP4 の GPQA Diamond 85.80 → 85.35。EXL3 3.0 bpw の KL 0.03〜0.056。Gemma 系は 4.25 bpw 未満で急に劣化する傾向（naive Q4_0 の 26B QAT で KLD 0.36）。

日本語専用の評価ハーネス（§8 の多くの判断に必須）: BF16 を基準に自分のログ 15〜20 万トークンで `llama-perplexity --kl-divergence-base` → 各候補を `--kl-divergence`（平均 KLD ≤ QAT-Q4_0 の 1.3 倍、p99 ≤ 2 倍、top-1 一致 ≥ 97%）、BF16-31B を審判にしたペア比較（勝率 ≥ 47%）、U+FFFD 率 ＝ 0。ログ基準 logits は約 1 MiB/トークンなのでディスクに注意。

---

## 6. 投機的デコードの計画

### 6.1 ドラフタの選択肢
| 方式 | Gemma 4 31B 用の提供物 | エンジン | 実測 |
|---|---|---|---|
| MTP（Google 公式、0.5B、4 層 d=1024、ターゲットの KV を読む） | `google/gemma-4-31B-it-assistant`、GGUF は unsloth 等 | llama.cpp（`draft-mtp`）、vLLM（`mtp`、0.21 で動作、0.25.1 で退行、0.29 で CUDA graph 対応）、SGLang（NEXTN）、Ollama | 英語 greedy 位置別受理 95/88/82/75%、vLLM H100 c=1 3.1 倍；llama.cpp B200 1.44 倍；**日本語長文 1.91 倍（K=2 のドラフト受理率 54.8% ＝ 位置 1 で約 0.66、DGX Spark、NVFP4 Marlin 経由、greedy）** |
| EAGLE-3（RedHatAI、2B） | `RedHatAI/gemma-4-31B-it-speculator.eagle3` | vLLM、llama.cpp（`draft-eagle3`）、SGLang（fork） | 位置別受理 86/72/60/48%。同条件で MTP に劣る |
| DFlash（z-lab、ブロック拡散） | `z-lab/gemma-4-31B-it-DFlash`、RedHatAI 版 | vLLM（triton_attn／flash_attn ドラフトパス）、SGLang、llama.cpp は Gemma 4 用 GGUF 未確認 | **RTX PRO 6000: 42.3 → 171.7（greedy）／144.7（T=1）**、N=7 でピーク、N を増やすと急落 |
| DSpark（DeepSeek、DFlash＋Markov ヘッド＋信頼度ヘッド） | `RedHatAI/gemma-4-31B-it-speculator.dspark`、Hikari07jp 版 | vLLM（`enable_adaptive_verification`）、llama.cpp は Qwen3 backbone のみ | RTX PRO 6000 T=1: 151.6（DFlash 比 +4.7%） |
| n-gram（ドラフト不要） | — | llama.cpp `ngram-mod` 等、vLLM ngram/suffix | 新規性の高い雑談にはほぼ効かない（1.05〜1.1 倍） |

### 6.2 同時数との関係
- 帯域律速なら「K+1 トークンの検証」はほぼ無料だが、4bit 重みでは 1 ステップ約 30〜80 行（カーネル経路による）で計算律速に転じる。dense 31B には MoE のようなバッチ増による追加利得は無い。
- llama.cpp では §3.1(B) の CPU サンプリングが先に効く。`-bs` を入れた後の最適 K は同時数に対して平坦（c=16 でも K=2〜3 が最適という計算モデル）【推定】。
- 推奨の出発点: 同時 1〜2 → K=3〜4、4 → 2〜3、8 → 1〜2、16 → 0〜1。llama.cpp には負荷に応じた自動調整が無い（vLLM には `num_speculative_tokens_per_batch_size`）。
- 日本語チャットの受理率は英語ベンチより 20〜40 ポイント低い前提で、`--spec-draft-n-max` を 1→4 で掃引し SPEED-Bench（`tools/server/bench/speed-bench`、roleplay／multilingual カテゴリ、`--extra-inputs '{"temperature":0.9,"min_p":0.05}'`）と `/metrics` の位置別受理数で測る。
- 温度サンプリング時: 既定の greedy ドラフトは「サンプル結果がドラフトと一致したら受理」なので受理率が落ちる。`--spec-draft-sampling probabilistic` で棄却サンプリング（分布は不変）。ドラフト側チェーンは top-k 10 ＋温度だけで、ターゲットの min-p/top-p が反映されない（改造案）。

---

## 7. 複数スロット運用（KV 予算・prefix cache・サンプラ）

| エンジン | SWA 対応 prefix cache | KV 精度（SM120） | 予算の目安（4bit 重み、8K） |
|---|---|---|---|
| vLLM | ハイブリッド KV マネージャ（SWA 層は窓外ブロックを解放）。多セッションのラウンドロビンでヒット率崩壊の報告（#48435、#57610） | `--kv-cache-dtype fp8`、`nvfp4`（v0.32、SM12x）、`--kv-cache-dtype-skip-layers sliding_window` で層種別に | 約 42 系列（bf16 KV）／約 85（fp8） |
| SGLang | UnifiedRadixCache（FULL＋SWA）。`--swa-full-tokens-ratio` を 0.2 程度に | fp8、nvfp4 | 既定比率だと約 11 系列 |
| llama.cpp | チェックポイント方式（§3.2） | f16 推奨 | 16 × 8K: SWA 13.5 GB ＋ グローバル 10.5 GB |
| TensorRT-LLM | **非対応** | fp8 | — |
| exllamav3 | ページハッシュ＋SWA を再帰状態扱い（粗い） | k/v 2〜8 bit | TabbyAPI 既定 max_batch_size 4 の可能性 |

- vLLM の既定 `max_num_batched_tokens` はこの GPU で 8,192 → ITL が荒れるので 2,048 に。
- API 互換: SSE／stop／logprobs は全エンジン可。**DRY/XTC は llama.cpp と TabbyAPI のみ**。vLLM・SGLang は LoRA の per-request 選択可。
- 計測は aiperf が全エンジン共通でマルチターン対応: `aiperf profile --endpoint-type chat --streaming --num-prefix-prompts 5 --prefix-prompt-length 2048 --conversation-num 64 --conversation-turn-mean 10 --output-tokens-mean 220 --concurrency 1,2,4,8,16`。

---

## 8. 改造案・新手法（発案 4 本 → 審査 3 本の結果で順位付け）

審査の採点軸: 実現性、同時 1/4/8/16 での期待効果、品質安全性、新規性。以下は審査後の順位。期待効果は【推定】。

### 8.1 今週（設定・スクリプト）
1. **`-bs` GPU サンプリング**（§3.1B）。c=8 で +16〜19%、c=16 で +23%、投機併用時はそれ以上。ペナルティ／DRY/XTC との非互換に注意。
2. **`--spec-draft-sampling probabilistic`**。T≈0.9 で位置 1 の受理率が約 0.46 → 0.62 相当（計算モデル）。c=1 で +20〜30%。
3. **ubatch 分割の A/B**（§3.1A）。`--kv-unified` 有無 × 同時数で実測。該当すれば最大の一手。
4. **`-b 512`**（他ユーザーの停止 0.45 → 0.12 秒）、**`-ub 256`**（SWA リング 1,536 → 1,280 セル、ローカル層 KV 読み出し −15%）。
5. **ドラフト長の静的スケジュール**（同時数別 K）＋ `--spec-draft-p-min 0.3`。固定 K=3 比で c=16 +18%（CPU サンプリング時）。
6. **MTP ループの d（1 位置あたりコスト）を実測**: `-np 1 --spec-draft-n-max K`（K=1..4）で `T_step = a + K·d` を当てはめ。公開データ（B200 70 → 101、point.free 35 → 49）は d≈3〜6 ms を示唆（帯域から見た理論値は 0.7 以下）。d ≥ 2 ms なら同時 1〜4 での最大レバー（+20〜40%）で、ドラフタの K 段ループを 1 つの ggml グラフに展開してホスト往復を無くす改造（§8.3）が本命。審査 3 本とも「過小評価されている」と指摘。
7. **日本語 KL／審判ハーネス**（§5）。
8. **窓 1,024 → 768／512 の試験のみ**: `--override-kv gemma4.attention.sliding_window=int:512`（キー名は `gguf-dump` で確認）。c=8 で −12%、c=16 で −18% の KV 読み出し。無治療で KL が量子化の 1.5 倍以内なら採用、治療（LoRA）はしない（ペルソナ挙動の再検証とドラフタ再学習が要るため）。
9. **日本語 imatrix による品質調整**: QAT の BF16 ベース（unquantized 版）から日本語ログの imatrix で Q4_0 を作り直し、グローバル層 `attn_k`（K と V の源、計 110M）だけ Q8_0 に（+0.05 GB）。速度は QAT Q4_0 と同じで品質の上積み狙い。gate/up を 3.4 bpw に落とす積極案は、Gemma 系が 4.25 bpw 未満で急劣化する傾向と QAT を捨てる代償から却下（審査）。4.45 bpw 混合は QAT Q4_0 比で速度利得ほぼゼロ。

現実的な積み上げの見積もり（審査後、±30%、llama.cpp、8K コンテキスト）【推定】: 同時 1: 約 80 → 110〜125 tok/s（2・6・14）、同時 8 合計: 約 290 → 420〜480（1・2・5）、q8_0 KV カーネル（19）が入れば約 550、同時 16 合計: 約 360 → 550〜650、同 700 以上。26B-A4B 蒸留（15）だけがその先の 2 倍。

### 8.2 数日〜2 週間（小さなパッチ・学習）
10. **ドラフト側チェーンの分布一致**: `spec_retune` にスロットの top_k/top_p/min_p を渡す（半日）。min-p で消えるトークンをドラフトしなくなる。
11. **ドラフタの語彙刈り込み（無損失）**: 262k×1024 のヘッドを日本語保持集合（4〜8 万語 ＋ 全単一文字 ＋ バイト ＋ 制御）に縮小（FR-Spec／VocabTrim、ドラフタ側は検証で守られる）。ターゲット側の出力語彙刈り込み（−7%）は「ソフト刈り込み（logits を −inf 埋めの 262k に散布）」で 1〜2 週間、落とした質量 ε を実測（平均 <1e-4）。
12. **CUDA グラフのバケット化**: デコード ubatch を 8/16/32/64 トークンにダミーでパディングしてグラフを安定化（複数スロット＋MTP では毎ステップ形状が変わり実質 eager）。+10〜15%。
13. **キャッシュ別 KV 型**（`--cache-type-k-swa`）: iSWA の 2 キャッシュに別の型を渡せるようにする（数時間）。非 unified・1 行デコードでのみローカル層が q8_0 をネイティブに読める。
14. **オンポリシー日本語ドラフタの追加学習**: 自分のログの応答を **配備中の 4bit モデル・本番サンプラ設定** で再生成し、Google の MTP ヘッド（NeMo Automodel の `Gemma4WithDrafter`【未確認】、基盤凍結）または DFlash/DSpark（speculators、z-lab レシピ 6,000 ステップ lr 6e-4、ウォームスタート）を学習。受理率 +0.10 で全同時数 +12〜16%、上振れで +30%。30〜50 GPU 時間、初回は 2〜4 週間。無損失。量子化ファイル・サンプラ・ペルソナを変えたら再学習。
15. **26B-A4B へのペルソナ蒸留（キャラ単位で合否）**: 160 対 51 tok/s の実測差（vLLM）。llama.cpp の MoE 経路は起動律速で単一 2〜2.5 倍、c=16 では触るエキスパートが 82/128 になり重み差 1.7 倍程度。まず **素の 26B-A4B-it を同じペルソナで 200 文脈に当て、人手ブラインド勝率が 31B 比 40% 以上なら** 蒸留（実ユーザー発話 ＋ 31B 再生成の応答で LoRA、20〜30 GPU 時間）に進み、25% 未満なら中止。審判を 31B にすると自己選好バイアスが入るので人手ペア比較（キャラあたり 200 以上）・ペルソナ事実 QA 50 問・20 ターン一貫性で合否。合格キャラだけ静的に切替、残りは 31B。

### 8.3 数週間（カーネル・エンジン改造。AI 発案の中核）
16. **ubatch 分割の根本修正**: FA に系列ごとのストリーム添字配列を渡す（カーネルが `K + strm[z]*nb03` で gather）か、`build_attn` でストリームごとに FA を発行して結合し、`split_equal(sequential=false)` にする。スロット番号の圧縮やストリーム再写像では解決しない（制約は K/V ビューと FA のバッチ→ストリーム対応にある）。
17. **GPU 常駐の棄却サンプラ**: 棄却判定と残差サンプリングを ggml のバックエンドサンプラとして実装し、受理トークン ID だけ返す。ペナルティ付きでも GPU 化できる設計に。3〜5 日〜。
18. **K==V グローバル層の KV 圧縮（厳密、−37.5%）**: 1 ヘッドあたり `[V(512) | K_rot(128)]` の 640 幅 1 本を保存し、k_norm の重み (1+w) は Q 側に掛ける（除算を避ける）。llama.cpp の MLA 用 `V_is_K_view`（DKQ 576/DV 512）と同じ機構を (640,512) で実体化。注意: llama.cpp はグローバル層を NEOX RoPE・n_rot 512・周波数係数 `[1.0]×64 + [1e30]×192` で実装しているので、回転する 128 次元は {0..63} ∪ {256..319} と非連続（gather が要る）。保存する V は重みなし正規化ベクトルなので q8_0/FP8 化（案 19）とも相性が良い。MTP ドラフタの Q にも同じ fold が必要。技術レポートの 37.5% をどのエンジンも未実装。4〜12K では数 %、32K で +20%、128K で +41%、32K の同時系列 22 → 31。1〜2 週間、代数的に厳密（丸め順序のみ変化）。
19. **MMA FA カーネルへの q8_0 KV ネイティブ読み込み**: `flash_attn_ext_f16_load_tile` にレジスタ内逆量子化（PR #30792 の tile 版が雛形）。head 512（グローバル）と複数行 Q で効く。c=8・8K で +21〜25%、32K で +37〜39%、同時系列 2 倍。4〜8 週間＋上流審査。
20. **MMVQ/MMQ の隙間カーネル（M=2..32）**: 活性化タイルを共有メモリに置き 64 行ブロックで再利用、`mma.sync m16n8k32.s32`。c=8 +10〜13%、c=16 +18〜25%。3〜5 週間以上。まず `llama-batched-bench -npl 1,2,4,5,6,8,9,12,16` で崖の有無を確認。
21. **PDL を MMQ／サンプリング系カーネルへ、MMVQ の `pdl_sync` を重みプリフェッチ後へ移動、残りの融合**: +5〜10%（batch 1）。1〜2 週間。
22. **vLLM: KV グループ別 dtype**（グローバル NVFP4／SWA FP8）— `--kv-cache-dtype-skip-layers` で二段階は既にできる。head 512 の XQA 対応が SM12x に来るまで速度利得は薄い（容量のみ）。
23. **永続デコード・メガカーネル**: 31B・4bit では帯域以外の時間が 1 トークン 3〜5 ms しか無く、上記 16〜21 の後の増分は +10〜13%。2〜4 か月。最後。

### 8.4 却下（理由）
活性化スパース化（バッチの和集合で密になる、GeGLU 未検証）／自己投機（4bit でも 60% の重みを読む）／深さ刈り込み＋治療（dense は崖、治療 20〜40 GPU 日）／2:4 スパース FP4（実効 3 bpw、再学習必須）／L2 常駐（128 MB 対 17 GB）／lm_head を最後の位置だけ（不正確かつ 1〜2%）／出力語彙の「硬い」刈り込み（稀な入力のトークン化が変わる。異体字の名前や絵文字が壊れる）／セントロイド近似の lm_head（稀トークンの再現率問題）／ツリードラフト（c≤2 でのみ +13%）／SageAttention3（head 128 prefill 専用）／31B の拡散化／ペルソナのソフトトークン化（−1〜9% の代償に人格忠実度の大きなリスク）／窓 512 への「治療」（50 層の学習時条件を変える。試験だけ行い、`-ub 256` で読み出し削減の一部を無料で得る）／KV 量子化の QAT と vLLM の層種別 NVFP4 KV（SM120 に速度に変わるカーネルが無い、容量はこの環境で余っている）／unified KV でのペルソナセル共有（MTP や c>8 では多重走査で逆効果）／3.4 bpw の積極量子化（QAT を捨てて +8%）／exllamav3 への全面移行（c=8 で vLLM の 55%、Gemma 4 用 MTP 無し）。

---

## 9. 計測プロトコル（作る前に測る）

```bash
# 0. ビルド（CUDA 12.8/12.9）
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j

# 1. 単一ストリームの天井と現実（Q4_0 の床は約 97 tok/s）
./build/bin/llama-bench -m gemma-4-31B_q4_0-it.gguf -fa 1 -p 0 -n 128 -r 5
GGML_CUDA_PDL=0 ./build/bin/llama-bench ...            # PDL の効き
GGML_CUDA_DISABLE_FUSION=1 ./build/bin/llama-bench ... # 融合の効き
GGML_CUDA_DISABLE_GRAPHS=1 ./build/bin/llama-bench ... # グラフの効き

# 2. 同時トークン数に対するステップ時間（MMVQ→MMQ の崖。Q4_K_M は 6、Q4_0 は 9 で切替）
./build/bin/llama-batched-bench -m model.gguf -c 70656 -b 2048 -ub 512 -fa 1 -npp 2048 -ntg 128 -npl 1,2,4,5,6,8,9,12,16,32
# -kvu を付けた場合と比較

# 3. ubatch 分割の確認（-np 8、非 unified）: id_slot で {0,1} → {0,2} → {0,2,5} を同時生成して合計 tok/s を比較。
#    2〜3 倍落ちれば該当。GGML_CUDA_GRAPH_DEBUG=1 でウォームアップのリセットを確認。--kv-unified で再測定。

# 4. カーネル時間 vs 壁時計（GPU 稼働率、to_fp16 変換、mul_mat_q と mul_mat_vec_q の比率）
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -o decode ./build/bin/llama-batched-bench ... -npl 1
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace decode.nsys-rep

# 5. 主要カーネルの達成帯域
ncu --kernel-name regex:mul_mat_vec_q --launch-skip 300 --launch-count 6 --section SpeedOfLight --section MemoryWorkloadAnalysis ./build/bin/llama-completion -m model.gguf -fa on -n 4 -no-cnv -p "..."
ncu --kernel-name-base demangled --kernel-name "regex:flash_attn_ext_f16<512" --launch-skip 20 --launch-count 4 --section SpeedOfLight ./build/bin/llama-completion ... -c 32768

# 6. MTP の d と位置別受理率: -np 1 --spec-type draft-mtp --spec-draft-n-max K（K=1..4）、/metrics の
#    llamacpp:spec_decode_num_accepted_tokens_per_pos_total を読む。greedy と probabilistic、T=0 と T=0.9 で比較。

# 7. サーバ全体: tools/server/bench/speed-bench（--category roleplay,multilingual --concurrency N）、
#    aiperf（§7）、vLLM は `vllm bench serve --dataset-name prefix_repetition ...`
```

判断の目安: GPU 稼働率 < 0.85（batch 1）→ サンプリング・起動オーバーヘッド対策（案 1,12,17,21）。`mul_mat_vec_q` の DRAM% < 80 → 案 21。`flash_attn_ext_f16<512>` の DRAM% < 60 → 案 18/19。`T_TG(npl=8)/T_TG(npl=1) > 1.6` → 案 20。スロット歯抜けテストで 1.5 倍以上落ちる → 案 3/16 を最優先。

---

## 10. 未確認事項（判断を変え得るもの）

1. Gemma 4 31B の **日本語チャット（温度あり）での受理率**。公開データは英語 greedy か、日本語でも DGX Spark の greedy 長文（55%）のみ。
2. dense 31B の **同時数別の投機利得の実測**（vLLM も「今後の課題」）。
3. 現行 vLLM 0.32 ＋ FlashInfer 0.7（head 512 対応、b12x）での **31B 単体の投機なし実測**。2026 年春の 40 tok/s 台はフォールバック期の数字。
4. llama.cpp の ubatch 分割の **実機での効果量**（コード上は確実、頻度と損失は推定）。
5. 量子化 KV × MTP の受理率崩壊（PR コメントと unsloth の記述が矛盾）。
6. `llama-quantize --tensor-type ...=nvfp4` が実用品質のファイルを作れるか（imatrix 無視、グローバルスケール無し）。
7. CISCai NVFP4 GGUF の埋め込みを Q8_0 に再量子化できるか（`.scale` テンソルの扱い）。
8. NeMo Automodel の Gemma 4 ドラフタ学習パス、speculators の Gemma 4 対応の最新状況。
9. CUDA 13.x ビルドの llama.cpp が SM120 で遅いという報告の真偽。
10. NVIDIA 公式 NVFP4 チェックポイントの現在のサイズ（32.7 GB → 約 18 GB に再アップロードされた可能性）。

---

## 付録 A. RTX PRO 5000 Blackwell の場合（改造以外の選択がどう変わるか）

RTX PRO 5000 Blackwell: 48 GB GDDR7 ECC（384-bit、**1,344 GB/s**）、14,080 CUDA コア（**110 SM**）、300 W、同じ SM120（CC 12.0）。72 GB 版（同じ帯域・コア数）が Puget Systems の製品一覧にある【未確認（単一出典）】。PRO 6000 比で帯域 0.75 倍、SM 数 0.585 倍、メモリ 0.5 倍（72 GB 版なら 0.75 倍）。Gemma 4 31B での公開実測は見つからなかったので、以下はすべて比例計算【推定】。

### A.1 何が変わらないか
- カーネル対応表（§2）はすべて同じ。SM120 の制約、vLLM の `--attention-backend FLASHINFER` 明示、llama.cpp の 3 つの問題（§3.1）、`-bs`、MTP、量子化 KV の f16 変換、`--cache-reuse` 無効、prefill が他スロットを止める点は全部そのまま当てはまる。

### A.2 何が変わるか
| 項目 | PRO 6000 | PRO 5000 48 GB | 影響 |
|---|---|---|---|
| 4bit 単一ストリーム上限 | 104 tok/s（現実 60 前後） | **78 tok/s（現実 45 前後）** | 帯域比 0.75 |
| Q8_0/FP8 重み | 55 tok/s、同時 20 系列以上 | 41 tok/s、**残り約 10 GiB で 8K が 5 系列** | 実質 4bit 一択。BF16 は不可 |
| prefill（TTFT） | Q4 で約 3,700 tok/s | 約 2,200 tok/s | SM 比 0.585。2K ペルソナの初回で 1 秒前後 |
| 検証バッチの計算の崖（4bit） | 30〜40 行 | **25〜30 行** | 投機の K を早く絞る: 同時 4 で K=2、8 で K=1、12 以上は切る |
| llama.cpp f16 KV の同時数（8K、`-ub 512`） | 約 40 | **約 14**（`-ub 256` で 16、16K で 10、32K で 7） | スロット数の上限がメモリで決まる |
| vLLM W4A16 ＋ FP8 KV（8K） | 約 85 | **約 33**（bf16 KV なら 16） | FP8 KV が「任意」から「必須」に |
| DFlash/DSpark ドラフタ（BF16 4〜5 GB） | 余裕 | 8K 系列 3 本分の KV を食う | MTP（0.5 GB）の方が合理的 |
| 合計スループット（8K、現実係数 0.65）| llama.cpp 16 並列 約 380、vLLM FP8 KV 約 580 | llama.cpp 14 並列で頭打ち **約 270**、vLLM FP8 KV 16 並列 **約 435** | 多人数では vLLM の優位が大きくなる |
| 300 W の電力枠 | 600 W（500 W でほぼ同等） | 固定 | 多人数の MMQ 計算・prefill で頭打ちになりやすい |

72 GB 版なら容量側は PRO 6000 の 7 割まで戻る（llama.cpp f16 KV で 8K 約 27 スロット、vLLM FP8 KV で約 66 系列）。帯域と SM 数は変わらないので単一ストリーム・prefill は 48 GB 版と同じ。

### A.3 改造以外でできる選択（優先順）
1. **同時数の目標で先に判断する。** 8K コンテキストで同時 8 人までなら llama.cpp（QAT Q4_0 ＋ MTP K=1〜2 ＋ `-bs` ＋ `-ub 256` ＋ `-c` を 8〜16K/スロットに制限）で足りる。同時 12 人以上、または 16K 以上の履歴を抱えるなら、FP8 KV で容量と KV 読み出しが半分になる vLLM（`--kv-cache-dtype fp8`、`--max-num-seqs 16〜24`、`--max-model-len 16384`）に切り替える価値が PRO 6000 より大きい。
2. **モデルサイズの選択が最大のレバーになる。** 31B は 4bit でも 17.3 GB/トークンで、この帯域では 45 tok/s 前後が現実値。`gemma-4-26B-A4B-it`（MoE、活性 3.8B、4bit 約 15 GB、1 ステップの重み読み出し 3〜10 GB）は単一 3 倍前後・同時 8 で 1 ステップ 7.5 GB → 合計の天井が 31B の 4 倍、しかも MoE は MTP の利得がバッチで増える。`gemma-4-12B-it`（4bit 6.8 GB → 上限 199 tok/s）も候補。キャラチャットでの品質差は §8 の日本語ハーネスとペア比較で確かめる。
3. **アイドルセッションはホスト RAM へ**（llama.cpp `--cache-ram 32768〜65536` ＋ `--cache-idle-slots`、vLLM は `kv_offloading_size`／LMCache）。48 GB では「常駐させるペルソナの数」が直接スループットを削るので、PCIe 5 の 40〜80 ms の復元コストを払ってでも退避する。
4. **KV を伸ばさない運用**: 履歴の要約・切り詰めで 1 系列 8K 以内に収める、ペルソナを 1〜1.5K に圧縮する、画像入力を切って視覚塔（1.1 GB）を載せない（llama.cpp は mmproj を付けない、vLLM は `--language-model-only`）。
5. **カード 2 枚（96 GB 相当）**: vLLM の TP=2 は PCIe（NVLink なし）で層ごとの all-reduce が入り、単一ストリームは 1 枚とほぼ同じ〜やや遅い（帯域の合算は効くが同期で相殺）。容量と合計スループットは 2 倍近くまで伸びる。llama.cpp の分割（`--split-mode layer`）は容量だけ増えて速度は 1 枚分。単一ストリーム重視なら PRO 6000 1 枚、同時数重視なら PRO 5000 2 枚 ＋ vLLM という選び方になる。
6. **投機の設定**: MTP ドラフタ（0.5 GB）は使う。K は同時 1〜2 で 3、4 で 2、8 で 1、12 以上で 0。DFlash/DSpark は VRAM と Triton 経路の点で後回し。
7. **計測**: §9 の手順をそのまま使う。崖の位置（`llama-batched-bench -npl 1,2,4,5,6,8,9,12`）と FP8 KV の有無での同時 16 の TPOT（vLLM）を先に測る。

## 付録 B. Gemma 4 26B-A4B（MoE）に切り替えた場合の見積もり

### B.1 構成（HF config の引用スニペット、vLLM／llama.cpp のソースで確認）
- 30 層 ＝ SWA（窓 1,024）25 層 ＋ グローバル 5 層（5:1）。hidden 2,816、16 ヘッド。ローカル KV 8 × 256、グローバル KV 2 × 512（K==V）。語彙 262,144（tied）。PLE なし。
- 全 30 層が MoE: 128 エキスパート中 top-8（softmax → top-k → 再正規化）、エキスパート幅 704、**それと並列に幅 2,112 の密な MLP（共有エキスパート相当）**が毎層走る。
- パラメータ（計算値、Google 公称 25.2B／3.8B と一致）: エキスパート 22.84B（1 個 5.95M、全体の 90%）、密 MLP 0.535B、アテンション 1.11B、埋め込み 0.738B → 合計 25.2B、**トークンあたり有効 3.82B**。
- ファイル: Google QAT Q4_0 GGUF 14.4 GB（＋mmproj 1.19 GB）、Unsloth QAT UD-Q4_K_XL 14.2 GB、Q4_K_M 約 16.5 GB、NVFP4 約 16〜16.5 GB、Q8_0 約 28.4 GB、FP8 約 26〜29 GB【未確認】、BF16 50.4 GB。注意: エキスパートの down 行列は幅 704 で 256 の倍数でないため、K 量子化／IQ 量子化が使えず約 31% の重みが Q5_0/Q4_0/IQ4_NL に落ちる（「Q4_K_M」の実 bpw がアップローダごとに違う）。
- MTP ドラフタ `google/gemma-4-26B-A4B-it-assistant`: 4 層（SWA 3 ＋ グローバル 1）、hidden 1,024、約 420M、BF16 839 MB。

### B.2 1 ステップの読み出しバイト数（独立ルーティング仮定: 同時 c で使うユニークなエキスパート数 ＝ 128 × (1 − (1 − 8/128)^c)）
| 同時数 c | ユニーク専門家/層 | 4.5 bpw（Q4_0/QAT） | FP8（head BF16） | 参考: 31B 4.5 bpw |
|---|---|---|---|---|
| 1 | 8.0 | 2.15 GB | 4.56 GB | 17.3 GB |
| 4 | 29.1 | 4.27 GB | 8.33 GB | 17.3 GB |
| 8 | 51.6 | 6.53 GB | 12.34 GB | 17.3 GB |
| 16 | 82.4 | 9.62 GB | 17.84 GB | 17.3 GB |
| 32 | 111.8 | 12.6 GB | 22.8 GB | 17.3 GB |

- 投機（MTP）を使うと検証トークン数 c×(K+1) でユニーク数が増える（c=4・K=3 で 16 トークン → 82 個 → 9.6 GB）。そのため **MoE では低同時数ほど投機のコストが相対的に高く、同時数が増えるほど無料に近づく**（vLLM の実測で MTP の利得が bs1 の 1.60 倍 → bs64 の 2.27 倍に増える理由）。
- KV（f16）: ローカル層 8 KiB/トークン/層 × 25 ＝ 200 KiB/トークン、窓 1,024 で 210 MB/系列（llama.cpp のリング 1,536 セルなら 300 MB）。グローバル層 4 KiB × 5 ＝ **20 KiB/トークン**（31B の 1/4）。32K で 0.82 GiB、128K で 2.7 GiB／系列。**16 ユーザー × 32K が f16 で 13.1 GiB**（31B は 52.5 GiB）。
- 帯域だけの上限は PRO 6000 で c=1 834 tok/s、FP8 でも 393 tok/s。しかし実測はどのエンジンも 140〜270 tok/s で、**単一ストリームは帯域ではなく起動・レイテンシ律速**（llama.cpp は 1 ステップ約 1,250 カーネル）。したがって単一ストリームでは量子化を軽くしても速くならず、逆に **26B なら FP8／Q8_0 を選んでも単一ストリームの速度はほぼ変わらない**（多人数では効く）。

### B.3 公開実測（RTX PRO 6000、2026 年）
| 条件 | 単一ストリーム | 並列 | 出典 |
|---|---|---|---|
| llama.cpp Q4 | **196 tok/s（4K）**、170（32K）、133（128K）、pp 9,437 tok/s | 未計測 | hardware-corner |
| Ollama Q4_K_M | 171〜181 | 4 並列で vLLM に劣る | allenkuo |
| vLLM NVFP4（Marlin MoE）＋ MTP γ=4 | **169.2** | **2,237 tok/s @16 並列**（v0.21）、1,807（v0.23 CUTLASS） | vLLM issue #47047 |
| vLLM NVFP4、投機なし | 160〜162 | 約 2,170 @32 | lna-lab、petronella |
| vLLM FP8 | 139（1K）、77（256K）、TTFT 81 ms | 674 @10（1K）、334 @10（32K） | MillstoneAI |
| RTX 5090 llama.cpp Q4 | 180〜249 | — | hardware-corner、Atomic Chat |
| H100 llama.cpp Q4_K_M | 215（メガカーネルで 342） | — | paper-19 |

MTP: vLLM は c=1 で 1.5〜1.6 倍、c=16 で 1.85〜2.3 倍。**llama-server では MoE に対して ≈0〜1.3 倍**（PR #23398 作者「MoE は高速化せず」、issue #27732 でも server はほぼ 0）。日本語長文（DGX Spark、NVFP4 Marlin）: 28.1 → 47.9 tok/s（1.71 倍、受理率 54.9%）、短い返答は −19%。

### B.4 見積もり（この環境で切り替えた場合）【推定、±30%】
| | 31B（現状の最適化後） | 26B-A4B PRO 6000 | 26B-A4B PRO 5000 48 GB |
|---|---|---|---|
| 単一ストリーム、投機なし | 55〜65 | llama.cpp Q4 170〜200／vLLM 140〜170 | 130〜170（起動律速なので帯域比 0.75 より落ちない） |
| 単一ストリーム、MTP あり | 80〜125 | vLLM 230〜270（日本語は ×1.7 で約 230）／llama-server はほぼ不変 | vLLM 190〜230 |
| 同時 8 合計（8K） | 290〜480 | vLLM FP8 約 670（実測）、NVFP4＋MTP 1,200〜1,500、llama.cpp 500〜900（`-np` の MoE スケーリングは未計測） | vLLM 500〜1,100 |
| 同時 16 合計（8K） | 360〜650 | **vLLM NVFP4＋MTP 1,800〜2,240（実測）**、FP8 投機なし 約 1,000 | 1,300〜1,700 |
| TTFT（2K ペルソナ、キャッシュミス） | 0.5 秒 | 0.2 秒（prefill 9,400 tok/s、FLOP は 31B の 1/8） | 0.35 秒 |
| 16 × 32K の KV（f16） | 52.5 GiB（PRO 5000 では不可） | 13.1 GiB | 13.1 GiB（Q4 14.4 GB と合わせて余裕、FP8 28 GB でも fp8 KV なら可） |

要約すると、同じカードで **単一 3 倍、多人数 3〜5 倍**。PRO 5000 48 GB でも 16 ユーザー × 32K が収まり、「並列数がメモリで決まる」制約が消える。vLLM の方が MTP とバッチングの利得を取り切れるので、26B-A4B にするなら vLLM 側の優位が 31B のときより大きい。

### B.5 品質面の代償（切り替え判断の本体）
- 公式ベンチ（26B／31B）: MMLU-Pro 82.6／85.2、GPQA Diamond 82.3／84.3、LiveCodeBench v6 77.1／80.0、IFEval 98.5／98.9、Tau2 68.2／76.9、**MRCR v2 128K 44.1／66.4（長文脈が最大の差）**。
- 日本語: Swallow Leaderboard v2 の Ja 平均 0.646／0.675、Ja MT-Bench 0.806／0.815、JCommonsenseQA 96.4／97.9。
- キャラクターチャットの報告（X・note、逸話レベル）: 26B-A4B は「人格の深みが表面的」「設定や関係性を取りこぼす」「一人称・二人称が入れ替わる」「長いシステムプロンプトで thinking を飛ばす」「突然キャラを崩して拒否する」。31B は長文脈でも人格を保つ。この用途では **31B との差が最も出やすい領域**。
- 量子化感度が高い: LocalBench では Q8_0 でも KL 0.544（31B は 0.163）、KV q8_0 で KL 0.377、q4_0 KV で 1.088。Unsloth: QAT を素朴に Q4_0 化すると top-1 一致 70.2%（UD-Q4_K_XL は 85.6%）。vLLM 公式レシピは「エキスパートが小さく 4bit で品質が落ちすぎる」として 26B-A4B を W4A16 QAT の対象から外し、`int8_per_channel_weight_only` を勧める。→ **26B-A4B なら Q8_0／FP8／INT8 を選ぶのが筋**（単一ストリームは起動律速なので速度はほぼ変わらず、PRO 5000 でも容量は足りる）。KV は f16 のまま。
- サンプラ: コミュニティ推奨は repetition penalty 無効（1.0）、Google 既定 temp 1.0／top-p 0.95／top-k 64。

### B.6 運用メモ
- vLLM: `vllm serve <FP8 または NVFP4 チェックポイント> --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --async-scheduling --limit-mm-per-prompt '{"image":0,"audio":0}' --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'`。TP=1、`--enable-expert-parallel` は不要。SM120 の MoE バックエンドは版で変わる（Marlin／CUTLASS／b12x、`--moe-backend` で指定、起動ログで確認。v0.21 → v0.23 で退行例あり）。
- llama.cpp: `-ngl 99 -fa on`、全エキスパートが VRAM に載るので `--n-cpu-moe` 不要、`--swa-full` は避ける（32K スロットあたり 6.9 GiB）。CUDA グラフ再有効化（PR #26802、2026-08）で並列デコード +13%。MTP は server では効かないので投機は期待しない。
- 現実的な進め方: 31B を主、26B-A4B を「キャラ単位の静的ルーティング」で併用（§8 案 15）。まず素の 26B-A4B-it を同じペルソナで 200 文脈に当て、人手ブラインドの勝率が 31B 比 40% 以上のキャラだけ切り替える。両モデル同時常駐は PRO 6000 なら余裕（14〜28 ＋ 18 GB）、PRO 5000 48 GB では 26B を Q4 にすれば可能（14.4 ＋ 17.7 ＋ KV）。

### B.7 未確認
llama-server の `-np` に対する MoE（MUL_MAT_ID）のスケーリング実測、PRO 5000 での実測、日本語ペルソナ対話での 26B 対 31B の統制比較、FP8 チェックポイントの実サイズ、独立ルーティング仮定（実際は偏るのでバイト数は少なめ）、vLLM 0.32 での MoE バックエンドの既定。

## 11. 主な参考 URL

- Gemma 4 技術レポート: https://arxiv.org/abs/2607.02770 ／ vLLM 公式レシピ: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html ／ Google MTP ドラフタ: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- llama.cpp: Gemma 4 MTP PR #23398、投機ドキュメント `docs/speculative.md`、サーバ README、`docs/build.md`（GGML_CUDA_MMQ_PREC）、issue #27050（-bs +48%）、#29175（prefill 停止）、#25522、#30368、#22527、#30792（q8_0 KV tile）
- vLLM: issue #40677（SM120 FlashInfer head 512）、#39133（Gemma 4 KV、k_eq_v 最適化は後回し）、#48848（MTP 退行）、PR #46963/#60452（NVFP4 KV SM12x）、PR #41745（Gemma 4 MTP）、blog 2026-08-23（MTP/EAGLE-3/DFlash/DSpark 比較）
- 実測: MillstoneAI（https://www.millstoneai.com/inference-benchmark/gemma-4-31b-nvfp4-1x-rtx-pro-6000-blackwell）、HexGrid（https://huggingface.co/blog/hexgridcloud/gemma-4-31b-vllm-on-rtx-6000-pro）、truespar（https://truespar.com/paddock/benchmarks/gemma4-31b）、hikarioyama DSpark（https://github.com/hikarioyama/dspark-gemma4-31b）、rtx6kpro wiki（https://github.com/local-inference-lab/rtx6kpro）、Classmethod 日本語 MTP（https://dev.classmethod.jp/en/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/）、kotonia 日本語受理率（https://kotonia.ai/en/articles/mtp-ja-acceptance/）、unsloth MTP README（https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/blob/main/MTP/README.md）
- 量子化: Google QAT（https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf）、CISCai NVFP4 GGUF（https://huggingface.co/CISCai/gemma-4-31B-it-NVFP4-turbo-GGUF）、Kaitchup 比較（https://kaitchup.substack.com/p/gemma-4-31b-quantization-comparison）、beri.net（https://www.beri.net/article/fp8-vs-nvfp4-vs-int4-awq-gguf-quantization-format-accuracy-throughput-2026）
- ドラフタ: RedHatAI EAGLE-3（https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3）、z-lab DFlash（https://github.com/z-lab/dflash）、speculators（https://github.com/vllm-project/speculators）
- SM120: FlashInfer PR #3051（b12x）、issue #2577、TensorRT-LLM issue #11799／#17052、CUTLASS Blackwell 機能表（https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md）
