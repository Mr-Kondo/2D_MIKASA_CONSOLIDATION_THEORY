# Colab CUDAとJupyterでの実行

[NotebookをColabで開く](https://colab.research.google.com/github/Mr-Kondo/2D_MIKASA_CONSOLIDATION_THEORY/blob/main/notebooks/consolidation_cuda.ipynb)。
公開リポジトリを取得し、計算パッケージをインストールして、両モデルの計算・図の表示・CPU/CUDA比較・CUDA実機テストを順に実行します。

## Google Colab

1. 「ランタイム」→「ランタイムのタイプを変更」でNVIDIA GPU（例: T4）を選択します。GPUが割り当てられない場合は、CPU検証に切り替えるか、割り当て後に再実行します。
2. 設定セルの`DEVICE="cuda"`を確認し、上から順に実行します。`REF`は取得するGitブランチまたはタグです。既存の取得先がある場合は`ff-only`で更新し、ローカル変更や分岐を強制的に消しません。
3. `cuda_info()`のGPU名・CUDA実行環境・CuPy版を確認します。CUDAが使えなければエラーで停止し、CPUへ自動切替しません。
4. 生成した3枚のPNGをNotebook内に表示します。画像と比較JSONは`outputs/colab/`へ保存します。同名ファイルは上書きします。
5. 「ファイル → ダウンロード → .ipynb」で実行済みNotebookを保存すれば、GPUがないローカルJupyterでも埋め込み済みコンター図を開けます。必要な結果をダウンロードした後、ランタイムを切断して削除します。Colabの一時ディスクは保存先ではありません。

Notebookは計算コードを複製しません。`src/mikasa_consolidation/`をインポートし、ローカルの入口スクリプトと共通の`run_examples()`を呼びます。
これは[rokko-geofusion](https://github.com/Mr-Kondo/rokko-geofusion)の、計算パッケージとNotebookの役割を分ける構成を参考にしています。
同一カーネルでのインストール後に`src/`を探索パスへ追加するため、初回実行でインポートできます。

## ローカルのJupyter

CPUで使う場合:

```bash
uv sync --locked --extra notebook
uv run --locked --extra notebook python -m ipykernel install --user --name mikasa --display-name "Mikasa"
```

お使いのJupyterで`notebooks/consolidation_cuda.ipynb`を開き、カーネルに「Mikasa」を選びます。
設定セルの`DEVICE="cpu"`に変更すると、図の生成と表示をCPUで実行します。CUDA比較とCUDA実機テストは実行しません。
カーネル登録が不要になったら、Jupyterの`kernelspec remove mikasa`で削除できます。

NVIDIA GPU・対応するドライバー・CUDA 12系ToolkitがあるLinux・Windowsでは、インストール時に`--extra cuda`も指定します。
ローカルの`cuda` extraはCUDA 12系向けCuPyを導入します。Colabでは既存のCuPy（CUDA 13系を含む）を再利用し、別系列を重複インストールしません。macOSはNVIDIA CUDAの対象外です。

```bash
uv sync --locked --extra notebook --extra cuda
uv run --locked python -m mikasa_consolidation --device cuda --output-dir outputs/cuda
```

## GPUで行う処理と測定範囲

| モデル | CUDAで実行する処理 | CPUに残る主な処理 |
| --- | --- | --- |
| 有限ひずみ | Newton反復内の三重対角連立方程式（Thomas法、列ごとに並列） | 残差・ヤコビアン構築、収束判定、時間分割、状態更新 |
| 線形小ひずみ | モード変換の行列積とモードの時間増幅 | 初期化・固有値分解、変位計算 |

両者とも倍精度`float64`を維持し、描画はCPUで行います。有限ひずみモデルは反復ごとにCPU/GPU間の転送が生じるため、小規模な計算では遅くなる可能性があります。
線形モデルはもともと時間ループを消した実装なので、GPUの選択だけで高速化を保証しません。

比較セルは同じ条件で初回の時間と、交互順序で各3回実行した中央値を記録します。コンストラクターから計算終了までを測り、CPUへの結果転送も含めます。パッケージ導入・描画・ファイル保存は含みません。「初回」は比較セル内での初回であり、前セルですでにCUDAを初期化しているため、プロセス起動からのコールドスタートではありません。
`speedup = CPU中央値 / CUDA中央値`で、1より大きければCUDAが速いという意味です。比較時に全状態の有限性とCPUとの一致（絶対・相対許容差とも`1e-8`）を検証し、不一致を黙認しません。

実機テストは三重対角解法の独立した行列残差、1D・2Dの状態一致、reset、収束失敗の通知を確認します。CPUのCIはこれらのGPUテストをskipします。
GPUの性能と実験・実地盤への適合性は別問題です。実データ比較は[Issue #32](https://github.com/Mr-Kondo/2D_MIKASA_CONSOLIDATION_THEORY/issues/32)で追跡します。

参照: [CuPy公式の導入方法](https://docs.cupy.dev/en/stable/install.html)（CUDA系列ごとのパッケージと、複数CuPyを混在させない条件）。

## 実測記録（2026-10-22）

ColabのTesla T4、Python 3.13.15、CuPy 14.0.1、CUDA runtime/driver 13.0、NumPy 2.1.3で実行しました。計算コードのcommitは`7acb40b5f219dfa6a27267362f6cf09b5fb5f05a`です。

21×21格子、`dt=5` day、`total_time=100` day、他は各モデルの既定条件での結果:

| モデル | CPU中央値[s] | CUDA中央値[s] | CPU/CUDA | 全状態の最大絶対差 |
| --- | ---: | ---: | ---: | ---: |
| 有限ひずみ | 1.578836 | 2.206595 | 0.716 | 0 |
| 線形 | 0.000615 | 0.001608 | 0.383 | 0 |

この条件では両モデルともCPUのほうが速い結果でした。[生の測定JSON](evidence/colab_t4_21.json)に初回・各試行・環境を保存しています。
CUDA実機テストは4件すべて成功しました（[実行ログ](evidence/colab_t4_cuda_tests.txt)）。
既定例の主要計算もCUDAで完走し、ダウンロードした実行済みNotebookに3枚のPNGが埋め込まれていることを確認しました。

このブラウザーでは、Colabが「出力表示用JavaScriptを読み込めない。ログイン期限またはサードパーティCookieの可能性」と報告し、画像出力を画面上で表示できませんでした。再読み込みでも改善せず、ブラウザー設定は変更していません。これはGPU計算やPNG生成の失敗とは区別します。ローカルの実JupyterカーネルではCPU設定の全セルが完走し、3枚のPNG出力を確認しました。Colabの実行済みNotebookをダウンロードしてローカルJupyterで開く場合は、計算済みの画像をそのまま閲覧できます。

追加の101×101格子測定は、Colabの接続が「Resuming execution」のままになり結果を回収できませんでした。原因は特定できていません。この試行の時間や成功・失敗を数値結果として扱わず、大きい格子での高速化は未検証です。
