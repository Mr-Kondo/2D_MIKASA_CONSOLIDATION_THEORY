# Colab CUDAとJupyterでの実行

[NotebookをColabで開く](https://colab.research.google.com/github/Mr-Kondo/2D_MIKASA_CONSOLIDATION_THEORY/blob/main/notebooks/consolidation_cuda.ipynb)。
公開リポジトリを取得し、計算パッケージをインストールして、両モデルの計算・図の表示・CPU/CUDA比較・CUDA実機テストを順に実行します。

## Google Colab

1. 「ランタイム」→「ランタイムのタイプを変更」でNVIDIA GPU（例: T4）を選択します。GPUが割り当てられない場合は、CPU検証に切り替えるか、割り当て後に再実行します。
2. 設定セルの`DEVICE="cuda"`を確認し、上から順に実行します。`REF`は取得するGitブランチまたはタグです。既存の取得先がある場合は`ff-only`で更新し、ローカル変更や分岐を強制的に消しません。
3. `cuda_info()`のGPU名・CUDA実行環境・CuPy版を確認します。CUDAが使えなければエラーで停止し、CPUへ自動切替しません。
4. 生成した3枚のPNGをNotebook内に表示します。画像と比較JSONは`outputs/colab/`へ保存します。同名ファイルは上書きします。
5. 必要な結果をダウンロードした後、ランタイムを切断して削除します。Colabの一時ディスクは保存先ではありません。

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

NVIDIA GPUと対応するドライバーがあるLinux・Windowsでは、インストール時に`--extra cuda`も指定します。
CUDA 12系向けCuPyを使用します。macOSはNVIDIA CUDAの対象外です。

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

比較セルは同じ条件で初回の時間と、交互順序で各3回実行した中央値を記録します。コンストラクターから計算終了までを測り、CPUへの結果転送も含めます。パッケージ導入・描画・ファイル保存は含みません。
`speedup = CPU中央値 / CUDA中央値`で、1より大きければCUDAが速いという意味です。比較時に全状態の有限性とCPUとの一致（絶対・相対許容差とも`1e-8`）を検証し、不一致を黙認しません。

実機テストは三重対角解法の独立した行列残差、1D・2Dの状態一致、reset、収束失敗の通知を確認します。CPUのCIはこれらのGPUテストをskipします。
GPUの性能と実験・実地盤への適合性は別問題です。実データ比較は[Issue #32](https://github.com/Mr-Kondo/2D_MIKASA_CONSOLIDATION_THEORY/issues/32)で追跡します。
