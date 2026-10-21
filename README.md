# 2D Consolidation — 三笠／Gibson 系の有限ひずみ圧密

物質座標で有限ひずみ圧密を解く教材用モデルです。間隙比の圧縮則と透水則、自重、移動表面を扱います。
一定係数の線形拡散（寺田型の小ひずみ圧密）は参照実装として併設し、両者を同じ `c_v` で比較できます。

平面ひずみ二次元への拡張は**著者による拡張**であり、三笠が公表した理論ではありません。
原著（三笠 1963 など）との逐語的な一致は主張せず、公開されている一次資料の本文も確認していません。

## 導入と実行

Python 3.13以上と [uv](https://docs.astral.sh/uv/) が必要です。計算はCPU上のNumPyで行い、GPUは使いません。

```bash
uv sync --locked
uv run --locked Mikasa.py
uv run --locked Terzaghi.py
```

- `Mikasa.py`: 有限ひずみ圧密（`finite_strain.py`）を既定条件で実行し、`mikasa_consolidation_result.png` を生成します。
- `Terzaghi.py`: 線形小ひずみモデル（`consolidation.py`）で `terzaghi_consolidation_result.png` を、両理論の比較図 `terzaghi_mikasa_comparison.png` を生成します。
- 画像は実行時の作業ディレクトリへ保存し、同名の既存画像を上書きします。通常は画面にも表示します。

画像を画面に表示しない環境では、`MPLBACKEND=Agg` を設定してください。

```python
from finite_strain import FiniteStrainConsolidation

simulator = FiniteStrainConsolidation(
    length=20.0, depth=10.0, nx=21, nz=21,
    e0=2.0, Cc=0.6, Ck=0.6, k0=0.02,
    sigma_ref=10.0, sigma_top=10.0, specific_gravity=2.65, unit_weight_water=9.81,
    load_intensity=50.0, load_width=4.0,
    dt=5.0, total_time=3650.0,
)
settlement = simulator.run_simulation(verbose=False)
simulator.plot_results(save_path="result.png", show=False)
settlement, horizontal, pressure, strain = simulator.get_current_state()
```

## 有限ひずみ圧密（三笠／Gibson 系）

単位は長さm、時間day、応力kPa、間隙比は無次元です。地表が `z=0`、固定底面が `z=depth` で、沈下は下向きを正とします。

### 座標と材料則

固体体積を基準にした物質座標 $\zeta$ を使います。地表で $\zeta=0$、増加方向は下向きです。

$$
\zeta=\int_0^z\frac{dz'}{1+e},\qquad
e=e_0-C_c\log_{10}\frac{\sigma'}{\sigma'_{\mathrm{ref}}},\qquad
k=k_0\,10^{-(e_0-e)/C_k}.
$$

有効応力 $\sigma'$ が増えると間隙比が減り、透水係数も減ります。$\zeta$ 座標では要素の固体体積が時間によらず一定で、層厚の減少は $\zeta$ から物理深さ $z$ への写像の変化として現れます（移動表面）。

### 支配方程式

過剰間隙水圧を $u$ とすると、鉛直一次元は

$$
\frac{\partial e}{\partial t}
=\frac{\partial}{\partial \zeta}\left[\frac{k(e)}{\gamma_w(1+e)}\frac{\partial u}{\partial \zeta}\right],\qquad
\frac{\partial u}{\partial \zeta}=\gamma_w(G_s+e)+\frac{1}{a_v}\frac{\partial e}{\partial \zeta},\quad
a_v=\frac{C_c}{\sigma'\ln 10}.
$$

全応力は地表応力と荷重、自重 $\gamma_w(G_s+e)$ の積分で与え、$u$ は全応力と有効応力の差です。
地表は排水 $u=0$、底面は非排水です。初期状態は、地表応力と自重に釣り合う $\sigma'(e)$ のプロファイルとして数値的に求めます。

### 二次元化（著者による拡張）

平面ひずみ $\varepsilon_x=0$ を仮定し、水平方向の流れ

$$
\frac{\partial}{\partial \xi}\left[\frac{k}{\gamma_w}\frac{\partial u}{\partial \xi}\right],\qquad
\frac{\partial u}{\partial \xi}=\frac{\partial \sigma_v}{\partial \xi}+\frac{1}{a_v}\frac{\partial e}{\partial \xi}
$$

を加えます。荷重は中央の等分布帯荷重に対する弾性半無限体の鉛直応力増分です（線形モデルと同じ式）。

$$
\theta=\operatorname{atan2}(X+a,z),\quad
\beta=\operatorname{atan2}(X-a,z),\quad
\alpha=\theta-\beta,\quad
\sigma_{\mathrm{load}}(X,z)=\frac{q}{\pi}\left[\alpha+\sin\alpha\cos(\theta+\beta)\right].
$$

せん断変形、水平力の釣合い、載荷にともなう全応力分布の変化は解きません。$nx=1$ は鉛直一次元です。

### 数値解法

セル中心の有限体積と後退Euler（陰的）を使います。未知数は間隙比 $e$ で、面のフラックスは
$\partial u/\partial \zeta$ の離散差から直接作ります。支配方程式の表式どおりであり、自重と釣り合った状態（$u=0$）では
面のフラックスが厳密に0になるため、無載荷で見かけの圧密が起きません。

- 鉛直方向は Newton 法で解きます。ヤコビアンは面ごとの厳密な微分から作る三重対角行列で、Thomas 法で解きます。
- 二次元では鉛直と水平の Newton 掃引を交互に行い、掃引による更新量が許容値以下になるまで外側で反復します。
- 反復が収束しないときは時間刻みを半分に分割して再試行します（`max_splits` 回まで）。それでも収束しなければ
  `RuntimeError` で報告します。
- 間隙比・水圧を任意の上限でクリップしません。非有限の水圧は `FloatingPointError` として報告します。
- 反復の許容値は1ステップの変化量に比例させ、丸め誤差より十分大きい床を置きます。

水平方向の面では $\partial \sigma'/\partial \xi$ を面の有効応力で近似するため、鉛直方向に比べて
境界近傍の次数が一段低くなります（[検証](#検証)の 2D→1D 一致と格子収束を参照）。

## 線形小ひずみの参照モデル（寺田型）

`consolidation.py` の `ConsolidationSimulator` は、一定係数の2次元水圧拡散を解き、各鉛直柱の拘束圧密で沈下へ換算します。
有限ひずみモデルとの比較と、解析解が既知の基準として残しています。

### 水圧の消散

$$
\frac{\partial u}{\partial t}=c_v\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial z^2}\right).
$$

地表は排水、底面・左右端は非排水です。空間差分には非排水端で鏡映点を使い、時間方向には Crank–Nicolson 法を使います。

$$
(I-\tfrac12c_v\Delta tL)u^{n+1}=(I+\tfrac12c_v\Delta tL)u^n.
$$

一定係数・固定境界なので、重み付き対称行列の固有値分解から各モードの増幅率を求め、$N$ ステップを累乗でまとめて評価します。
反復線形解法や時間ループは不要です。

`dt` は時間刻みの上限です。増幅率 $(1+h)/(1-h)$（$h=\tfrac12c_v\Delta t\lambda$）が非負になるよう、
$\Delta t c_v(1/\Delta x^2+1/\Delta z^2)\le 1/2$ を満たす刻みに縮めます。整数ステップ数を決めた後で
`total_time / nt` に調整し、端数時間を切り捨てません。実際の刻みは `simulator.dt` で確認できます。

### ひずみと沈下

$$
D=\frac{E(1-\nu)}{(1+\nu)(1-2\nu)},\qquad
\varepsilon_z=\frac{\sigma_0-u}{D},\qquad
w(z)=\int_z^H\varepsilon_z(\zeta)\,d\zeta.
$$

積分は台形則です。固定底面の沈下は0、水平変位は0です。局所的に $u>\sigma_0$ となった場合の負のひずみも消去しません。
水圧拡散の結果を局所的に換算する近似で、せん断応力や空間的に変化する沈下場の完全な弾性適合条件を解くものではありません。

## APIと制約

`FiniteStrainConsolidation`（別名 `MikasaConsolidationSimulator`）:

- `length`, `depth`, `e0`, `Cc`, `Ck`, `k0`, `sigma_ref`, `sigma_top`, `specific_gravity`, `unit_weight_water`, `load_width`, `dt`, `total_time`, `tolerance`: 有限の正数。
- `load_intensity`: 有限の非負数。`nx`, `nz`, `max_iterations`, `max_splits`: 整数で、`nx`・`max_*` は1以上、`nz` は2以上。
- 数値でない値（文字列・`None`）と `True`/`False` は数値引数として受理しません。不正な入力には `ValueError`、演算範囲を超えた条件には `FloatingPointError` または `RuntimeError` が発生します。
- `run_simulation(verbose=True)`: 現在の状態から `total_time` だけ進め、形状 `(nz, nx)` の沈下配列[m]を返します。再度呼ぶと、さらに同じ時間だけ進みます。
- `get_current_state()`: 沈下、水平変位、過剰間隙水圧、鉛直ひずみの順で、NumPy配列のコピーを返します。
- `get_consolidation_degree()`: 水圧積分の消散率[%]。沈下量に基づく圧密度とは区別します。初期の過剰水圧が応力スケールに対して無視できる（載荷なし）場合は100%とします。
- `reset()`: 初期の間隙比・応力・変位を復元します。
- `plot_results(save_path=None, show=True, figsize=(18, 5), dpi=300)`: 沈下・過剰間隙水圧・間隙比を描画します。`nx=1` は奥行き方向の線図、`nx>=2` は物理深さを使った塗り分け図です。保存・表示後には図を閉じます。

`ConsolidationSimulator`（別名 `TerzaghiConsolidationSimulator`）は上記に加えて `cv`, `E`, `poisson_ratio`（`0 <= poisson_ratio < 0.5`）, `device`（`"auto"` または `"cpu"`）を取ります。
`"mps"`, `"cuda"`, `"mlx"` は明示的に拒否します。`get_current_state()` の順序は同じで、`plot_results` は沈下・水平変位・水圧を描画します。

生成後に格子・材料・時間・荷重条件を変更するときは、新しいインスタンスを作ってください。
二次圧密、時間変動荷重、不均一な透水性、載荷にともなう全応力の再配分、せん断・水平力の釣合いは扱いません。
実地盤への適合性・設計精度は検証していません。

`Terzaghi.py` には比較用の補助関数があります。`matched_cv()` と `matched_modulus()` は圧縮則から換算した
同じ $c_v$ と弾性係数を返し、`run_pair(T)` は両モデルを同じ条件で走らせます。`terzaghi_degree(T)` は
片面排水の解析解、`plot_comparison()` は比較図を保存します。

## 検証

```bash
uv run --locked python -m unittest -v
```

29テストで、有限ひずみモデルと線形モデルの両方を検証します。

- 自重の釣り合い: 無載荷では初期状態が保たれ、`max|u| ≤ 1.7e-13 kPa`、沈下は厳密に0（`nz=4〜64`）。
- 定常解: 自重と有効応力の釣り合いを独立に RK4 積分した参照解との `e` 誤差が
  `1.05e-4 → 3.17e-5 → 8.77e-6 → 2.31e-6`（`nz=11, 21, 41, 81`）、各段で2倍未満に減少。定常状態の `max|u|` はほぼ0。
- 小ひずみ極限: 荷重を初期有効応力の $10^{-6}$ 倍に取り、線形モデルと同じ $c_v$ を与えた1次元解析で、
  寺田の解析解との差が `T=0.1, 0.2, 0.4` で `-0.10, -0.08, -0.08` ポイント。
- 時間刻み: `nz=21`、`T=0.4` で $U=$ `68.654, 69.178, 69.613, 69.724` %（`dt=0.008, 0.004, 0.001, 0.00025` day、解析解 69.788 %）。
  刻みを4分の1にすると差が約4分の1になり、後退Eulerの1次精度と整合します。
- 二次元化: 帯荷重を十分広く取ると2次元解が1次元解に一致（沈下の相対差 `1.7e-5`、`e` の最大差 `1.7e-5`）。
- 線形モデルとの比較: 小ひずみ極限で $U$ の差が `nx=nz=11, 21, 41` で `+2.7, +1.4, +0.6` ポイント（`T=0.1`）。
  格子を細かくすると差が半分ずつになり、離散化の違いが収束することを確認します。
- 二次元既定例（21×21、730ステップ）: 中央沈下 `376.5 mm`、間隙比 `1.247〜1.906`、消散率 `100.1 %`、自動分割なし。
- 入力異常、収束失敗の `RuntimeError`、リセット、1次元・2次元の画像出力、両入口スクリプトの完走を検証します。

GitHub Actions が push と pull_request で、Linux 上の Python 3.13・3.15 の同じテストを実行します。
実施環境・Issue・レビューの記録は [MAINTENANCE.md](MAINTENANCE.md) にあります。

## 未検証の範囲

- 対話的バックエンドでの画面表示経路（この環境では `MPLBACKEND=Agg` のみ実行）。
- macOS 以外のOS、GPU、実地盤データでの実行。
- 原著論文との逐語的な一致、三次元化、せん断をともなう二次元圧密。

## ドキュメント

理論の導出と、参照文献の確認状況は [docs/](docs/) にあります。

- [docs/theory.md](docs/theory.md): 物質座標、材料則、貯留式の導出、数値解法、二次元化の位置づけ。
- [docs/references.md](docs/references.md): 参考論文と資料の一覧、用途、確認した範囲。

## 参考文献

- [Pane and Schiffman (1981), Comparison Between Two Theories of Finite Strain Consolidation](https://doi.org/10.3208/sandf1972.21.4_81): 三笠理論が有限ひずみ圧密を扱うことを確認する資料。今回確認したのは要約・書誌情報です。
- [USACE (1990), EM 1110-1-1904, Settlement Analysis, Table C-1](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf): 帯荷重の鉛直応力式。
- [MIT (2014), 18.303 Lecture 21](https://ocw.mit.edu/courses/18-303-linear-partial-differential-equations-analysis-and-numerics-fall-2014/1788556bb2a2f52f03bdcfc720987d1e_MIT18_303F14_Lecture21.pdf): 拡散方程式とCrank–Nicolson法の符号・安定性。
- [NumPy: `numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html): 対称行列の固有値分解。

各文献の用途と、本文・要約・書誌のどこまでを確認したかは [docs/references.md](docs/references.md) にあります。

## ライセンス

MIT ライセンスです。全文は [LICENSE](LICENSE) を参照してください。
