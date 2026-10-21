# 2D Consolidation — 排水・拘束圧密の簡易モデル

一定係数の2次元水圧拡散を解き、拘束圧密の弾性係数で沈下へ換算する教材用モデルです。**三笠の有限ひずみ圧密理論や、変形と水圧を連成させた2次元圧密解析の実装ではありません。**

従来の `MikasaConsolidationSimulator` と `TerzaghiConsolidationSimulator` は、同じ `ConsolidationSimulator` を指す互換用の名前です。旧実装の鉛直ひずみ係数は数式上同じであり、別途加えていた水平変位はその拘束条件と整合しませんでした。

## 導入と実行

Python 3.13以上と [uv](https://docs.astral.sh/uv/) が必要です。計算はCPU上のNumPyで行い、MLX・GPU・tqdmは使いません。

```bash
uv sync --locked
uv run --locked Mikasa.py
uv run --locked Terzaghi.py
```

- `Mikasa.py`: `mikasa_consolidation_result.png` を生成します。
- `Terzaghi.py`: `terzaghi_consolidation_result.png` と `terzaghi_mikasa_comparison.png` を生成します。
- 画像は実行時の作業ディレクトリへ保存し、同名の既存画像を上書きします。通常は画面にも表示します。
- 比較画像の2系列は同じモデルです。異なる圧密理論の比較を表しません。

画像を画面に表示しない環境では、`MPLBACKEND=Agg` を設定してください。

```python
from consolidation import ConsolidationSimulator

simulator = ConsolidationSimulator(
    length=20.0, depth=10.0, nx=51, nz=26,
    cv=1.0, E=5000.0, poisson_ratio=0.3,
    dt=0.01, total_time=10.0,
    load_intensity=100.0, load_width=4.0,
)
settlement = simulator.run_simulation(verbose=False)
simulator.plot_results(save_path="result.png", show=False)
settlement, horizontal, pressure, strain = simulator.get_current_state()
```

## 計算するもの

単位は長さm、時間day、応力kPaです。`cv` はm²/day、`E` はkPaです。`z=0` が地表、`z=depth` が固定底面で、沈下は下向きを正とします。

### 初期応力と水圧

中央の等分布帯荷重に対し、弾性半無限体の鉛直応力増分 $\sigma_0$ を与えます。荷重幅を $2a$、荷重中心からの距離を $X$ とすると、

$$
\theta=\operatorname{atan2}(X+a,z),\quad
\beta=\operatorname{atan2}(X-a,z),\quad
\alpha=\theta-\beta,
$$

$$
\sigma_0(X,z)=\frac{q}{\pi}\left[\alpha+\sin\alpha\cos(\theta+\beta)\right].
$$

地表の荷重内は $q$、荷重外は0、荷重端は $q/2$ です。旧実装の経験的な減衰式と深さへの `0.01` 加算を置き換えています。

内部の初期水圧を $u_0=\sigma_0$ と仮定し、地表は最初から排水条件 $u=0$ にします。この初期水圧の仮定は、2次元の非排水応力解析から導いたものではありません。初期状態のひずみも、この排水境界を反映します。

### 水圧の消散

$$
\frac{\partial u}{\partial t}=c_v\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial z^2}\right).
$$

地表は排水、底面・左右端は非排水です。空間差分には非排水端で鏡映点を使い、時間方向には Crank–Nicolson 法を使います。ラプラシアンを $L$ とすると、正しい符号は

$$
(I-\tfrac12c_v\Delta tL)u^{n+1}=(I+\tfrac12c_v\Delta tL)u^n
$$

です。一定係数・固定境界なので、重み付き対称行列の固有値分解から各モードの増幅率を求め、$N$ ステップを累乗でまとめて評価します。反復線形解法や時間ループは不要です。

`dt` は時間刻みの上限です。各モードの増幅率 $(1+h)/(1-h)$（$h=\tfrac12c_v\Delta t\lambda$）が非負になるよう、$\Delta t c_v(1/\Delta x^2+1/\Delta z^2)\le 1/2$ を満たす刻みに縮めます。$\lambda$ の下限側の見積もり $\lambda_{\max}\le 4c_v(1/\Delta x^2+1/\Delta z^2)$ を使っているため、この条件で全モードの $h$ は $-1$ 以上になります。整数ステップ数を決めた後で `total_time / nt` に調整し、端数時間を切り捨てません。実際の刻みは `simulator.dt` で確認できます。

水圧・ひずみを任意の上限でクリップしません。非有限の水圧は `FloatingPointError` として報告します。

### ひずみと沈下

各鉛直柱に $\varepsilon_x=\varepsilon_y=0$ の換算を適用します。

$$
D=\frac{E(1-\nu)}{(1+\nu)(1-2\nu)},\qquad
\varepsilon_z=\frac{\sigma_0-u}{D},\qquad
w(z)=\int_z^H\varepsilon_z(\zeta)\,d\zeta.
$$

積分は台形則です。固定底面の沈下は0、地表の沈下は全層のひずみ積分です。水平変位は0です。局所的に $u>\sigma_0$ となった場合の負のひずみも、任意に消去しません。

これは水圧拡散の結果を局所的に換算する近似です。せん断応力・水平力の釣合いや、空間的に変化する沈下場の完全な弾性適合条件を解くものではありません。

## APIと制約

- `nx`, `nz`: 2以上の整数。`length`, `depth`, `E`, `dt`, `total_time`, `load_width` は有限の正数です。
- `cv`, `load_intensity`: 有限の非負数。`0 <= poisson_ratio < 0.5` を指定します。数値でない値（文字列・`None`）と `True`/`False` は数値引数として受理しません。不正な入力には `ValueError`、演算範囲を超えた条件には `FloatingPointError` または `ValueError` が発生します。
- `device`: `"auto"` または `"cpu"`。どちらもCPUを使用します。従来無視されていた `"mps"`, `"cuda"`, `"mlx"` は明示的に拒否します。
- `run_simulation()`: 現在の水圧から `total_time` だけ進め、形状 `(nz, nx)` の沈下配列を返します。再度呼ぶと、さらに同じ時間だけ進みます。
- `get_current_state()`: 沈下、水平変位、水圧、鉛直ひずみの順で、NumPy配列のコピーを返します。
- `get_consolidation_degree()`: 初期の境界条件適用後の水圧積分に対する消散率[%]です。沈下量に基づく圧密度とは区別します。初期水圧積分が0なら100%とします。
- `reset()`: 初期水圧・ひずみ・変位を復元します。
- `plot_results(save_path=None, show=True, figsize=(18, 5), dpi=300)`: 沈下・水平変位・水圧を描画します。保存・表示後には図を閉じます。非対話バックエンド（Agg など）では `plt.show()` を呼ばず、警告も出しません。値が一定、または荷重・層厚に対して無視できる場は、丸め誤差のスケールをカラーバーにせず、値をパネル内に注記します。

生成後に格子・材料・時間・荷重条件を変更するときは、新しいインスタンスを作ってください。一定係数の固定格子を前提とした解法なので、時間変動荷重、不均一な透水性、間隙比依存の材料則、大変形にはそのまま適用できません。実地盤への適合性・設計精度は検証していません。

## 検証

```bash
uv run --locked python -m unittest -v
```

連続方程式の解析解への格子収束、時間刻みの2次収束、小格子での直接Crank–Nicolson解との一致、帯荷重の応力、固定底面、ゼロ荷重、入力異常、リセット、両入口と画像出力を検証します。実施環境・Issue・最小性の3回のレビューは [MAINTENANCE.md](MAINTENANCE.md) に記録します。

## 参考文献

- [Pane and Schiffman (1981), Comparison Between Two Theories of Finite Strain Consolidation](https://doi.org/10.3208/sandf1972.21.4_81): 三笠理論が有限ひずみ圧密を扱うことを確認する資料。今回確認したのは要約・書誌情報です。
- [USACE (1990), EM 1110-1-1904, Settlement Analysis, Table C-1](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf): 帯荷重の鉛直応力式。
- [MIT (2014), 18.303 Lecture 21](https://ocw.mit.edu/courses/18-303-linear-partial-differential-equations-analysis-and-numerics-fall-2014/1788556bb2a2f52f03bdcfc720987d1e_MIT18_303F14_Lecture21.pdf): 拡散方程式とCrank–Nicolson法の符号・安定性。
- [NumPy: `numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html): 対称行列の固有値分解。
