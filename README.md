# 2D Mikasa Consolidation Theory

三笠法による二次元圧密沈下シミュレーター

## 概要

このプロジェクトは、側方変位を考慮した三笠の2次元圧密理論（Mikasa, 1965）を実装したシミュレーターです。Apple SiliconのGPU加速を活用するため、MLXフレームワークを使用して最適化されています。

### 主な特徴

- **MLXフレームワーク**: Apple Silicon（M1/M2/M3など）のGPU加速に対応
- **Crank-Nicolson陰解法**: 無条件安定な時間積分スキーム
- **Jacobi反復法**: 線形方程式の安定した求解
- **側方変位考慮**: 三笠法による realistic な変形計算
- **可視化機能**: 沈下量・水平変位・過剰間隙水圧の分布図を自動生成

## 理論背景

### 三笠の2次元圧密理論

従来の1次元圧密理論では側方変位を無視していましたが、三笠法では以下の特徴があります：

- 側方変位を考慮した応力-ひずみ関係
- 2次元の過剰間隙水圧消散
- より realistic な沈下予測

### 支配方程式

#### 過剰間隙水圧の消散

$$
\frac{\partial u}{\partial t} = c_v \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial z^2} \right)
$$

ここで：
- $u$: 過剰間隙水圧 (kPa)
- $c_v$: 圧密係数 (m²/day)
- $t$: 時間 (day)
- $x, z$: 空間座標 (m)

#### ひずみと変位の計算（三笠法の核心）

三笠法では、有効応力の増加からひずみを計算し、それを積分して変位を求めます：

**1. 有効応力増加**

$$
\Delta\sigma'_{\text{eff}} = u_{\text{initial}} - u(t)
$$

消散した過剰間隙水圧が有効応力の増加に等しい。

**2. 静止土圧係数**

$$
K_0 = \frac{\nu}{1 - \nu}
$$

**3. 鉛直ひずみ（側方変位を考慮）**

$$
\varepsilon_z = \frac{\Delta\sigma'_{\text{eff}}}{E} \times (1 - 2\nu K_0)
$$

ここで：
- $\varepsilon_z$: 鉛直ひずみ
- $E$: ヤング率 (kPa)
- $\nu$: ポアソン比
- $K_0$: 静止土圧係数

係数 $(1 - 2\nu K_0)$ が**三笠法の特徴**で、側方変位の影響を考慮しています。

**4. 鉛直変位（沈下量）**

$$
w(z) = \int_0^z \varepsilon_z \, dz
$$

各深さでのひずみを積分することで、沈下量分布を求めます。

**5. 水平変位**

$$
\varepsilon_x \approx -\nu \times \varepsilon_z
$$

$$
u_x(x) = \varepsilon_x \times \frac{x - x_{\text{center}}}{2}
$$

ポアソン効果による側方変位を簡易的に計算します。

## インストール

### 必要環境

- Python 3.8以上
- macOS (Apple Silicon推奨)
- uv (Pythonパッケージマネージャー)

### 依存パッケージ

```bash
uv sync
```

主要パッケージ:
- `mlx>=0.0.1` - Apple Silicon GPU加速
- `numpy>=2.4.1` - 数値計算
- `matplotlib>=3.10.8` - 可視化
- `tqdm>=4.67.1` - 進捗表示

## 使用方法

### 基本的な実行

```bash
uv run ./main.py
```

### カスタマイズ例

```python
from main import MikasaConsolidationSimulator

# シミュレーター初期化
simulator = MikasaConsolidationSimulator(
    length=20.0,           # 土層の横幅 (m)
    depth=10.0,            # 土層の深さ (m)
    nx=51,                 # x方向の格子点数
    nz=26,                 # z方向の格子点数
    cv=1.0,                # 圧密係数 (m²/day)
    E=5000.0,              # ヤング率 (kPa)
    dt=0.01,               # 時間刻み (day)
    total_time=3650.0,     # 総シミュレーション時間 (10年)
    load_intensity=100.0,  # 荷重強度 (kPa)
    load_width=4.0,        # 荷重幅 (m)
    poisson_ratio=0.3,     # ポアソン比
)

# シミュレーション実行
result = simulator.run_simulation(verbose=True)

# 結果の可視化
simulator.plot_results(save_path="result.png")
```

## 出力結果

### コンソール出力例

```
Running on: mlx (Apple Silicon with MLX)
Implicit method coefficients:
α_x = 0.062500
α_z = 0.062500
α_x + α_z = 0.125000
  Method: Crank-Nicolson (Unconditionally stable)
============================================================
Mikasa 2D Consolidation Simulation (10 years)
Implicit Method (Crank-Nicolson) with GPU Optimization
============================================================
Mikasa Consolidation: 100%|██████████| 365000/365000 [23:58<00:00, 253.77it/s]
Simulation time: 1438.293 seconds
Max settlement: 59.95 mm
Surface center settlement: 5.94 mm
Max horizontal displacement: 202.56 mm
Remaining excess pore pressure: 110.00 kPa
Consolidation degree: -10.0%
============================================================
```

### 可視化

シミュレーション終了後、以下の3つのプロットを含む画像が生成されます：

1. **鉛直変位分布** (Vertical Displacement)
2. **水平変位分布** (Horizontal Displacement)
3. **過剰間隙水圧分布** (Excess Pore Pressure)

## 技術的詳細

### 数値解法

#### Crank-Nicolson法

時間積分に無条件安定なCrank-Nicolson陰解法を使用：

$$
(I + 0.5 \Delta t L) u_{n+1} = (I - 0.5 \Delta t L) u_n
$$

ここで $L$ はラプラシアン演算子です。

#### Jacobi反復法

線形方程式の求解にJacobi反復法を採用：

$$
x^{k+1} = \text{rhs} - 0.5 L x^k
$$

共役勾配法よりも条件数に強く、数値安定性が高い特徴があります。

### 境界条件

- **上面**: $u = 0$ (排水境界)
- **下面**: $\frac{\partial u}{\partial z} = 0$ (非排水境界)
- **側面**: $\frac{\partial u}{\partial x} = 0$ (対称境界)

### 数値安定性

以下の安定化手法を実装：

- 過剰間隙水圧のクリッピング: [0, 1.1 × 荷重強度]
- ひずみのクリッピング: [-0.1, 0.1]
- NaN/Inf値の検出と処理
- 境界条件の各ステップでの再適用

## パフォーマンス

### ベンチマーク (Apple Silicon)

- **Grid**: 51 × 26 points
- **Time steps**: 365,000 steps (10 years, dt=0.01 day)
- **Execution time**: ~24 minutes
- **Iteration speed**: ~250 it/s
- **Memory**: GPU最適化により効率的

### 最適化のポイント

- MLXによるGPU並列化
- 配列操作の効率化（concatenateによる不変配列の構築）
- Jacobi法の収束判定（100反復、許容誤差1e-4）

## パラメータガイド

### 推奨値

| パラメータ | 一般的な範囲 | デフォルト値 |
|-----------|------------|------------|
| $c_v$ (圧密係数) | 0.01 - 10 m²/day | 1.0 |
| $E$ (ヤング率) | 1000 - 10000 kPa | 5000 |
| $\nu$ (ポアソン比) | 0.2 - 0.4 | 0.3 |
| $\Delta t$ (時間刻み) | 0.001 - 0.1 day | 0.01 |

### グリッド解像度

安定性係数 $\alpha = \frac{c_v \Delta t}{\Delta x^2}$ が0.5以下になるように設定することを推奨します。

## トラブルシューティング

### 数値不安定

症状: 異常に大きな値やNaNが発生

対処:
- 時間刻み $\Delta t$ を小さくする
- グリッド解像度 $n_x$, $n_z$ を増やす
- 安定性係数 $\alpha$ を確認（0.5以下推奨）

### 収束が遅い

対処:
- Jacobi法の最大反復回数 `max_iter` を増やす
- 許容誤差 `tol` を緩める
- より細かい時間刻みを使用

## ライセンス

このプロジェクトは研究・教育目的で自由に使用できます。

## 参考文献

- Mikasa, M. (1965). "The consolidation of soft clay - A new consolidation theory and its application." Japanese Society of Civil Engineers.

## 開発情報

- **フレームワーク**: MLX (Apple Machine Learning Framework)
- **言語**: Python 3.8+
- **最適化対象**: Apple Silicon (M1/M2/M3 chips)

## 更新履歴

### v1.0.0 (2026-01-30)
- MLXフレームワークへの移行完了
- Jacobi反復法による安定化
- 数値安定性の向上（クリッピング、NaN処理）
- Apple Silicon GPU加速対応
