# 参考論文と資料

本実装が参照した論文と資料を、書誌情報、用途、確認した範囲に分けて記録します。確認は2026年10月22日時点です。
**一次資料の本文を読んだかどうかを区別し、読んでいないものを読んだとは扱いません。**

2026-10-22の比較候補の再調査（梅原・善1979、小林1982、高田1983、Morris 2002、Fox・Pu 2015、Puら2018、Songら2024）と、各本文の確認範囲・論文別Issueは[論文ごとの計算比較](literature-validation.md)に記録しています。

## 確認状況の区分

| 区分 | 意味 |
| --- | --- |
| 本文 | 本文または公開文書を確認した。 |
| 要約 | 公開された要約・索引を確認した。本文は未確認。 |
| 書誌 | 出版社やCrossrefの書誌データで存在と属性を確認した。本文は未確認。 |
| 参照リスト | 他文献の参考文献リストで書誌を確認した。一次資料は未確認。 |
| API文書 | 公式ドキュメントを確認した。 |

## 1. 有限ひずみ圧密の理論

### Mikasa (1963) — 原著

- M. Mikasa, *The Consolidation of Soft Clay — A New Consolidation Theory and Its Application*, Kajima Institution Publishing Co., Ltd., Tokyo, 1963（日本語）。
- 用途: 本実装が目標とする理論。
- 解説: TRIDの要約は「Terzaghi理論が適用できない軟弱埋立粘土の安定化の圧密試験をもとに、より一般化した圧密式を立てた。Terzaghi式の過剰間隙水圧の代わりに圧縮ひずみを用いる」と述べます。未知数を水圧ではなく間隙比の変化に取る点が、本実装の定式化と共通します。
- 確認状況: **要約と書誌**（TRIDの要約、複数の公開文献の参考文献リストで書誌を確認）。**本文は未確認。** 単行本のためDOIは確認できていません。
- TRID: <https://trid.trb.org/view/119549>

### Mikasa (1965)

- M. Mikasa, "The consolidation of soft clay", *Civil Engineering in Japan*, Japan Society of Civil Engineers, 1965, pp. 21-26.
- 用途: 1963の理論を英語で示したもの。
- 確認状況: 参照リスト。

### Mikasa, Takada, Oshima and Kiyama (1998)

- M. Mikasa, N. Takada, A. Oshima and M. Kiyama, "Nonlinear Consolidation Theory for Nonhomogeneous Clay Layers and its Application", *Soils and Foundations*, Vol. 38, No. 4, pp. 205-212, 1998.
- 用途: 三笠による非線形圧密理論の展開。非均質な粘土層への適用。
- 確認状況: 書誌。DOI: [10.3208/sandf.38.4_205](https://doi.org/10.3208/sandf.38.4_205)。

### Gibson, England and Hussey (1967)

- R. E. Gibson, G. L. England and M. J. L. Hussey, "The Theory of One-Dimensional Consolidation of Saturated Clays", *Géotechnique*, Vol. 17, No. 3, pp. 261-273, 1967.
- 用途: 有限ひずみ（非線形）圧密を薄い一様層について定式化したもの。物質座標と間隙比依存の材料則を使う点が、三笠理論と同じ系譜です。
- 確認状況: 書誌。DOI: [10.1680/geot.1967.17.3.261](https://doi.org/10.1680/geot.1967.17.3.261)。

### Gibson, Schiffman and Cargill (1981)

- R. E. Gibson, R. L. Schiffman and K. W. Cargill, "The theory of one-dimensional consolidation of saturated clays. II. Finite nonlinear consolidation of thick homogeneous layers", *Canadian Geotechnical Journal*, Vol. 18, No. 2, pp. 280-293, 1981.
- 用途: 厚い層の有限非線形圧密。層厚の変化が大きい場合の扱い。
- 確認状況: 書誌。DOI: [10.1139/t81-030](https://doi.org/10.1139/t81-030)。

### Schiffman (1980)

- R. L. Schiffman, "Finite and infinitesimal strain consolidation", *Journal of the Geotechnical Engineering Division*, ASCE, Vol. 106, No. GT2, Proceedings Paper 15193, pp. 203-207, 1980.
- 用途: 有限ひずみ圧密と無限小ひずみ圧密の関係。
- 確認状況: 参照リスト（Pane and Schiffman (1981) の引用）。

### Pane and Schiffman (1981)

- V. Pane and R. L. Schiffman, "A Comparison Between Two Theories of Finite Strain Consolidation", *Soils and Foundations*, Vol. 21, No. 4, pp. 81-84, 1981.
- 用途: 三笠理論とGibson理論を比較し、両者が同じ有限ひずみ圧密を扱うことを示す資料。本実装では、三笠理論が有限ひずみを扱うことの確認に使いました。
- 確認状況: **要約と書誌**。本文は未確認。DOI: [10.3208/sandf1972.21.4_81](https://doi.org/10.3208/sandf1972.21.4_81)。

### 有限ひずみ・自重圧密の関連文献

- R. L. Schiffman and V. Pane, "Non-linear Finite Strain Consolidation of Soft Marine Sediments", *Seabed Mechanics*, pp. 123-130, 1984. 確認状況: 書誌。DOI: [10.1007/978-94-009-4958-4_14](https://doi.org/10.1007/978-94-009-4958-4_14)。
- V. Pane and R. L. Schiffman, "A note on sedimentation and consolidation", *Géotechnique*, Vol. 35, No. 1, pp. 69-72, 1985. 確認状況: 書誌。DOI: [10.1680/geot.1985.35.1.69](https://doi.org/10.1680/geot.1985.35.1.69)。
- N. Takada and M. Mikasa, "Determination of Consolidation Parameters by Selfweight Consolidation Test in Centrifuge", *Consolidation of Soils: Testing and Evaluation*, ASTM, pp. 548-566, 1986. 遠心模型の自己重圧密試験。自重圧密の考え方に関係します。確認状況: 書誌。DOI: [10.1520/stp34634s](https://doi.org/10.1520/stp34634s)。
- N. Takada, "Numerical Analysis of Selfweight Consolidation of Very Soft Clay", *Proceedings of the Japan Society of Civil Engineers*, No. 334, 1983. 超軟弱粘土の自己重圧密の数値解析。確認状況: 書誌。DOI: [10.2208/jscej1969.1983.334_113](https://doi.org/10.2208/jscej1969.1983.334_113)。

## 2. 線形小ひずみの参照と比較

### Terzaghi (1943)

- K. Terzaghi, *Theoretical Soil Mechanics*, Wiley, 1943.
- 用途: 一次元圧密理論。本実装が小ひずみ極限で一致することを確認する基準（解析解）です。
- 確認状況: 書誌。DOI: [10.1002/9780470172766](https://doi.org/10.1002/9780470172766)。

### Biot (1941)

- M. A. Biot, "General Theory of Three-Dimensional Consolidation", *Journal of Applied Physics*, Vol. 12, No. 2, pp. 155-164, 1941.
- 用途: 変形と水圧を連成させた三次元圧密の理論。**本実装はこれを解きません。** 対比のために挙げます。
- 確認状況: 書誌。DOI: [10.1063/1.1712886](https://doi.org/10.1063/1.1712886)。

## 3. 数値解法

### Crank and Nicolson (1947)

- J. Crank and P. Nicolson, "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type", *Mathematical Proceedings of the Cambridge Philosophical Society*, Vol. 43, No. 1, pp. 50-67, 1947.
- 用途: 線形モデル（`src/mikasa_consolidation/linear.py`）の時間積分。増幅率の符号と安定性の確認。
- 確認状況: 書誌。DOI: [10.1017/S0305004100023197](https://doi.org/10.1017/S0305004100023197)。

### MIT 18.303 Lecture 21 (2014)

- MIT OpenCourseWare, *18.303 Linear Partial Differential Equations*, Lecture 21, 2014.
- 用途: 拡散方程式とCrank–Nicolson法の符号・安定性の確認。
- 確認状況: 本文（公開PDF）。<https://ocw.mit.edu/courses/18-303-linear-partial-differential-equations-analysis-and-numerics-fall-2014/1788556bb2a2f52f03bdcfc720987d1e_MIT18_303F14_Lecture21.pdf>

### NumPy

- `numpy.linalg.eigh`（対称行列の固有値分解）<https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>
- `numpy.log1p`（1に近い増幅率の評価）<https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>
- 確認状況: API文書。

## 4. 荷重の応力式

### USACE (1990) EM 1110-1-1904

- U.S. Army Corps of Engineers, *Settlement Analysis*, Engineer Manual EM 1110-1-1904, 1990, Table C-1.
- 用途: 帯荷重の鉛直応力増分の式。`src/mikasa_consolidation/finite_strain.py` の `sigma_load` と `src/mikasa_consolidation/linear.py` が同じ式を使います。
- 確認状況: 索引（検索索引にある表）。公式PDFの直接取得はHTTP 403で失敗しました。式は、線荷重の独立な数値積分とも一致させて確認しています。<https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-1-1904.pdf>

## 5. 確認できていないこと

- 三笠 (1963) の本文（日本語の単行本）。確認したのは要約と書誌のみです。
- Mikasa (1965) の本文、Schiffman (1980) の原本。
- Gibson (1967, 1981)、Pane and Schiffman (1981, 1985)、Schiffman and Pane (1984)、Takada and Mikasa (1986)、Takada (1983)、Mikasa et al. (1998) の各本文。
- USACE EM 1110-1-1904 の公式PDF原本。
- 原著との逐語的な一致。本実装の定式化が三笠理論の表記と一致するかは、本文を確認するまで判断しません。

## 6. 二次元化の位置づけ

平面ひずみ二次元への拡張は著者によるもので、上記のどの文献にも基づきません。三笠が公表した理論ではありません。
導出と限界は [theory.md](theory.md) の9節と11節にあります。
