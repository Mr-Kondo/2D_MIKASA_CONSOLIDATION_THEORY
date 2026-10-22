# ドキュメント

ルートの `README.md` は概要と使い方を示します。この `docs/` は、理論の導出と参照文献の確認状況を詳しく扱います。

- [theory.md](theory.md) — 有限ひずみ圧密の理論と数値解法の解説。
- [references.md](references.md) — 参考論文と資料の一覧、用途、確認した範囲。
- [maintenance.md](maintenance.md) — Issueごとの対応と検証の記録。
- [literature-validation.md](literature-validation.md) — 論文別のIssue、数値解との照合、条件対応と未確認事項。
- [experimental-validation.md](experimental-validation.md) — クリープを含む室内試験への探索的適用、差と仮定。
- [colab.md](colab.md) — Colab CUDA・ローカルJupyterの実行、コンター図表示、CPU比較。
- [validation.md](validation.md) — GUI・OSの確認範囲と、Apple M5でのGPU併用実験。

本文の数値・記号・式は、リポジトリの実装（`src/mikasa_consolidation/`）に合わせています。実装を変えたときは、この文書も更新してください。

> 二次元化（平面ひずみ）は著者による拡張で、三笠が公表した理論ではありません。原著（三笠 1963 など）の本文は確認していません。逐語的な一致は主張しません。
