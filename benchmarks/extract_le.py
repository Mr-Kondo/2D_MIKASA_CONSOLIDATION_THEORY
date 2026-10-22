"""Extract measured markers from Le (2015), not the plotted EVP predictions.

Run with Python 3.13: uv run --with pymupdf==1.26.7 python benchmarks/extract_le.py PDF
The source PDF is obtained separately; see data/le_2015_source.json.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATA_DIR = Path(__file__).with_name("data")


def axis_values(x, y, panel):
    left, top, right, bottom = panel["axes_pdf_pt"]
    minutes = 10 ** (-1 + 6 * (x - left) / (right - left))
    low, high = panel["vertical_limits"]
    return minutes, low + (high - low) * (y - top) / (bottom - top)


def measured_markers(page, panel):
    markers = []
    for drawing in page.get_drawings():
        fill, box = drawing.get("fill"), drawing["rect"]
        if (fill is None or len(drawing["items"]) != 4
                or not (3 < box.width < 6 and 3 < box.height < 6)):
            continue
        if any(abs(actual - expected) > 0.002 for actual, expected in zip(fill, panel["fill_rgb"])):
            continue
        x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        minutes, ordinate = axis_values(x, y, panel)
        if panel["quantity"] == "strain_percent":
            inside = 0.1 < minutes < 1e5 and 27 < ordinate < (40.5 if minutes < 1e4 else 42)
        else:
            inside = minutes > 50 and 0 < ordinate < 40 and y < panel["axes_pdf_pt"][3]
        if inside:
            markers.append((x, y, minutes, ordinate))
    if not markers or len({point[:2] for point in markers}) != len(markers):
        raise ValueError("measured markers are absent or duplicated")
    return sorted(markers)


def main():
    import pymupdf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    source = json.loads((DATA_DIR / "le_2015_source.json").read_text())
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("source PDF hash differs; inspect the figures before changing the extraction")
    with pymupdf.open(args.pdf) as pdf:
        for panel in source["panels"]:
            markers = measured_markers(pdf[panel["pdf_page"] - 1], panel)
            with (DATA_DIR / panel["csv"]).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("pdf_x_pt", "pdf_y_pt", "time_min", panel["quantity"]))
                writer.writerows(tuple(f"{v:.8g}" for v in marker) for marker in markers)
            print(f"{panel['csv']}: {len(markers)} measured markers")


if __name__ == "__main__":
    main()
