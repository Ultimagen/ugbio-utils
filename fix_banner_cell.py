"""Write the banner cell content to both MRD report notebooks."""
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def save(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)


def find(nb, cid):
    for i, c in enumerate(nb["cells"]):
        if c["id"] == cid:
            return i, c
    return None, None


def set_src(cell, text):
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines]
    cell["source"][-1] = cell["source"][-1].rstrip("\n")


# NOTE: use double-quotes for font-family values to avoid single-quote escaping issues
BANNER_BODY = """\
from IPython.display import HTML, display
import datetime

if detection.detected is True:
    banner_color = "#c0392b"; banner_bg = "#fdedec"; banner_border = "#e74c3c"
elif detection.detected is False:
    banner_color = "#1e8449"; banner_bg = "#eafaf1"; banner_border = "#2ecc71"
else:
    banner_color = "#b7950b"; banner_bg = "#fef9e7"; banner_border = "#f1c40f"

lod_str = format_scientific(detection.personal_lod) if detection.personal_lod else "N/A"
vaf_str = format_scientific(detection.matched_ctdna_vaf) if detection.matched_ctdna_vaf > 0 else "0"
report_date = datetime.date.today().isoformat()
ep = detection.p_value
emp_p_str = f"{ep:.3f}" if ep >= 0.001 else f"{ep:.2e}"
fp = detection.fitted_p_value
fit_p_str = (f"{fp:.3f}" if fp >= 0.001 else f"{fp:.2e}") if fp is not None else None
fit_p_html = (
    "<div>"
    "<div style=\\"font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px;\\">p-value (fitted)</div>"
    f"<div style=\\"font-size:18px;font-weight:700;color:#2c3e50;\\">{fit_p_str}</div>"
    f"<div style=\\"font-size:11px;color:#7f8c8d;\\">{detection.fitted_distribution}-fitted null</div>"
    "</div>"
) if fit_p_str is not None else ""

# margin: 0 -32px breaks out of .jp-RenderedHTML padding so header spans full page width
html = (
    "<div style=\\"margin:0 -32px;\\">"
    "<div style=\\"background:#1a3a5c;color:#fff;padding:24px 32px;"
    "display:flex;align-items:center;justify-content:space-between;"
    "font-family:system-ui,sans-serif;\\">"
    "<div>"
    "<div style=\\"font-size:22px;font-weight:700;color:#1abc9c;letter-spacing:1px;margin-bottom:4px;\\">ULTIMA GENOMICS</div>"
    f"<div style=\\"font-size:18px;font-weight:600;letter-spacing:.3px;\\">{report_title}</div>"
    "<div style=\\"font-size:11px;color:rgba(255,255,255,.65);margin-top:2px;\\">Tumor-Informed Minimal Residual Disease Detection &middot; Whole Genome Sequencing</div>"
    "</div>"
    f"<div style=\\"text-align:right;font-size:11px;color:rgba(255,255,255,.7);line-height:1.9;\\">"
    f"<strong style=\\"color:#fff;\\">Sample</strong> {basename or 'N/A'}<br>"
    f"<strong style=\\"color:#fff;\\">Report Date</strong> {report_date}"
    "</div></div>"
    "<div style=\\"padding:20px 32px 0;font-family:system-ui,sans-serif;\\">"
    "<div style=\\"font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;"
    "color:#7f8c8d;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #dde1e7;\\">Detection Result</div>"
    f"<div style=\\"background:{banner_bg};border:1.5px solid {banner_border};"
    f"border-left:5px solid {banner_color};border-radius:6px;"
    "padding:16px 22px;display:flex;align-items:center;gap:24px;flex-wrap:wrap;\\">"
    f"<div style=\\"background:{banner_color};color:white;font-size:13px;font-weight:700;"
    "padding:6px 14px;border-radius:4px;text-transform:uppercase;letter-spacing:.8px;white-space:nowrap;\\">"
    f"{detection.call}</div>"
    "<div style=\\"display:flex;gap:24px;flex-wrap:wrap;\\">"
    "<div><div style=\\"font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px;\\">Tumor Fraction</div>"
    f"<div style=\\"font-size:18px;font-weight:700;color:#2c3e50;\\">{vaf_str}</div></div>"
    "<div><div style=\\"font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px;\\">p-value (empirical)</div>"
    f"<div style=\\"font-size:18px;font-weight:700;color:#2c3e50;\\">{emp_p_str}</div>"
    f"<div style=\\"font-size:11px;color:#7f8c8d;\\">{detection.n_synthetic_controls} synthetic controls</div></div>"
    f"{fit_p_html}"
    "<div><div style=\\"font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px;\\">Personal LOD (95%)</div>"
    f"<div style=\\"font-size:18px;font-weight:700;color:#2c3e50;\\">{lod_str}</div>"
    "<div style=\\"font-size:11px;color:#7f8c8d;\\">95% detection power</div></div>"
    "<div><div style=\\"font-size:10px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px;\\">cfDNA Supporting Reads</div>"
    f"<div style=\\"font-size:18px;font-weight:700;color:#2c3e50;\\">{detection.matched_supporting_reads}</div>"
    f"<div style=\\"font-size:11px;color:#7f8c8d;\\">noise median: {detection.null_median_reads:.1f}</div></div>"
    + (
        f"<div style=\\"margin-top:10px;padding:8px 12px;background:#fef9e7;border:1px solid #f1c40f;"
        f"border-radius:4px;font-size:11px;color:#7f8c8d;\\">"
        f"&#9888; Only {detection.n_synthetic_controls} synthetic controls &mdash; p-value reliability is reduced."
        "</div>"
        if detection.n_synthetic_controls < 20 else ""
    )
    + "</div></div>"
    + "</div></div>"
)
display(HTML(html))
"""

RESULTS_PREFIX = """\
report_title = "MRD Analysis Report"
"""

QC_PREFIX = """\
report_title = "MRD QC Report"
"""

BASE = Path("src/mrd/ugbio_mrd/reports")

# Results report
nb = load(BASE / "mrd_results_report.ipynb")
_, c = find(nb, "863cb689")
set_src(c, RESULTS_PREFIX + BANNER_BODY)
save(nb, BASE / "mrd_results_report.ipynb")
print("Results report banner updated")

# QC report
nb = load(BASE / "mrd_qc_report.ipynb")
_, c = find(nb, "863cb689")
set_src(c, QC_PREFIX + BANNER_BODY)
save(nb, BASE / "mrd_qc_report.ipynb")
print("QC report banner updated")

# Quick syntax check
import ast
for path in [BASE / "mrd_results_report.ipynb", BASE / "mrd_qc_report.ipynb"]:
    nb = load(path)
    for c in nb["cells"]:
        if c["id"] == "863cb689":
            src = "".join(c["source"])
            try:
                ast.parse(src)
                print(f"  {path.name}: syntax OK")
            except SyntaxError as e:
                print(f"  {path.name}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
                # print the offending line
                lines = src.split("\n")
                if e.lineno:
                    print(f"    >>> {lines[e.lineno - 1]!r}")
