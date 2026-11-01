import pandas as pd
from ugbio_consensus import consensus_report


def test_sample_names_values_and_title_are_escaped():
    stats = pd.DataFrame({"sample": ["<script>alert(1)</script>"], "n_reads": [5.0]})
    table = consensus_report._build_table_html(stats, "<b>t</b>")
    page = consensus_report._assemble_html("<img src=x onerror=alert(1)>", [table], [])
    assert "<script>" not in page and "<img src=x" not in page and "<b>t</b>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
