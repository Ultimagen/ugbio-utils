"""Run the report notebooks of ugbio_comparison.reports end to end"""

from importlib.resources import files
from pathlib import Path

import nbformat
import pandas as pd
import papermill
import pytest
from nbconvert import HTMLExporter
from ugbio_comparison import sv_stats_collect

RESOURCES_DIR = Path(__file__).parent.parent / "resources"
REPORTS_DIR = files("ugbio_comparison.reports")


def run_report(notebook: str, parameters: dict, work_dir: Path) -> Path:
    """Execute a report notebook with papermill and convert it to HTML; return the HTML path"""
    output_notebook = work_dir / notebook.replace(".ipynb", ".nbconvert.ipynb")
    papermill.execute_notebook(
        str(REPORTS_DIR / notebook),
        str(output_notebook),
        parameters=parameters,
        kernel_name="python3",
        cwd=str(work_dir),
    )
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True
    body, _ = html_exporter.from_notebook_node(nbformat.read(str(output_notebook), as_version=4))
    output_html = output_notebook.with_suffix(".html")
    output_html.write_text(body)
    return output_html


@pytest.mark.parametrize(
    ("h5_concordance_file", "model_name"),
    [
        ("test.untrained.h5", "untrained_ignore_gt_excl_hpol_runs"),
        ("test.ug_hcr.sec.comp.h5", "threshold_model_recall_precision_ignore_gt_excl_hpol_runs"),
    ],
)
@pytest.mark.parametrize("verbosity", [1, 5])
def test_var_report(tmp_path, h5_concordance_file, model_name, verbosity):
    h5outfile = tmp_path / "var_report.h5"
    run_report(
        "createVarReport.ipynb",
        {
            "verbosity": verbosity,
            "run_id": "140479-BC21",
            "pipeline_version": "2.4.1",
            "h5_concordance_file": str(RESOURCES_DIR / "vc_report" / h5_concordance_file),
            "h5outfile": str(h5outfile),
            "model_name": model_name,
        },
        tmp_path,
    )
    with pd.HDFStore(h5outfile, "r") as store:
        keys = set(store.keys())
    assert {"/all_data", "/ug_hcr", "/exome"} <= keys
    if verbosity > 1:
        error_types = pd.read_hdf(h5outfile, "all_data_error_types")
        assert "hmer Indel 4" not in error_types.index


def test_sv_report(tmp_path):
    statistics_file = tmp_path / "sv.stats.pkl"
    sv_stats_collect.run(
        ["collect_sv_stats", str(RESOURCES_DIR / "GRCh38_HG2-T2TQ100-V1.1_stvar.chr9.vcf.gz"), str(statistics_file)]
    )
    h5outfile = tmp_path / "sv_report.h5"
    run_report(
        "createSVReport.ipynb",
        {
            "run_id": "test",
            "pipeline_version": "1.0",
            "statistics_file": str(statistics_file),
            "h5outfile": str(h5outfile),
        },
        tmp_path,
    )
    assert not pd.read_hdf(h5outfile, "type_counts").empty


@pytest.mark.parametrize("is_somatic", [False, True])
def test_no_gt_report(tmp_path, is_somatic):
    data_dir = RESOURCES_DIR / "no_gt_report"
    if is_somatic:
        prefix, folder = "somatic_test", "somatic_signature_plots_folder"
        annotations = "EXOME,LCR,MAP_UNIQUE,LONG_HMER,UG_HCR"
        filtered_vcf = str(data_dir / "somatic_test.ann.CHR19.vcf.gz")
    else:
        prefix, folder = "014790-NA12878", "signature_plots_folder"
        annotations = "LCR,EXOME,MAP_UNIQUE,LONG_HMER"
        filtered_vcf = "014790-NA12878.chr1_head.vcf.gz"
    h5outfile = tmp_path / "no_gt_report.h5"
    html = run_report(
        "report_wo_gt.ipynb",
        {
            "run_id": "test_run",
            "pipeline_version": "1.5.0",
            "h5_statistics": str(data_dir / f"{prefix}.filt.no_gt_stats.h5"),
            "h5_statistics_wgs": str(data_dir / f"{prefix}.filt.no_gt_stats_wgs.h5"),
            "annotation_intervals_names": annotations,
            "is_somatic": is_somatic,
            "filtered_vcf": filtered_vcf,
            # As in the pipeline, an absolute path (the notebook joins it twice to the plot names)
            "signature_plots_folder": str(data_dir / folder),
            "cosmic_signatures": str(data_dir / "somatic_test.cosmic_signatures_v3.3.json"),
            "h5outfile": str(h5outfile),
        },
        tmp_path,
    )
    parameters = pd.read_hdf(h5outfile, "parameters")
    assert parameters.loc["is_somatic", "value"] == str(is_somatic)
    if is_somatic:
        assert not pd.read_hdf(h5outfile, "sbs_activities").empty
        assert "Significant SBS Signatures" in html.read_text()
