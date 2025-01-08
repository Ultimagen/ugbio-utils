import sys
from pathlib import Path

from ugbio_core.reports.report_utils import generate_report

from ugbio_methylation.concat_methyldackel_csvs import (
    concat_methyldackel_csvs,
    parse_args,
)
from ugbio_methylation.globals import (
    HTML_REPORT,
    TEMPLATE_NOTEBOOK,
    MethylDackelConcatenationCsvs,
)


def generate_methylation_report(
    methyl_dackel_concatenation_csvs: MethylDackelConcatenationCsvs, base_file_name, output_prefix: Path = None
) -> Path:
    h5_file = concat_methyldackel_csvs(methyl_dackel_concatenation_csvs, base_file_name, output_prefix)

    paramaters = {"input_h5_file": h5_file, "input_base_file_name": base_file_name}

    output_report_html = Path(base_file_name + HTML_REPORT)
    if output_prefix:
        output_report_html = output_prefix / output_report_html

    generate_report(
        template_notebook_path=TEMPLATE_NOTEBOOK, parameters=paramaters, output_report_html_path=output_report_html
    )

    return output_report_html


def main(argv: list[str] | None = None):
    if argv is None:
        argv: list[str] = sys.argv
    methyl_dackel_concatenation_csvs, base_file_name = parse_args(argv)

    generate_methylation_report(methyl_dackel_concatenation_csvs, base_file_name)


if __name__ == "__main__":
    main()
