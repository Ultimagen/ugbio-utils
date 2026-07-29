import json
from argparse import ArgumentParser
from pathlib import Path

import nbformat
import papermill
from nbconvert import HTMLExporter
from ugbio_core.logger import logger


def modify_jupyter_notebook_html(
    input_html: str,
    output_html: str = None,
    font_size: int = 24,
    font_family: str = "Arial, sans-serif",
    max_width: str = "800px",
):
    """
    Modify the style of a Jupyter notebook HTML export.

    Parameters
    ----------
    input_html : str
        Path to the input HTML file.
    output_html : str, optional
        Path to the output HTML file. If not provided, the input HTML file will be modified in-place.
    font_size : int, optional
        The desired font size in pixels. Default is 16.
    font_family : str, optional
        The desired font family. Default is "Arial, sans-serif".
    max_width : str, optional
        The maximum width of the content. Default is "700px".

    """

    # Define the CSS to insert.
    css = f"""
    body {{
      font-size: {font_size}px;
      font-family: {font_family};
      margin: 0 auto;
      max-width: {max_width};
      text-align: left;
    }}
    div.output_text {{
      font-size: {font_size}px;
      font-family: {font_family};
      text-align: left;
    }}
    """

    # Read the HTML file.
    with open(input_html, encoding="utf-8") as file:
        html = file.read()

    # Insert the CSS into the HTML.
    html = html.replace("</head>", f'<style type="text/css">{css}</style></head>')

    # Write the updated HTML back to the file.
    output_html = output_html if output_html else input_html
    with open(output_html, "w", encoding="utf-8") as file:
        file.write(html)


def generate_report(
    template_notebook_path: Path, parameters: dict, output_report_html_path: Path, tmp_files: list[Path] = None
) -> None:  # type: ignore
    """
    Generate report based on jupyter notebook template.

    Parameters
    ----------
    template_notebook_path : Path
        Path to jupyter notebook template
    parameters : dict
        Parameters for report
    output_report_html_path : Path
        Path to output html report file
    tmp_files : list[Path]
        List of temporary files to be removed after report generation

    """
    if tmp_files is None:
        tmp_files = []

    output_report_ipynb = output_report_html_path.with_suffix(".ipynb")
    tmp_files.append(output_report_ipynb)

    # inject parameters and run notebook
    logger.info(f"Executing notebook (with papermill): {template_notebook_path}")

    def convert_path_to_str(value):
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, list):
            return [str(p) if isinstance(p, Path) else p for p in value]
        else:
            return value

    parameters = {k: convert_path_to_str(v) for k, v in parameters.items()}
    papermill.execute_notebook(
        input_path=str(template_notebook_path),
        output_path=str(output_report_ipynb),
        parameters=parameters,
        kernel_name="python3",
    )

    # convert to html
    logger.info(f"Converting notebook {output_report_ipynb} to HTML file {output_report_html_path}")
    notebook = nbformat.read(str(output_report_ipynb), as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True
    (body, resources) = html_exporter.from_notebook_node(notebook)

    with open(output_report_html_path, "w") as f:
        f.write(body)

    # edit html for readability
    logger.info(f"Modifying HTML style for {output_report_html_path}")
    modify_jupyter_notebook_html(output_report_html_path)

    # remove temporary files
    logger.debug(f"Removing tmp_files={[str(tmp_file) for tmp_file in tmp_files]}")
    for temp_file in tmp_files:
        if temp_file.is_file():
            temp_file.unlink()


def main():
    parser = ArgumentParser()
    parser.add_argument("--template_path", type=Path, help="Path to jupyter notebook template", required=True)
    parser.add_argument(
        "--params",
        type=json.loads,
        help='Parameters for the report, in a json format. E.g. \'{"a":"b", "c":"d"}\'',
        required=True,
    )
    parser.add_argument("--output_path", type=Path, help="Path to output html report file", required=True)
    parser.add_argument(
        "--tmp_files",
        type=str,
        help="Comma-separated list of temporary files to be removed after report generation",
        required=False,
    )
    args = parser.parse_args()

    # Split the tmp_files argument into a list of Path objects
    tmp_files = [Path(file) for file in args.tmp_files.split(",")] if args.tmp_files else []

    generate_report(args.template_path, args.params, args.output_path, tmp_files)


if __name__ == "__main__":
    main()
