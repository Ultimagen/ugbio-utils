"""
Methylation QC Report Generator

This module provides classes and functions to generate QC reports for methylation analysis
from MethylDackel output data stored in HDF5 format.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    input_h5_file: str
    input_base_file_name: str
    human_genome_key: str = "hg"
    control_genomes: list[str] = None

    def __post_init__(self):
        if self.control_genomes is None:
            self.control_genomes = ["Lambda", "pUC19"]


class DataProcessor:
    """Handles data loading and processing from HDF5 files."""

    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a table from the HDF5 file."""
        return pd.read_hdf(self.h5_file_path, key=table_name).reset_index()

    def get_available_tables(self) -> list[str]:
        """Get list of available tables in the HDF5 file."""
        with pd.HDFStore(self.h5_file_path, "r") as store:
            return store.keys()


class DataFormatter:
    """Handles data formatting and transformation."""

    @staticmethod
    def format_metric_names(df_in: pd.DataFrame) -> pd.DataFrame:
        """Format metric names for better display."""
        df_in = df_in.copy()
        replacements = {
            r"PercentMethylation": "Percent Methylation: ",
            r"PercentMethylationPosition": "Percent Methylation Position: ",
            r"CumulativeCoverage": "Cumulative Coverage",
            r"Coverage": "Coverage: ",
            r"TotalCpGs": "Total CpGs: ",
            r"_": " ",
        }

        for pattern, replacement in replacements.items():
            df_in["metric"] = df_in["metric"].str.replace(pattern, replacement, regex=True)

        return df_in

    @staticmethod
    def parse_metric_names(df_in: pd.DataFrame) -> pd.DataFrame:
        """Parse metric names to extract bins and base metric names."""
        df_in = df_in.copy()
        df_in["metric_orig"] = df_in["metric"]

        # Extract bin numbers
        df_in["bin"] = df_in["metric"].str.extract(r"\w+_(\d+)")

        # Extract base metric name
        df_in["metric"] = df_in["metric"].str.extract(r"(\w+)_\d+")

        return df_in

    @staticmethod
    def format_values_by_type(df_full: pd.DataFrame, stat_type_col: str = "stat_type") -> pd.DataFrame:
        """Format values based on their statistical type."""
        df_full = df_full.copy()

        if stat_type_col in df_full.columns:
            # Format percentages
            percent_mask = df_full[stat_type_col] == "Percent"
            if percent_mask.any():
                df_full.loc[percent_mask, "value"] = (df_full.loc[percent_mask, "value"] / 100).map("{:,.2%}".format)

            # Format coverage values
            coverage_mask = df_full[stat_type_col] == "Coverage"
            if coverage_mask.any():
                df_full.loc[coverage_mask, "value"] = df_full.loc[coverage_mask, "value"].map("{:,.2f}".format)

            # Format total counts
            total_mask = df_full[stat_type_col] == "Total"
            if total_mask.any():
                df_full.loc[total_mask, "value"] = df_full.loc[total_mask, "value"].map("{:,.0f}".format)

        return df_full


class DisplayHelper:
    """Handles display and visualization utilities."""

    @staticmethod
    def display_side_by_side(dfs: list[pd.DataFrame], captions: list[str], table_spacing: int = 2):
        """Display multiple DataFrames side by side with captions."""
        output = ""
        for caption, df in zip(captions, dfs, strict=False):
            output += df.style.set_table_attributes("style='display:inline-table'").set_caption(caption)._repr_html_()
            output += table_spacing * "\xa0"

        display(HTML(output))

    @staticmethod
    def wrap_df_text(df: pd.DataFrame):
        """Display DataFrame with wrapped text."""
        return display(HTML(df.to_html().replace("\\n", "<br>")))


class PlotGenerator:
    """Handles plot generation for the report."""

    def __init__(self):
        plt.style.use("ggplot")

    def plot_mbias(self, data_list: list[pd.DataFrame], colors: list[str] = None):
        """Generate M-bias plots."""
        if colors is None:
            colors = ["tomato", "indianred", "tomato", "indianred"]

        n_plots = len(data_list)

        if n_plots == 4:  # noqa: PLR2004 (Four plots)
            fig, axes = plt.subplots(2, 2, figsize=[12, 12])
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, min(n_plots, 2), figsize=[12, 5.5])
            if n_plots == 1:
                axes = [axes]

        for i, (data, ax) in enumerate(zip(data_list, axes, strict=False)):
            title = data["detail"].unique()[0] if "detail" in data.columns else f"Plot {i+1}"

            sns.lineplot(data=data, x="bin", y="value", lw=2.5, ax=ax, color=colors[i % len(colors)])
            ax.set_xlabel("Position", fontsize=14)
            ax.set_ylabel("Fraction of Methylation", fontsize=14)
            ax.set_title(title, fontsize=14)
            ax.tick_params(labelsize=14)
            ax.set_ylim([0, 1])
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()

    def plot_control_genome_methylation(self, data_list: list[pd.DataFrame]):
        """Plot methylation across control genomes."""
        fig, axes = plt.subplots(1, 2, figsize=[12, 5])
        colors = ["forestgreen", "steelblue"]

        for i, data in enumerate(data_list):
            data = data.reset_index()  # noqa: PLR2004,PLW2901 (Reset index for consistent display)
            genome_name = data["detail"].unique()[0]

            data.plot(
                kind="area",
                ylim=[0, 102],
                y="value",
                title=f"{genome_name}: Percent methylation",
                legend=False,
                color=colors[i],
                alpha=0.6,
                ax=axes[i],
                fontsize=14,
            )
            axes[i].set(xlabel="Position", ylabel="Percent Methylation")

    def plot_bar_distribution(self, data: pd.DataFrame, genome_name: str = "Human Genome"):
        """Generate bar plots for methylation and coverage distributions."""
        colors = ["salmon", "tomato"]
        n_rows = data.shape[0]

        if n_rows > 10:  # noqa: PLR2004 (More than 10 rows)
            # Multiple metrics
            grouped_data = [group for _, group in data.groupby("metric", sort=False)]
            fig, axes = plt.subplots(1, 2, figsize=[13.5, 5])

            for i, group_data in enumerate(grouped_data):
                if i >= len(axes):
                    break

                metric_name = group_data["metric"].unique()[0].replace(":", "")
                title = f"{genome_name}: {metric_name}"

                sns.barplot(data=group_data, x="bin", y="value", ax=axes[i], color=colors[i])
                axes[i].set_xlabel("Value Bins", fontsize=14)
                axes[i].set_ylabel(metric_name, fontsize=14)
                axes[i].set_title(title, fontsize=13)
                axes[i].tick_params(labelsize=14)
                plt.setp(axes[i].get_xticklabels(), rotation=45)
        else:
            # Single metric
            fig, ax = plt.subplots(1, 1, figsize=[5.5, 5])
            metric_name = data["metric"].unique()[0].replace(":", "")
            title = f"{genome_name}: {metric_name}"

            sns.barplot(data=data, x="bin", y="value", ax=ax, color=colors[0])
            ax.set_xlabel("Value Bins", fontsize=14)
            ax.set_ylabel(metric_name, fontsize=14)
            ax.set_title(title, fontsize=13)
            ax.tick_params(labelsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45)


class ReportSection:
    """Base class for report sections."""

    def __init__(
        self,
        data_processor: DataProcessor,
        formatter: DataFormatter,
        display_helper: DisplayHelper,
        plot_generator: PlotGenerator,
    ):
        self.data_processor = data_processor
        self.formatter = formatter
        self.display_helper = display_helper
        self.plot_generator = plot_generator


class GlobalStatsSection(ReportSection):
    """Handles global methylation statistics section."""

    def generate(self, genome_key: str = "hg"):
        """Generate global methylation statistics."""
        df_merge_context = self.data_processor.load_table("merge_context_desc")
        df_merge_context = df_merge_context.query("detail == @genome_key").drop(columns=["detail"])
        df_merge_context = self.formatter.format_metric_names(df_merge_context)

        # Format values based on metric type
        total_cpgs_mask = df_merge_context["metric"] == "Total CpGs: "
        df_merge_context.loc[total_cpgs_mask, "value"] = df_merge_context.loc[total_cpgs_mask, "value"].map(
            "{:,.0f}".format
        )
        df_merge_context.loc[~total_cpgs_mask, "value"] = df_merge_context.loc[~total_cpgs_mask, "value"].map(
            "{:,.2f}".format
        )

        return df_merge_context.set_index("metric")


class PerReadStatsSection(ReportSection):
    """Handles per-read statistics section."""

    def generate(self):
        """Generate per-read statistics."""
        df_per_read = self.data_processor.load_table("per_read_desc")
        df_per_read = df_per_read.drop(columns=["detail"])
        df_per_read = self.formatter.format_metric_names(df_per_read)
        df_per_read["value"] = df_per_read["value"].map("{:,.2f}".format)

        return df_per_read.set_index("metric")


class NonCpGStatsSection(ReportSection):
    """Handles non-CpG cytosine statistics section."""

    def generate(self):
        """Generate non-CpG cytosine statistics."""
        df_non_cpg = self.data_processor.load_table("merge_context_non_cpg_desc")
        df_non_cpg = self.formatter.format_metric_names(df_non_cpg)

        # Extract statistic type and format values
        df_non_cpg["stat_type"] = df_non_cpg["metric"].str.extract(r"([A-Za-z]+)[\s:]")
        df_non_cpg["metric"] = df_non_cpg["metric"].str.title()
        df_non_cpg = self.formatter.format_values_by_type(df_non_cpg)

        # Group by detail and prepare for side-by-side display
        grouped_data = [group.set_index("metric")["value"].to_frame() for _, group in df_non_cpg.groupby("detail")]
        table_names = df_non_cpg["detail"].unique()

        return grouped_data, table_names


class MBiasSection(ReportSection):
    """Handles M-bias analysis section."""

    def generate_plots(self, table_name: str):
        """Generate M-bias plots."""
        df_mbias = self.data_processor.load_table(table_name)
        df_mbias = self.formatter.parse_metric_names(df_mbias)
        df_mbias["bin"] = df_mbias["bin"].astype(int)

        # Prepare data for plotting
        df_mbias["stat_type"] = df_mbias["metric"].str.extract(r"([A-Za-z]+)\s")
        df_mbias["metric"] = df_mbias["metric"].str.title()

        grouped_data = [group for _, group in df_mbias.groupby("detail", sort=False)]
        self.plot_generator.plot_mbias(grouped_data)

        return df_mbias

    def generate_stats_table(self, table_name: str):
        """Generate M-bias descriptive statistics table."""
        df_mbias = self.data_processor.load_table(table_name)
        df_mbias = self.formatter.format_metric_names(df_mbias)

        df_mbias["stat_type"] = df_mbias["metric"].str.extract(r"([A-Za-z]+)\s")
        df_mbias["metric"] = df_mbias["metric"].str.title()

        # Format percentage values
        percent_mask = df_mbias["stat_type"] == "Percent"
        df_mbias.loc[percent_mask, "value"] = df_mbias.loc[percent_mask, "value"].map("{:,.2%}".format)

        # Group and prepare for display
        grouped_data = [group for _, group in df_mbias.groupby("detail")]

        # Reorder if needed (specific to the original logic)
        if len(grouped_data) == 4:  # noqa: PLR2004 (OT, OB, COOT, COOB)
            order = [3, 2, 1, 0]
        else:
            order = [1, 0]  # noqa: PLR2004 (OT, OB)
        grouped_data = [grouped_data[i] for i in order]

        # Prepare for side-by-side display
        display_data = []
        table_names = []
        for group in grouped_data:
            table_names.append(group["detail"].iloc[0])
            display_data.append(group.set_index("metric")["value"].to_frame())

        return display_data, table_names


class ControlGenomeSection(ReportSection):
    """Handles control genome analysis section."""

    def check_control_genomes_exist(self, control_genomes: list[str]) -> bool:
        """Check if control genomes exist in the data."""
        df_per_position = self.data_processor.load_table("merge_context_per_position")
        available_genomes = list(set(df_per_position["detail"]))
        return all(genome in available_genomes for genome in control_genomes)

    def generate_methylation_plots(self, control_genomes: list[str]):
        """Generate control genome methylation plots."""
        df_per_position = self.data_processor.load_table("merge_context_per_position")
        df_per_position = self.formatter.parse_metric_names(df_per_position)
        df_per_position["bin"] = df_per_position["bin"].astype(int)
        df_per_position = self.formatter.format_metric_names(df_per_position)

        # Filter for methylation position data
        df_per_position = df_per_position.query('metric == "Percent Methylation: Position"')
        grouped_data = [group for _, group in df_per_position.groupby("detail")]

        self.plot_generator.plot_control_genome_methylation(grouped_data)

    def generate_stats_table(self, control_genomes: list[str], human_genome_key: str = "hg"):
        """Generate control genome statistics table."""
        df_merge_context_desc = self.data_processor.load_table("merge_context_desc")
        df_merge_context_desc = df_merge_context_desc[df_merge_context_desc["detail"] != human_genome_key].reset_index()
        df_merge_context_desc = self.formatter.format_metric_names(df_merge_context_desc)

        df_merge_context_desc["stat_type"] = df_merge_context_desc["metric"].str.extract(r"([A-Za-z]+)[\s:]")
        df_merge_context_desc["metric"] = (
            df_merge_context_desc["metric"].str.title().str.replace(r"Cpgs", "CpGs", regex=True)
        )
        df_merge_context_desc = self.formatter.format_values_by_type(df_merge_context_desc)

        # Prepare for side-by-side display
        display_data = []
        for _, group in df_merge_context_desc.groupby("detail"):
            display_data.append(group.set_index("metric")["value"].to_frame())

        return display_data, control_genomes


class HistogramSection(ReportSection):
    """Handles histogram analysis section."""

    def generate_plots_and_tables(self, genome_key: str = "hg"):
        """Generate histogram plots and tables."""
        df_merge_context = self.data_processor.load_table("merge_context_hist")
        df_merge_context = df_merge_context[df_merge_context["detail"] == genome_key]
        df_merge_context = self.formatter.parse_metric_names(df_merge_context)
        df_merge_context = self.formatter.format_metric_names(df_merge_context)
        df_merge_context["stat_type"] = df_merge_context["metric"].str.extract(r"([A-Za-z]+):")

        # Generate plots
        grouped_for_plots = [group for _, group in df_merge_context.groupby("stat_type", sort=False)]
        for group_data in grouped_for_plots:
            self.plot_generator.plot_bar_distribution(group_data)

        # Generate tables
        grouped_for_tables = [group for _, group in df_merge_context.groupby("metric", sort=False)]
        display_data = []
        table_names = []

        for i, group in enumerate(grouped_for_tables):
            group = group.reset_index()  # noqa: PLR2004,PLW2901 (Reset index for consistent display)
            if i < 2:  # noqa: PLR2004 (First two are counts)
                group["value"] = group["value"].map("{:,.0f}".format)
            else:  # Rest are percentages
                group["value"] = group["value"].map("{:,.2%}".format)

            table_names.append(group["metric"].iloc[0].replace(":", ""))
            display_data.append(group.set_index("bin")["value"].to_frame())

        return display_data, table_names


class MethylationQCReportGenerator:
    """Main class for generating methylation QC reports."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.data_processor = DataProcessor(config.input_h5_file)
        self.formatter = DataFormatter()
        self.display_helper = DisplayHelper()
        self.plot_generator = PlotGenerator()

        # Initialize sections
        self._init_sections()

    def _init_sections(self):
        """Initialize all report sections."""
        args = (self.data_processor, self.formatter, self.display_helper, self.plot_generator)

        self.global_stats = GlobalStatsSection(*args)
        self.per_read_stats = PerReadStatsSection(*args)
        self.non_cpg_stats = NonCpGStatsSection(*args)
        self.mbias_section = MBiasSection(*args)
        self.control_genome_section = ControlGenomeSection(*args)
        self.histogram_section = HistogramSection(*args)

    def display_sample_info(self):
        """Display sample information."""
        sample_info = pd.DataFrame(
            data={"value": [self.config.input_base_file_name, str(self.config.input_h5_file)]},
            index=["Sample name", "h5 file"],
        )
        sample_info["value"] = sample_info["value"].str.wrap(100)

        self.display_helper.wrap_df_text(sample_info.style.set_properties(**{"text-align": "left"}))

    def generate_global_stats(self):
        """Generate and display global methylation statistics."""
        stats_df = self.global_stats.generate(self.config.human_genome_key)
        display(stats_df)

    def generate_per_read_stats(self):
        """Generate and display per-read statistics."""
        stats_df = self.per_read_stats.generate()
        display(stats_df)

    def generate_non_cpg_stats(self):
        """Generate and display non-CpG statistics."""
        display_data, table_names = self.non_cpg_stats.generate()
        self.display_helper.display_side_by_side(display_data, table_names)

    def generate_mbias_analysis(self):
        """Generate M-bias plots and statistics."""
        # CpG M-bias
        self.mbias_section.generate_plots("mbias_per_position")
        display_data, table_names = self.mbias_section.generate_stats_table("mbias_desc")
        self.display_helper.display_side_by_side(display_data, table_names)

    def generate_mbias_non_cpg_analysis(self):
        """Generate non-CpG M-bias plots and statistics."""
        self.mbias_section.generate_plots("mbias_non_cpg_per_position")
        display_data, table_names = self.mbias_section.generate_stats_table("mbias_non_cpg_desc")
        self.display_helper.display_side_by_side(display_data, table_names)

    def generate_control_genome_analysis(self):
        """Generate control genome analysis if data exists."""
        if self.control_genome_section.check_control_genomes_exist(self.config.control_genomes):
            self.control_genome_section.generate_methylation_plots(self.config.control_genomes)

            display(HTML("<h2>Control Genomes: Methylation and Coverage Descriptive Statistics</h2>"))
            display_data, table_names = self.control_genome_section.generate_stats_table(
                self.config.control_genomes, self.config.human_genome_key
            )
            self.display_helper.display_side_by_side(display_data, table_names)

    def generate_histogram_analysis(self):
        """Generate histogram analysis."""
        display_data, table_names = self.histogram_section.generate_plots_and_tables(self.config.human_genome_key)
        self.display_helper.display_side_by_side(display_data, table_names)
