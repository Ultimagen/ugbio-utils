"""
Analyze cost distribution per workflow for a specific version or version prefix and account.

This script queries the Papyrus database for runs matching a specific version
(exact match or prefix match) and workspace, groups them by workflow name, and
generates histograms of cost distribution for each workflow. Optionally creates
a tar.gz archive containing all outputs.

The script automatically saves raw data to a CSV cache file, which can be reused
in subsequent runs with the --use-cache flag to avoid redundant database queries.

Outlier filtering using the IQR (Interquartile Range) method can be applied to
improve histogram visualization by removing extreme values that skew the distribution.

Usage:
    python cost_distribution_by_version.py <version> [--account 471112545487] [--days 180]
                                                      [--create-tarball] [--use-cache] [--filter-outliers]

Example:
    python cost_distribution_by_version.py v1.25.0          # Exact version, query database
    python cost_distribution_by_version.py v1.25            # Version prefix (matches v1.25.0, v1.25.1, etc.)
    python cost_distribution_by_version.py v1.25 --account 471112545487 --days 90
    python cost_distribution_by_version.py v1.25 --create-tarball  # Also create tar.gz archive
    python cost_distribution_by_version.py v1.25 --use-cache  # Use cached data from previous run
    python cost_distribution_by_version.py v1.25 --use-cache --filter-outliers  # Use cache and filter outliers
"""

import argparse
import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ugbio_omics import db_access

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_date_cutoff(days: int) -> datetime:
    """Get the cutoff date for recent runs.

    Parameters
    ----------
    days : int
        Number of days to look back

    Returns
    -------
    datetime
        Cutoff datetime (now minus `days`)
    """
    return datetime.utcnow() - timedelta(days=days)


def load_cached_data(cache_file: Path) -> pd.DataFrame | None:
    """Load cached data from a CSV file if it exists.

    Parameters
    ----------
    cache_file : Path
        Path to the cache file

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with cached data if file exists, None otherwise
    """
    if cache_file.exists():
        logger.info(f"Loading cached data from: {cache_file}")
        df = pd.read_csv(cache_file)
        logger.info(f"Loaded {len(df)} records from cache")
        return df
    return None


def query_cost_data(
    version: str, account_id: str, days: int = 180, workflows: list | None = None, output_dir: Path = Path(".")
) -> pd.DataFrame:
    """Query database for runs matching version (or version prefix) and account, within time window.

    Parameters
    ----------
    version : str
        Version string or prefix (e.g., "v1.25.0" for exact match or "v1.25" for prefix match)
    account_id : str
        AWS account ID (e.g., "471112545487")
    days : int, optional
        Number of days to look back (default 180)
    workflows : list, optional
        List of workflow names to filter by (default None = all workflows)
    output_dir : Path, optional
        Directory to save cached data (default: current directory)

    Returns
    -------
    pd.DataFrame
        DataFrame with queried run data
    """
    workspace = f"aws-{account_id}"
    cutoff_date = get_date_cutoff(days)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define cache file path
    cache_file = output_dir / f"raw_data_v{version}_account{account_id}_days{days}.csv"

    logger.info("Querying database for:")
    logger.info(f"  Version: {version}")
    logger.info(f"  Workspace: {workspace}")
    logger.info(f"  Created after: {cutoff_date}")
    logger.info("  Status: COMPLETED")

    # Build query with regex for prefix matching
    # If version looks like a prefix (e.g., "v1.25"), use regex to match any version starting with it
    # Otherwise use exact match
    version_part_threshold = 2  # Number of dots for full version (v1.25.0 has 2 dots)
    if version.count(".") < version_part_threshold:  # Likely a prefix like "v1.25"
        query = {
            "inputs.pipeline_version": {"$regex": f"^{version}"},
            "metadata.workspace": workspace,
            "metadata.createdAt": {"$gte": cutoff_date},
            "metadata.status": "COMPLETED",
        }
        logger.info(f"  Using prefix match (regex: ^{version})")
    else:  # Exact match like "v1.25.0"
        query = {
            "inputs.pipeline_version": version,
            "metadata.workspace": workspace,
            "metadata.createdAt": {"$gte": cutoff_date},
            "metadata.status": "COMPLETED",
        }
        logger.info("  Using exact match")

    docs = db_access.query_database(query, collection="pipelines")
    logger.info(f"Found {len(docs)} matching records")

    if not docs:
        logger.warning("No records found matching the query criteria")
        return pd.DataFrame()

    # Extract relevant fields from documents
    data = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        inputs = doc.get("inputs", {})
        data.append(
            {
                "workflowId": metadata.get("workflowId"),
                "workflowName": metadata.get("workflowName"),
                "cost": metadata.get("cost"),
                "createdAt": metadata.get("createdAt"),
                "status": metadata.get("status"),
                "runId": metadata.get("runId"),
                "pipeline_version": inputs.get("pipeline_version"),
            }
        )

    df = pd.DataFrame(data)

    # Filter out records with no cost data
    df = df.dropna(subset=["cost"])
    logger.info(f"Records with cost data: {len(df)}")

    # Filter by workflows if specified
    if workflows:
        df = df[df["workflowName"].isin(workflows)]
        logger.info(f"Records after workflow filtering: {len(df)}")

    # Save raw data to cache file
    df.to_csv(cache_file, index=False)
    logger.info(f"Raw data saved to: {cache_file}")

    return df


def filter_outliers_iqr(costs: np.ndarray, iqr_multiplier: float = 1.5) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter outliers using the IQR method.

    Parameters
    ----------
    costs : np.ndarray
        Array of cost values
    iqr_multiplier : float, optional
        Multiplier for IQR to determine outlier threshold (default: 1.5)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Filtered costs, outlier costs, and count of outliers
    """
    q1 = np.percentile(costs, 25)
    q3 = np.percentile(costs, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    mask = (costs >= lower_bound) & (costs <= upper_bound)
    filtered_costs = costs[mask]
    outliers = costs[~mask]

    return filtered_costs, outliers, len(outliers)


def create_histograms(  # noqa: PLR0915
    df: pd.DataFrame,
    output_dir: Path = Path("."),
    version: str = "unknown",
    *,
    filter_outliers: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Create histograms of cost distribution per workflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: workflowName, cost
    output_dir : Path, optional
        Directory to save plots (default: current directory)
    version : str, optional
        Version string for plot titles (default: "unknown")
    filter_outliers : bool, optional
        Whether to filter outliers for better visualization (default: False)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = df["workflowName"].unique()
    logger.info(f"\nCreating histograms for {len(workflows)} workflows:")
    if filter_outliers:
        logger.info("  (Filtering outliers using IQR method)")

    # Create a figure with subplots for all workflows
    fig = plt.figure(figsize=(16, 4 * len(workflows)))

    for idx, workflow in enumerate(sorted(workflows), 1):
        workflow_data = df[df["workflowName"] == workflow]
        costs_all = workflow_data["cost"].to_numpy()

        # Filter outliers if requested
        if filter_outliers:
            costs, outliers_arr, outlier_count = filter_outliers_iqr(costs_all)
            outlier_info = f", {outlier_count} outliers filtered"
        else:
            costs = costs_all
            outlier_count = 0
            outlier_info = ""

        ax = fig.add_subplot(len(workflows), 1, idx)

        # Create histogram
        ax.hist(costs, bins=30, edgecolor="black", alpha=0.7, color="skyblue")

        # Add statistics
        mean_cost = costs.mean()
        median_cost = np.median(costs)
        max_cost = costs.max()
        min_cost = costs.min()
        count = len(costs)

        ax.axvline(mean_cost, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_cost:.2f}")
        ax.axvline(median_cost, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_cost:.2f}")

        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Frequency")
        title_str = f"{workflow} (v{version})\nn={count}{outlier_info}, Min=${min_cost:.2f}, Max=${max_cost:.2f}"
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

        logger.info(f"  {workflow}: {count} runs, mean=${mean_cost:.2f}, median=${median_cost:.2f}{outlier_info}")

    plt.tight_layout()

    # Save the combined figure
    output_file = output_dir / f"cost_distribution_v{version}_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"\nCombined histogram saved: {output_file}")

    plt.close(fig)

    # Create individual plots for each workflow
    for workflow in sorted(workflows):
        workflow_data = df[df["workflowName"] == workflow]
        costs_all = workflow_data["cost"].to_numpy()

        # Filter outliers if requested
        if filter_outliers:
            costs, _, outlier_count = filter_outliers_iqr(costs_all)
            outlier_info = f", {outlier_count} outliers filtered"
        else:
            costs = costs_all
            outlier_count = 0
            outlier_info = ""

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(costs, bins=30, edgecolor="black", alpha=0.7, color="skyblue")

        mean_cost = costs.mean()
        median_cost = np.median(costs)
        max_cost = costs.max()
        min_cost = costs.min()
        count = len(costs)

        ax.axvline(mean_cost, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_cost:.2f}")
        ax.axvline(median_cost, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_cost:.2f}")

        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("Frequency")
        title_str = (
            f"Cost Distribution - {workflow} (v{version})\n"
            f"n={count}{outlier_info}, Min=${min_cost:.2f}, Max=${max_cost:.2f}"
        )
        ax.set_title(title_str, fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

        # Save individual plot
        safe_workflow_name = workflow.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"cost_distribution_v{version}_{safe_workflow_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Individual histogram saved: {output_file}")

        plt.close(fig)


def create_summary_csv(df: pd.DataFrame, output_dir: Path = Path("."), version: str = "unknown") -> None:
    """Create a CSV summary of cost statistics per workflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: workflowName, cost
    output_dir : Path, optional
        Directory to save CSV (default: current directory)
    version : str, optional
        Version string (default: "unknown")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for workflow in sorted(df["workflowName"].unique()):
        workflow_data = df[df["workflowName"] == workflow]
        costs = workflow_data["cost"].to_numpy()

        summary_data.append(
            {
                "workflowName": workflow,
                "count": len(costs),
                "mean_cost": costs.mean(),
                "median_cost": np.median(costs),
                "std_cost": costs.std(),
                "min_cost": costs.min(),
                "max_cost": costs.max(),
                "total_cost": costs.sum(),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    output_file = output_dir / f"cost_summary_v{version}.csv"
    summary_df.to_csv(output_file, index=False)
    logger.info(f"\nSummary CSV saved: {output_file}")
    logger.info("\nCost Summary:")
    logger.info(f"\n{summary_df.to_string()}")


def create_tarball(output_dir: Path, version: str) -> None:
    """Create a tar.gz file containing all outputs (histograms and CSV).

    Parameters
    ----------
    output_dir : Path
        Output directory containing all generated files
    version : str
        Version string for the tarball name
    """
    output_dir = Path(output_dir)

    # Create a temporary directory for staging files
    staging_dir = output_dir / f"cost_analysis_v{version}"
    staging_dir.mkdir(exist_ok=True)

    # Copy all PNG and CSV files to staging directory
    for file_pattern in ["*.png", "*.csv"]:
        for file_path in output_dir.glob(file_pattern):
            if not file_path.parent.name.endswith(f"v{version}"):  # Skip files in staging dir
                shutil.copy2(file_path, staging_dir / file_path.name)

    # Create tar.gz
    tarball_path = output_dir / f"cost_analysis_v{version}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=f"cost_analysis_v{version}")

    logger.info(f"\nTarball created: {tarball_path}")

    # Clean up staging directory
    shutil.rmtree(staging_dir)
    logger.info("Staging directory cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cost distribution per workflow for a specific version or version prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("version", help="Version string or prefix (e.g., v1.25.0 for exact match or v1.25 for prefix)")
    parser.add_argument("--account", default="471112545487", help="AWS account ID (default: 471112545487)")
    parser.add_argument("--days", type=int, default=180, help="Number of days to look back (default: 180)")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for plots and CSV")
    parser.add_argument("--create-tarball", action="store_true", help="Create a tar.gz file with all outputs")
    parser.add_argument(
        "--use-cache", action="store_true", help="Use cached data if available instead of querying database"
    )
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Filter outliers using IQR method for better histogram visualization",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define cache file path
    cache_file = args.output_dir / f"raw_data_v{args.version}_account{args.account}_days{args.days}.csv"

    # Try to load from cache if requested
    df = None
    if args.use_cache:
        df = load_cached_data(cache_file)

    # Query database if no cached data available
    if df is None:
        df = query_cost_data(args.version, args.account, args.days, output_dir=args.output_dir)

    if df.empty:
        logger.warning("No data to analyze")
        return

    # Create visualizations and summary
    # If filter_outliers is enabled, create both filtered and non-filtered outputs in separate directories
    if args.filter_outliers:
        # Create non-filtered outputs in 'original' subdirectory
        original_dir = args.output_dir / "original"
        logger.info("\n" + "=" * 80)
        logger.info("Creating outputs WITHOUT outlier filtering:")
        logger.info("=" * 80)
        create_histograms(df, original_dir, args.version, filter_outliers=False)
        create_summary_csv(df, original_dir, args.version)

        # Create filtered outputs in 'filtered' subdirectory
        filtered_dir = args.output_dir / "filtered"
        logger.info("\n" + "=" * 80)
        logger.info("Creating outputs WITH outlier filtering:")
        logger.info("=" * 80)
        create_histograms(df, filtered_dir, args.version, filter_outliers=True)
        create_summary_csv(df, filtered_dir, args.version)

        # Create tarballs if requested
        if args.create_tarball:
            create_tarball(original_dir, args.version)
            create_tarball(filtered_dir, args.version + "_filtered")
    else:
        # Only create non-filtered outputs in the main output directory
        create_histograms(df, args.output_dir, args.version, filter_outliers=False)
        create_summary_csv(df, args.output_dir, args.version)

        # Create tarball if requested
        if args.create_tarball:
            create_tarball(args.output_dir, args.version)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
