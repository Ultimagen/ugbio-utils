"""
Analyze cost distribution per workflow for a specific version or version prefix and account.

This script queries the Papyrus database for runs matching a specific version
(exact match or prefix match) and workspace, groups them by workflow name, and
generates histograms of cost distribution for each workflow. Optionally creates
a tar.gz archive containing all outputs.

The script automatically saves raw data to a CSV cache file, which can be reused
in subsequent runs with the --use-cache flag to avoid redundant database queries.

Individual workflow plots show side-by-side comparison of original data and outlier-filtered
data using the IQR (Interquartile Range) method for better visualization.

Usage:
    python cost_distribution_by_version.py <version> [--account 471112545487] [--days 180]
                                                      [--create-tarball] [--use-cache]

Example:
    python cost_distribution_by_version.py v1.25.0          # Exact version, query database
    python cost_distribution_by_version.py v1.25            # Version prefix (matches v1.25.0, v1.25.1, etc.)
    python cost_distribution_by_version.py v1.25 --account 471112545487 --days 90
    python cost_distribution_by_version.py v1.25 --create-tarball  # Also create tar.gz archive
    python cost_distribution_by_version.py v1.25 --use-cache  # Use cached data from previous run
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
) -> None:
    """Create histograms of cost distribution per workflow with side-by-side comparison.

    Creates both original and outlier-filtered views for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: workflowName, cost
    output_dir : Path, optional
        Directory to save plots (default: current directory)
    version : str, optional
        Version string for plot titles (default: "unknown")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = df["workflowName"].unique()
    logger.info(f"\nCreating side-by-side comparison histograms for {len(workflows)} workflows:")

    # Create a figure with side-by-side subplots for all workflows (original vs filtered)
    fig = plt.figure(figsize=(20, 4 * len(workflows)))

    for idx, workflow in enumerate(sorted(workflows), 1):
        workflow_data = df[df["workflowName"] == workflow]
        costs_all = workflow_data["cost"].to_numpy()
        costs_filtered, _, outlier_count = filter_outliers_iqr(costs_all)

        # Left subplot: Original data
        ax1 = fig.add_subplot(len(workflows), 2, 2 * idx - 1)
        ax1.hist(costs_all, bins=30, edgecolor="black", alpha=0.7, color="skyblue")

        mean_all = costs_all.mean()
        median_all = np.median(costs_all)
        ax1.axvline(mean_all, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_all:.2f}")
        ax1.axvline(median_all, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_all:.2f}")

        ax1.set_xlabel("Cost (USD)")
        ax1.set_ylabel("Frequency")
        title_str = (
            f"{workflow} - Original (v{version})\n"
            f"n={len(costs_all)}, Min=${costs_all.min():.2f}, Max=${costs_all.max():.2f}"
        )
        ax1.set_title(title_str, fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(visible=True, alpha=0.3)

        # Right subplot: Filtered data
        ax2 = fig.add_subplot(len(workflows), 2, 2 * idx)
        ax2.hist(costs_filtered, bins=30, edgecolor="black", alpha=0.7, color="lightcoral")

        mean_filtered = costs_filtered.mean()
        median_filtered = np.median(costs_filtered)
        ax2.axvline(mean_filtered, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_filtered:.2f}")
        ax2.axvline(
            median_filtered, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_filtered:.2f}"
        )

        ax2.set_xlabel("Cost (USD)")
        ax2.set_ylabel("Frequency")
        title_str = (
            f"{workflow} - Filtered ({outlier_count} outliers removed)\n"
            f"n={len(costs_filtered)}, Min=${costs_filtered.min():.2f}, Max=${costs_filtered.max():.2f}"
        )
        ax2.set_title(title_str, fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(visible=True, alpha=0.3)

        # Log with both original and filtered stats for comparison
        logger.info(
            f"  {workflow}: {len(costs_all)} runs (original), mean=${mean_all:.2f}, median=${median_all:.2f} | "
            f"{len(costs_filtered)} runs (filtered), mean=${mean_filtered:.2f}, median=${median_filtered:.2f}"
        )

    plt.tight_layout()

    # Save the combined figure
    output_file = output_dir / f"cost_distribution_v{version}_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"\nCombined histogram saved: {output_file}")

    plt.close(fig)

    # Create individual side-by-side comparison plots for each workflow
    for workflow in sorted(workflows):
        workflow_data = df[df["workflowName"] == workflow]
        costs_all = workflow_data["cost"].to_numpy()
        costs_filtered, _, outlier_count = filter_outliers_iqr(costs_all)

        # Create side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Left plot: Original data
        ax1.hist(costs_all, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
        mean_all = costs_all.mean()
        median_all = np.median(costs_all)
        ax1.axvline(mean_all, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_all:.2f}")
        ax1.axvline(median_all, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_all:.2f}")
        ax1.set_xlabel("Cost (USD)")
        ax1.set_ylabel("Frequency")
        ax1.set_title(
            f"Original Data\nn={len(costs_all)}, Min=${costs_all.min():.2f}, Max=${costs_all.max():.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(visible=True, alpha=0.3)

        # Right plot: Filtered data
        ax2.hist(costs_filtered, bins=30, edgecolor="black", alpha=0.7, color="lightcoral")
        mean_filtered = costs_filtered.mean()
        median_filtered = np.median(costs_filtered)
        ax2.axvline(mean_filtered, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_filtered:.2f}")
        ax2.axvline(
            median_filtered, color="green", linestyle="--", linewidth=2, label=f"Median: ${median_filtered:.2f}"
        )
        ax2.set_xlabel("Cost (USD)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(
            f"Filtered Data ({outlier_count} outliers removed)\n"
            f"n={len(costs_filtered)}, Min=${costs_filtered.min():.2f}, Max=${costs_filtered.max():.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.legend()
        ax2.grid(visible=True, alpha=0.3)

        # Overall title
        fig.suptitle(f"Cost Distribution - {workflow} (v{version})", fontsize=14, fontweight="bold", y=1.02)

        safe_workflow_name = workflow.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"cost_distribution_v{version}_{safe_workflow_name}_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison histogram saved: {output_file}")
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

    # Copy all PNG and CSV files to staging directory (glob doesn't traverse subdirs, so no need to filter)
    for file_pattern in ["*.png", "*.csv"]:
        for file_path in output_dir.glob(file_pattern):
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

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize version string for file names (remove leading 'v' if present)
    version_normalized = args.version.lstrip("v")

    # Define cache file path
    cache_file = args.output_dir / f"raw_data_v{version_normalized}_account{args.account}_days{args.days}.csv"

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

    # Create visualizations and summary with side-by-side comparisons
    create_histograms(df, args.output_dir, version_normalized)
    create_summary_csv(df, args.output_dir, version_normalized)

    # Create tarball if requested
    if args.create_tarball:
        create_tarball(args.output_dir, version_normalized)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
