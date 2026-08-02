import subprocess


def check_r_environment(packages=("cn.mops",)):
    """Check that R and the given R packages are available.

    Parameters
    ----------
    packages : Iterable[str]
        R package names that must load for the tests to run.

    Returns
    -------
    bool
        True if Rscript is available and every package loads, False otherwise.
    """
    load_pkgs = "; ".join(f"suppressPackageStartupMessages(library({pkg}))" for pkg in packages)
    try:
        # Use --vanilla so the probe sees the same library set as the test scripts, which are
        # all run with "Rscript --vanilla" (--vanilla implies --no-environ, i.e. .Renviron /
        # R_LIBS_USER are ignored). A plain "Rscript -e" could find packages the tests cannot.
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", load_pkgs],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
