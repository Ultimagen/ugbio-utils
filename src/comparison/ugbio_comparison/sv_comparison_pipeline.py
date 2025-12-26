# SV comparison pipeline
import logging
import os
import shutil
import sys
import tempfile
from os.path import basename, dirname
from os.path import join as pjoin

import pandas as pd
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools


class SVComparison:
    """
    SV comparison pipeline
    """

    def __init__(self, simple_pipeline: SimplePipeline | None = None, logger: logging.Logger | None = None):
        """Combines VCF in parts from GATK and indices the result

        Parameters
        ----------
        simple_pipeline : SimplePipeline, optional
            Optional SimplePipeline object for executing shell commands
        """
        self.sp = simple_pipeline
        self.vu = VcfUtils(self.sp, logger=logger)
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def __execute(self, command: str, output_file: str | None = None):
        """Summary

        Parameters
        ----------
        command : str
            Description
        output_file : str, optional
            Description
        """
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp, module_name=__name__)

    def run_truvari(  # noqa: PLR0913
        self,
        calls: str,
        gt: str,
        outdir: str,
        bed: str | None = None,
        pctseq: float = 0.0,
        pctsize: float = 0.0,
        maxsize: int = 50000,
        *,
        erase_outdir: bool = True,
        ignore_filter: bool = False,
        ignore_type: bool = True,
        skip_collapse: bool = False,
    ):
        """
        Run truvari, generate truvari report and concordance VCF
        Parameters
        ----------
        calls : str
            Calls file
        gt : str
            Ground truth file
        outdir : str
            Output directory
        bed : str, optional
            Bed file, by default None
        pctseq : float, optional
            Percentage of sequence identity, by default 0.0
        pctsize : float, optional
            Percentage of size identity, by default 0.0
        maxsize : int, optional
            Maximum size for SV comparison, by default 50000. For CNV comparison, consider increasing this value.
            (-1 - unlimited)
        erase_outdir : bool, optional
            Erase output directory if it exists, by default True
        ignore_filter : bool, optional
            If True, ignore FILTER field (remove --passonly flag), by default False
        ignore_type : bool, optional
            If True, ignore SVTYPE when matching to truth, by default True
        skip_collapse : bool, optional
            If True, VCF collapsing step was skipped (usually for defining truthset) and
            truvari should run with -p multi
        Returns
        -------
        None
        """
        if erase_outdir and os.path.exists(outdir):
            shutil.rmtree(outdir)

        truvari_cmd = ["truvari", "bench", "-b", gt, "-c", calls, "-o", outdir]
        if ignore_type:
            truvari_cmd.append("-t")
        if not ignore_filter:
            truvari_cmd.insert(len(truvari_cmd), "--passonly")

        if bed:
            truvari_cmd.extend(["--includebed", bed])
        if skip_collapse:
            truvari_cmd.extend(["--pick", "multi"])
        truvari_cmd.extend(["--pctseq", str(pctseq)])
        truvari_cmd.extend(["--pctsize", str(pctsize)])
        truvari_cmd.extend(["--sizemax", str(maxsize)])

        self.logger.info(f"truvari command: {' '.join(truvari_cmd)}")
        self.__execute(" ".join(truvari_cmd))

    def truvari_to_dataframes(
        self,
        truvari_dir: str,
        custom_info_fields: tuple = (),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert truvari report to dataframes

        Parameters
        ----------
        truvari_dir : str
            Truvari directory
        custom_info_fields : tuple, optional
            Custom info fields to read from the VCFs, in addition SVTYPE and SVLEN

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple of truvari base and calls concordance dataframes
        """

        def assign_svlen_int(df):
            return (
                df["svlen"]
                .apply(lambda x: x[0] if isinstance(x, tuple) else (x if x is not None else 0))
                .fillna(0)
                .astype(int)
            )

        info_fields_to_read: list[str] = list(("SVTYPE", "SVLEN", "MatchId") + custom_info_fields)
        df_tp_base = vcftools.get_vcf_df(pjoin(truvari_dir, "tp-base.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_tp_base["svlen_int"] = assign_svlen_int(df_tp_base)
        df_tp_base["label"] = "TP"
        df_fn = vcftools.get_vcf_df(pjoin(truvari_dir, "fn.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_fn["svlen_int"] = assign_svlen_int(df_fn)
        df_fn["label"] = "FN"
        df_base = pd.concat((df_tp_base, df_fn))

        df_tp_calls = vcftools.get_vcf_df(pjoin(truvari_dir, "tp-comp.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_tp_calls["label"] = "TP"
        df_tp_calls["svlen_int"] = assign_svlen_int(df_tp_calls)

        df_fp = vcftools.get_vcf_df(pjoin(truvari_dir, "fp.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_fp["label"] = "FP"
        df_fp["svlen_int"] = assign_svlen_int(df_fp)

        df_calls = pd.concat((df_tp_calls, df_fp))

        # Add label_type field
        # For non-TP variants, label_type equals label
        df_fn["label_type"] = df_fn["label"]
        df_fp["label_type"] = df_fp["label"]

        # For TP variants, label_type is the SVTYPE from the matching variant in the paired VCF
        # Create mapping from MatchId to SVTYPE for each TP dataframe
        if not df_tp_base.empty and "matchid" in df_tp_base.columns:
            matchid_to_svtype_calls = df_tp_calls.set_index("matchid")["svtype"].to_dict()
            df_tp_base["label_type"] = df_tp_base["matchid"].map(matchid_to_svtype_calls)
        else:
            df_tp_base["label_type"] = df_tp_base["label"]

        if not df_tp_calls.empty and "matchid" in df_tp_calls.columns:
            matchid_to_svtype_base = df_tp_base.set_index("matchid")["svtype"].to_dict()
            df_tp_calls["label_type"] = df_tp_calls["matchid"].map(matchid_to_svtype_base)
        else:
            df_tp_calls["label_type"] = df_tp_calls["label"]

        # Recreate concatenated dataframes with label_type field
        df_base = pd.concat((df_tp_base, df_fn))
        df_calls = pd.concat((df_tp_calls, df_fp))

        return df_base, df_calls

    def run_pipeline(  # noqa: PLR0913
        self,
        calls: str,
        gt: str,
        output_file_name: str,
        outdir: str,
        hcr_bed: str | None = None,
        pctseq: float = 0.0,
        pctsize: float = 0.0,
        maxsize: int = 50000,
        custom_info_fields: tuple = (),
        *,
        erase_outdir: bool = True,
        ignore_filter: bool = False,
        ignore_type: bool = True,
        skip_collapse: bool = False,
    ):
        """
        Run truvari pipeline

        Parameters
        ----------
        calls : str
            Calls file
        gt : str
            Ground truth file
        output_file_name : str
            Name of the output H5 concordance file
        outdir : str
            Output directory
        hcr_bed : str, optional
            High confidence region bed file, by default None
        pctseq : float, optional
            Percentage of sequence identity, by default 0.0
        pctsize : float, optional
            Percentage of size identity, by default 0.0
        maxsize : int, optional
            Maximum size for SV comparison, by default 50000. For CNV comparison, consider increasing this value.
        erase_outdir : bool, optional
            Erase output directory if it exists, by default True
        ignore_filter : bool, optional
            If True, ignore FILTER field (remove --passonly flag), by default False
        custom_info_fields : list[str], optional
            Custom info fields to read from the VCFs, in addition SVTYPE and SVLEN
        ignore_type : bool, optional
            If True, ignore SVTYPE when matching to truth, by default True
        skip_collapse : bool, optional
            If True, skip VCF collapsing step for calls (ground truth is always collapsed), by default False

        Returns
        -------
        None
        """

        if dirname(output_file_name) == outdir:
            raise ValueError(
                "output_file_name must not be under outdir to avoid conflicts (with running truvari bench)."
            )
        with tempfile.TemporaryDirectory(dir=dirname(output_file_name)) as workdir:
            if ignore_filter:
                self.logger.info("Ignoring FILTER field in VCFs")
                self.vu.remove_filters(input_vcf=calls, output_vcf=pjoin(workdir, "calls.nofilter.vcf.gz"))
                calls = pjoin(workdir, "calls.nofilter.vcf.gz")
                self.vu.index_vcf(calls)
            self.logger.info(f"Running truvari pipeline with calls: {calls} and gt: {gt}")
            calls_fn = calls
            tmpfiles_to_move = []

            if not skip_collapse:
                collapsed_fn = pjoin(workdir, basename(calls).replace(".vcf.gz", "_collapsed.vcf.gz"))
                self.vu.collapse_vcf(
                    calls_fn,
                    collapsed_fn,
                    bed=hcr_bed,
                    pctseq=pctseq,
                    pctsize=pctsize,
                    maxsize=maxsize,
                    ignore_filter=ignore_filter,
                    ignore_sv_type=ignore_type,
                )
                calls_fn = collapsed_fn
                tmpfiles_to_move.append(calls_fn)

                self.vu.sort_vcf(calls_fn, calls_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz"))
                calls_fn = calls_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz")
                tmpfiles_to_move.append(calls_fn)
                tmpfiles_to_move.append(calls_fn + ".tbi")

                self.vu.index_vcf(calls_fn)
            else:
                self.logger.info("Skipping VCF collapsing for calls")

            gt_fn = gt
            gt_collapsed_fn = pjoin(workdir, basename(gt).replace(".vcf.gz", "_collapsed.vcf.gz"))
            self.vu.collapse_vcf(
                gt_fn,
                gt_collapsed_fn,
                bed=hcr_bed,
                pctseq=pctseq,
                pctsize=pctsize,
                maxsize=maxsize,
                ignore_filter=ignore_filter,
            )
            gt_fn = gt_collapsed_fn
            tmpfiles_to_move.append(gt_fn)
            self.vu.sort_vcf(gt_fn, gt_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz"))
            gt_fn = gt_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz")
            tmpfiles_to_move.append(gt_fn)
            tmpfiles_to_move.append(gt_fn + ".tbi")
            self.vu.index_vcf(gt_fn)

            self.run_truvari(
                calls=calls_fn,
                gt=gt_fn,
                outdir=outdir,
                bed=hcr_bed,
                pctseq=pctseq,
                pctsize=pctsize,
                maxsize=maxsize,
                erase_outdir=erase_outdir,
                ignore_filter=ignore_filter,
                ignore_type=ignore_type,
                skip_collapse=skip_collapse,
            )
            df_base, df_calls = self.truvari_to_dataframes(outdir, custom_info_fields=custom_info_fields)
            df_base.to_hdf(output_file_name, key="base", mode="w")
            df_calls.to_hdf(output_file_name, key="calls", mode="a")
            for tmpfile in tmpfiles_to_move:
                if os.path.exists(tmpfile):
                    shutil.move(tmpfile, outdir)

        self.logger.info(f"truvari pipeline finished with calls: {calls} and gt: {gt}")


def get_parser():
    """
    Get argument parser for SVComparison

    Returns
    -------
    argparse.ArgumentParser
        Argument parser
    """
    import argparse

    parser = argparse.ArgumentParser(description="SV Comparison Pipeline")
    parser.add_argument("--calls", required=True, help="Input calls VCF file")
    parser.add_argument("--gt", required=True, help="Input ground truth VCF file")
    parser.add_argument("--output_filename", required=True, help="output h5 with concordance file")
    parser.add_argument("--outdir", required=True, help="Full path to output dir to TRUVARI results")
    parser.add_argument("--hcr_bed", help="High confidence region bed file")
    parser.add_argument("--pctseq", type=float, default=0.0, help="Percentage of sequence identity")
    parser.add_argument("--pctsize", type=float, default=0.0, help="Percentage of size identity")
    parser.add_argument(
        "--maxsize",
        type=int,
        default=50000,
        help="Maximum size for SV comparison (default: 50000). For CNV comparison, consider increasing this value.",
    )
    parser.add_argument(
        "--custom_info_fields", type=str, action="append", default=[], help="Custom info fields to read from the VCFs"
    )
    parser.add_argument(
        "--ignore_filter",
        action="store_true",
        help="Ignore FILTER field in VCF (remove --passonly flag from truvari commands)",
    )
    parser.add_argument(
        "--skip_collapse",
        action="store_true",
        help="Skip VCF collapsing step for calls (ground truth is always collapsed)",
    )
    parser.add_argument("--verbosity", default="INFO", help="Logging verbosity level")
    return parser


def run(argv):
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    sp = SimplePipeline(args.fc, args.lc, debug=args.d, print_timing=True)
    pipeline = SVComparison(sp, logger)
    pipeline.run_pipeline(
        calls=args.calls,
        gt=args.gt,
        hcr_bed=args.hcr_bed,
        pctseq=args.pctseq,
        pctsize=args.pctsize,
        maxsize=args.maxsize,
        outdir=args.outdir,
        output_file_name=args.output_filename,
        custom_info_fields=tuple(args.custom_info_fields),
        ignore_filter=args.ignore_filter,
        skip_collapse=args.skip_collapse,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
