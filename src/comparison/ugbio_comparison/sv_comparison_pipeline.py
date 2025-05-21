# SV comparison pipeline
import logging
import os
import shutil
import subprocess
import sys
from os.path import join as pjoin

import pandas as pd
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_comparison.vcf_pipeline_utils import VcfPipelineUtils


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
        self.vpu = VcfPipelineUtils(self.sp)
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

    def collapse_vcf(
        self,
        vcf: str,
        output_vcf: str,
        bed: str | None = None,
        pctseq: float = 0.0,
        pctsize: float = 0.0,
    ):
        """
        Collapse VCF using truvari collapse

        Parameters
        ----------
        vcf : str
            Input VCF file
        output_vcf : str
            Output VCF file
        bed : str, optional
            Bed file, by default None
        pctseq : float, optional
            Percentage of sequence identity, by default 0.0
        pctsize : float, optional
            Percentage of size identity, by default 0.0

        Returns
        -------
        None
        """

        truvari_cmd = ["truvari", "collapse", "-i", vcf, "--passonly", "-t"]

        if bed:
            truvari_cmd.extend(["--bed", bed])
        truvari_cmd.extend(["--pctseq", str(pctseq)])
        truvari_cmd.extend(["--pctsize", str(pctsize)])

        self.logger.info(f"truvari command: {' '.join(truvari_cmd)}")
        p1 = subprocess.Popen(truvari_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["bcftools", "view", "-Oz", "-o", output_vcf], stdin=p1.stdout)  # noqa: S607
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        p2.communicate()  # Wait for p2 to finish
        p1.wait()  # Wait for p1 to finish
        if p1.returncode != 0:
            raise RuntimeError(f"truvari collapse failed with error code {p1.returncode}")
        if p2.returncode != 0:
            raise RuntimeError(f"bcftools view failed with error code {p2.returncode}")

        removed_vcf_path = "removed.vcf"  # Parameterize the file path
        if os.path.exists(removed_vcf_path):
            os.unlink(removed_vcf_path)
            self.logger.info(f"Deleted temporary file: {removed_vcf_path}")
        else:
            self.logger.warning(f"Temporary file not found: {removed_vcf_path}")

    def run_truvari(
        self,
        calls: str,
        gt: str,
        outdir: str,
        bed: str | None = None,
        pctseq: float = 0.0,
        pctsize: float = 0.0,
        *,
        erase_outdir: bool = True,
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
        erase_outdir : bool, optional
            Erase output directory if it exists, by default True

        Returns
        -------
        None
        """
        if erase_outdir and os.path.exists(outdir):
            shutil.rmtree(outdir)

        truvari_cmd = [
            "truvari",
            "bench",
            "-b",
            gt,
            "-c",
            calls,
            "-o",
            outdir,
            "-t",
            "--passonly",
        ]

        if bed:
            truvari_cmd.extend(["--includebed", bed])
        truvari_cmd.extend(["--pctseq", str(pctseq)])
        truvari_cmd.extend(["--pctsize", str(pctsize)])

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
        info_fields_to_read: list[str] = list(("SVTYPE", "SVLEN") + custom_info_fields)
        df_tp_base = vcftools.get_vcf_df(pjoin(truvari_dir, "tp-base.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_tp_base["svlen"] = df_tp_base["svlen"].apply(lambda x: x[0] if isinstance(x, tuple) else x).fillna(0)
        df_tp_base["label"] = "TP"
        df_fn = vcftools.get_vcf_df(pjoin(truvari_dir, "fn.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_fn["svlen"] = df_fn["svlen"].apply(lambda x: x[0] if isinstance(x, tuple) else x).fillna(0)
        df_fn["label"] = "FN"
        df_base = pd.concat((df_tp_base, df_fn))

        df_tp_calls = vcftools.get_vcf_df(pjoin(truvari_dir, "tp-comp.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_tp_calls["label"] = "TP"
        df_tp_calls["svlen"] = df_tp_calls["svlen"].apply(lambda x: x[0] if isinstance(x, tuple) else x).fillna(0)

        df_fp = vcftools.get_vcf_df(pjoin(truvari_dir, "fp.vcf.gz"), custom_info_fields=info_fields_to_read)
        df_fp["label"] = "FP"
        df_fp["svlen"] = df_fp["svlen"].apply(lambda x: x[0] if isinstance(x, tuple) else x).fillna(0)

        df_calls = pd.concat((df_tp_calls, df_fp))
        return df_base, df_calls

    def run_pipeline(
        self,
        calls: str,
        gt: str,
        output_file_name: str,
        outdir: str,
        hcr_bed: str | None = None,
        pctseq: float = 0.0,
        pctsize: float = 0.0,
        custom_info_fields: tuple = (),
        *,
        erase_outdir: bool = True,
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
        erase_outdir : bool, optional
            Erase output directory if it exists, by default True
        custom_info_fields : list[str], optional
            Custom info fields to read from the VCFs, in addition SVTYPE and SVLEN

        Returns
        -------
        None
        """
        self.logger.info(f"Running truvari pipeline with calls: {calls} and gt: {gt}")
        calls_fn = calls
        tmpfiles_to_move = []
        self.collapse_vcf(
            calls_fn,
            calls_fn.replace(".vcf.gz", "_collapsed.vcf.gz"),
            bed=hcr_bed,
            pctseq=pctseq,
            pctsize=pctsize,
        )
        calls_fn = calls_fn.replace(".vcf.gz", "_collapsed.vcf.gz")
        tmpfiles_to_move.append(calls_fn)

        self.vpu.sort_vcf(calls_fn, calls_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz"))
        calls_fn = calls_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz")
        tmpfiles_to_move.append(calls_fn)
        tmpfiles_to_move.append(calls_fn + ".tbi")

        self.vpu.index_vcf(calls_fn)

        gt_fn = gt

        self.collapse_vcf(
            gt_fn,
            gt_fn.replace(".vcf.gz", "_collapsed.vcf.gz"),
            bed=hcr_bed,
            pctseq=pctseq,
            pctsize=pctsize,
        )
        gt_fn = gt_fn.replace(".vcf.gz", "_collapsed.vcf.gz")
        tmpfiles_to_move.append(gt_fn)
        self.vpu.sort_vcf(gt_fn, gt_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz"))
        gt_fn = gt_fn.replace("_collapsed.vcf.gz", "_collapsed.sort.vcf.gz")
        tmpfiles_to_move.append(gt_fn)
        tmpfiles_to_move.append(gt_fn + ".tbi")
        self.vpu.index_vcf(gt_fn)

        self.run_truvari(
            calls=calls_fn,
            gt=gt_fn,
            outdir=outdir,
            bed=hcr_bed,
            pctseq=pctseq,
            pctsize=pctsize,
            erase_outdir=erase_outdir,
        )
        df_base, df_calls = self.truvari_to_dataframes(outdir, custom_info_fields=custom_info_fields)
        df_base.to_hdf(output_file_name, key="base", mode="w")
        df_calls.to_hdf(output_file_name, key="calls", mode="a")
        for tmpfile in tmpfiles_to_move:
            if os.path.exists(tmpfile):
                shutil.move(tmpfile, outdir)
        self.logger.info(f"truvari pipeline finished with calls: {calls_fn} and gt: {gt_fn}")


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
    parser.add_argument("--custom_info_fields", nargs="+", default=[], help="Custom info fields to read from the VCFs")
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
        outdir=args.outdir,
        output_file_name=args.output_filename,
        custom_info_fields=tuple(args.custom_info_fields),
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
