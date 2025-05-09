# SV comparison pipeline
import os
import shutil

import simppl
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute


class SVComparison(simppl.Pipeline):
    """
    SV comparison pipeline
    """

    def __init__(self, simple_pipeline: SimplePipeline | None = None):
        """Combines VCF in parts from GATK and indices the result

        Parameters
        ----------
        simple_pipeline : SimplePipeline, optional
            Optional SimplePipeline object for executing shell commands
        """
        self.sp = simple_pipeline

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

    def run_truvari(
        self,
        calls: str,
        gt: str,
        outdir: str,
        bed: str = None,
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
        gt : st
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
            "--passonly",
        ]

        if bed:
            truvari_cmd.extend(["--includebed", bed])
        if pctseq:
            truvari_cmd.extend(["--pctseq", str(pctseq)])
        if pctsize:
            truvari_cmd.extend(["--pctsize", str(pctsize)])

        self.logger.info(f"truvari command: {' '.join(truvari_cmd)}")
        self.__execute(" ".join(truvari_cmd), output_file=outdir)
