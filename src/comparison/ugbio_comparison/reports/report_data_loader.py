import dill as pickle
import numpy as np
from ugbio_core.h5_utils import read_hdf

from ugbio_comparison.reports.concordance_report_utils import ErrorType


class ReportDataLoader:
    def __init__(self, concordance_file: str, reference_version: str, exome_column_name: str):
        self.concordance_file = concordance_file
        self.reference_version = self.__simplify_reference_version(reference_version)
        self.columns = self.__columns_subset(exome_column_name)
        self.rename_dict = self.__get_rename_dict()

    def load_concordance_df(self):
        concordance = read_hdf(
            self.concordance_file, key="all", skip_keys=["concordance", "input_args"], columns_subset=self.columns
        )
        concordance = concordance.rename(columns=self.rename_dict)
        concordance["fp"] = (concordance["call"] == "FP") | (concordance["call"] == "FP_CA")
        concordance["fn"] = (concordance["base"] == "FN") | (concordance["base"] == "FN_CA")
        concordance["tp"] = concordance["call"] == "TP"
        np.seterr(invalid="ignore")
        if "vaf" not in concordance.columns:
            concordance["vaf"] = concordance[["ad", "dp"]].apply(
                lambda x: (0,) if not isinstance(x.ad, tuple) else tuple(np.array(x.ad) / x.dp), axis=1
            )
        concordance["max_vaf"] = concordance["vaf"].apply(lambda x: 0 if isinstance(x, float) else max(x))
        if "qual" not in concordance or (~concordance.qual.isna()).sum() == 0:
            concordance["qual"] = concordance["tree_score"]

        genotypes = concordance["gt_ground_truth"] + concordance["gt_ultima"]
        concordance["error_type"] = genotypes.apply(self.get_error_type)
        return concordance.rename(columns={"hmer_indel_length": "hmer_length"})

    def load_sv_concordance_df(self) -> tuple[dict, dict]:
        """
        Read a pickle file and convert it to a dictionary of DataFrames.

        Parameters
        ----------
        Returns
        -------
        tuple[dict,dict]
            No GT statistics and GT statistics dictionaries.
        """
        with open(self.concordance_file, "rb") as f:
            data = pickle.load(f)  # noqa: S301 - statistics pickled by collect_sv_stats
        dfs_no_gt = {k: v for k, v in data.items() if k.endswith("counts")}
        dfs_with_gt = {k: v for k, v in data.items() if not k.endswith("counts")}

        return dfs_no_gt, dfs_with_gt

    def __get_rename_dict(self):
        if self.reference_version == "hg38":
            return {}
        if self.reference_version == "hg19":
            return {
                "mappability.hg19.0_tab_no_chr": "mappability.0",
                "ug_hcr_hg19_no_chr": "ug_hcr",
            }
        return {}

    def __simplify_reference_version(self, reference_version: str) -> str:
        """
        Simplify the reference version string to a standard form.

        Parameters
        ----------
        reference_version : str
            The reference version string to simplify

        Returns
        -------
        str
            Simplified reference version ('hg38' or 'hg19')
        """
        # Other options not covered: hs38DH, GCA_000001405.15, GCF_000001405.26 (hg38),
        # hs37d5, human_g1k_v37, GCA_000001405.1, GCF_000001405.13 (hg19)
        hg38_patterns = ["hg38", "GRCh38"]
        hg19_patterns = ["hg19", "GRCh37", "b37"]

        for pattern in hg38_patterns:
            if pattern in reference_version:
                return "hg38"

        for pattern in hg19_patterns:
            if pattern in reference_version:
                return "hg19"

        # If no pattern matches, return the original version
        return reference_version

    def __columns_subset(self, exome_column_name):
        common_columns = [
            "indel",
            "hmer_indel_length",
            "tree_score",
            "filter",
            "blacklst",
            "classify",
            "classify_gt",
            "indel_length",
            "hmer_indel_nuc",
            "well_mapped_coverage",
            "base",
            "call",
            "gt_ground_truth",
            "gt_ultima",
            "ad",
            "dp",
            "vaf",
            "ref",
            "alleles",
            exome_column_name,
            "gc_content",
            "indel_classify",
            "qual",
            "gq",
        ]
        if self.reference_version == "hg38":
            return common_columns + ["mappability.0", "ug_hcr", "callable"]

        if self.reference_version == "hg19":
            return common_columns + [
                "mappability.hg19.0_tab_no_chr",
                "ug_hcr_hg19_no_chr",
                "callable",
            ]

        return common_columns

    @staticmethod
    def get_error_type(genotype_pair: tuple) -> ErrorType:
        gtr_gt = set(genotype_pair[0:2])
        call_gt = set(genotype_pair[2:4])

        if gtr_gt == call_gt:
            return ErrorType.NO_ERROR

        if gtr_gt in ({0}, {None}):
            return ErrorType.NOISE

        if call_gt in ({0}, {None}):
            return ErrorType.NO_VARIANT

        if gtr_gt.intersection(call_gt) == gtr_gt:
            return ErrorType.HOM_TO_HET

        if gtr_gt.intersection(call_gt) == call_gt:
            return ErrorType.HET_TO_HOM

        return ErrorType.WRONG_ALLELE
