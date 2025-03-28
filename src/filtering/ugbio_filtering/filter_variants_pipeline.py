#!/env/python
import argparse
import logging
import os.path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import tqdm
from ugbio_core import math_utils
from ugbio_core.vcfbed import vcftools

from ugbio_filtering import multiallelics, training_prep, variant_filtering_utils
from ugbio_filtering.blacklist import blacklist_cg_insertions, merge_blacklists


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(prog="filter_variants_pipeline.py", description="Filter VCF")
    ap_var.add_argument(
        "--input_file", help="Name of the input VCF file (requires .tbi index)", type=str, required=True
    )
    ap_var.add_argument("--model_file", help="Pickle model file", type=str, required=False)
    ap_var.add_argument("--blacklist", help="Blacklist file", type=str, required=False)
    ap_var.add_argument(
        "--custom_annotations",
        help="Custom INFO annotations to read from the VCF (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )

    ap_var.add_argument(
        "--blacklist_cg_insertions",
        help="Should CCG/GGC insertions be filtered out?",
        action="store_true",
    )
    ap_var.add_argument(
        "--treat_multiallelics",
        help="Should special treatment be applied to multiallelic and spanning deletions",
        default=False,
        action="store_true",
    )
    ap_var.add_argument(
        "--recalibrate_genotype", help="Use if the model allows to re-call genotype", default=False, action="store_true"
    )

    ap_var.add_argument(
        "--ref_fasta", help="Reference FASTA file (only required for multiallelic treatment)", required=False, type=str
    )
    ap_var.add_argument("--output_file", help="Output VCF file", type=str, required=True)
    ap_var.add_argument(
        "--limit_to_contigs", help="Limit filtering to these contigs", nargs="+", type=str, default=None
    )
    return ap_var.parse_args(argv)


def protected_add(hdr, field, n_vals, param_type, description):
    if field not in hdr:
        hdr.add(field, n_vals, param_type, description)


def run(argv: list[str]):  # noqa C901 PLR0912 PLR0915# pylint: disable=too-many-branches, disable=too-many-statements
    "POST-GATK variant filtering"
    args = parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        if args.model_file is not None:
            logger.info(f"Loading model from {args.model_file}")
            with open(args.model_file, "rb") as model_file:
                mf = pickle.load(model_file)  # noqa S301
                model = mf["xgb"]
                transformer = mf["transformer"]
        if args.blacklist is not None:
            logger.info(f"Loading blacklist from {args.blacklist}")
            with open(args.blacklist, "rb") as blf:
                blacklists = pickle.load(blf)  # noqa S301
        if args.treat_multiallelics and args.ref_fasta is None:
            raise ValueError("Reference FASTA file is required for multiallelic treatment")
        assert os.path.exists(args.input_file), f"Input file {args.input_file} does not exist"  # noqa S101
        assert os.path.exists(args.input_file + ".tbi"), f"Index file {args.input_file}.tbi does not exist"  # noqa S101
        with pysam.VariantFile(args.input_file) as infile:
            hdr = infile.header
            if args.model_file is not None:
                protected_add(hdr.filters, "LOW_SCORE", None, None, "Low decision tree score")
            if args.blacklist is not None or args.blacklist_cg_insertions:
                protected_add(hdr.info, "BLACKLST", ".", "String", "blacklist")
            if args.model_file is not None:
                protected_add(hdr.info, "TREE_SCORE", 1, "Float", "Filtering score")

            with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
                if args.limit_to_contigs is None:
                    it = ((x, infile.fetch(str(x))) for x in infile.header.contigs.keys())
                else:
                    it = ((x, infile.fetch(x)) for x in args.limit_to_contigs)
                for contig, chunk in it:  # pylint: disable=too-many-nested-blocks
                    logger.info(f"Filtering variants from {contig}")
                    vcf_df = vcftools.get_vcf_df(
                        args.input_file, chromosome=str(contig), custom_info_fields=args.custom_annotations
                    )
                    if vcf_df.shape[0] == 0:
                        logger.info(f"No variants found on {contig}")
                        continue
                    logger.info(f"{vcf_df.shape[0]} variants found on {contig}")
                    if args.blacklist is not None:
                        blacklist_app = [x.apply(vcf_df) for x in blacklists]
                        blacklist = merge_blacklists(blacklist_app)
                        logger.info("Applying blacklist")
                    else:
                        blacklist = pd.Series("PASS", index=vcf_df.index, dtype=str)

                    if args.blacklist_cg_insertions:
                        cg_blacklist = blacklist_cg_insertions(vcf_df)
                        blacklist = merge_blacklists([cg_blacklist, blacklist])
                        logger.info("Marking CG insertions")

                    if args.model_file is not None:
                        if args.treat_multiallelics:
                            df_original = vcf_df.copy()
                            logger.info("Processing multiallelics -> pre-classifier")
                            vcf_df = training_prep.process_multiallelic_spandel(
                                vcf_df, args.ref_fasta, str(contig), args.input_file
                            )
                        logger.info("Applying classifier")
                        _, scores = variant_filtering_utils.apply_model(vcf_df, model, transformer)

                        if args.treat_multiallelics:
                            logger.info("Treating multiallelics -> post-classifier")

                            set_source = [x in df_original.index for x in vcf_df.index]
                            set_dest = [x in vcf_df.index for x in df_original.index]
                            df_original["ml_lik"] = pd.Series(
                                [list(x) for x in scores[set_source, :]], index=df_original.loc[set_dest].index
                            )
                            df_original = variant_filtering_utils.combine_multiallelic_spandel(  # noqa E501
                                vcf_df, df_original, scores
                            )
                            vcf_df = df_original.copy()
                        else:
                            vcf_df["ml_lik"] = pd.Series([list(x) for x in scores], index=vcf_df.index)

                        likelihoods = np.zeros((vcf_df.shape[0], max(vcf_df["ml_lik"].apply(len))))
                        for i, r in enumerate(vcf_df["ml_lik"]):
                            likelihoods[i, : len(r)] = r

                        phreds = math_utils.phred(likelihoods + 1e-10)
                        quals = np.clip(30 + phreds[:, 0] - np.min(phreds[:, 1:], axis=1), 0, None)
                        tmp = np.argsort(phreds, axis=1)
                        gq = (
                            phreds[np.arange(phreds.shape[0]), tmp[:, 1]]
                            - phreds[np.arange(phreds.shape[0]), tmp[:, 0]]
                        )

                    logger.info("Writing records")
                    for i, rec in tqdm.tqdm(enumerate(chunk)):
                        if args.model_file is not None:
                            if quals[i] <= 30:  # noqa PLR2004
                                if "PASS" in rec.filter.keys():
                                    del rec.filter["PASS"]
                                rec.filter.add("LOW_SCORE")
                            if not args.recalibrate_genotype:
                                rec.info["TREE_SCORE"] = float(quals[i])
                            else:
                                rec.samples[0]["GQ"] = int(gq[i])
                                assert rec.alleles is not None  # noqa S101
                                rec.samples[0]["PL"] = [
                                    int(x) for x in phreds[i, : (len(rec.alleles) + 1) * len(rec.alleles) // 2]
                                ]
                                rec.samples[0]["GT"] = multiallelics.get_gt_from_pl_idx(
                                    np.argmin(rec.samples[0]["PL"]).astype(int)
                                )

                        if blacklist is not None:
                            if blacklist[i] != "PASS":
                                blacklists_info = []
                                for value in blacklist[i].split(";"):
                                    if value != "PASS":
                                        blacklists_info.append(value)
                                if len(blacklists_info) != 0:
                                    rec.info["BLACKLST"] = blacklists_info
                        if len(rec.filter) == 0:
                            rec.filter.add("PASS")

                        outfile.write(rec)
                    logger.info(f"{contig} done")

        cmd = ["bcftools", "index", "-t", args.output_file]
        subprocess.check_call(cmd)
        logger.info("Variant filtering run: success")

    except Exception as err:
        exc_info = sys.exc_info()
        logger.error(exc_info[:2])
        logger.exception(err)
        logger.error("Variant filtering run: failed")
        raise err


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
