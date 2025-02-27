import argparse
import logging
import sys
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import featuremap_xgb_prediction
from ugbio_featuremap.featuremap_utils import FeatureMapFields

added_agg_features = featuremap_xgb_prediction.added_agg_features
ppm_added_agg_features = featuremap_xgb_prediction.ppm_added_agg_features
custom_info_fields = [
    FeatureMapFields.READ_COUNT.value,
    FeatureMapFields.FILTERED_COUNT.value,
    FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value,
    FeatureMapFields.HMER_CONTEXT_REF.value,
    FeatureMapFields.HMER_CONTEXT_ALT.value,
    FeatureMapFields.PREV_1.value,
    FeatureMapFields.PREV_2.value,
    FeatureMapFields.PREV_3.value,
    FeatureMapFields.NEXT_1.value,
    FeatureMapFields.NEXT_2.value,
    FeatureMapFields.NEXT_3.value,
    FeatureMapFields.IS_CYCLE_SKIP.value,
]
custom_info_fields.extend(list(added_agg_features))


def split_data_every_2nd_variant(X, y):  # noqa: N802, N803
    # Split the data into training and testing sets - every 2nd record
    X_train = X.iloc[::2]  # Select rows with even indices  # noqa: N806
    X_test = X.iloc[1::2]  # Select rows with odd indices  # noqa: N806
    y_train = y.iloc[::2]
    y_test = y.iloc[1::2]
    return [X_train, X_test, y_train, y_test]


def split_data(X, y, test_size=0.25, random_state=42):  # noqa: N802, N803
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  # noqa: N806
    return [X_train, X_test, y_train, y_test]


def XGBoost_train(X_train, y_train):  # noqa: N802, N803
    # Create XGBoost classifier
    xgb_clf_es = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,  # Set a high number, early stopping will decide when to stop
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        enable_categorical=True,
    )
    # Fit the model
    xgb_clf_es.fit(X_train, y_train, verbose=True)
    return xgb_clf_es


def XGBoost_test(xgb_clf_es, X_test):  # noqa: N802, N803
    # Make predictions
    y_pred_es = xgb_clf_es.predict(X_test)
    return y_pred_es


def XGBoost_evaluate_model(xgb_clf_es, y_test, y_pred_es):  # noqa: N802, N803
    # Evaluate the model
    accuracy_es = accuracy_score(y_test, y_pred_es)
    logger.debug(f"\nAccuracy : {accuracy_es:.2f}")

    logger.debug(f"Accuracy:{accuracy_score(y_test, y_pred_es)}")
    logger.debug(f"\nClassification Report:\n{classification_report(y_test, y_pred_es)}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": xgb_clf_es.get_booster().feature_names, "importance": xgb_clf_es.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.debug("\nFeature Importance:")
    logger.debug(f"{feature_importance}")


def variants_classification_vs_probability(xgb_clf_es, X_test, y_test, out_figure_path):  # noqa: N802, N803
    probabilities = xgb_clf_es.predict_proba(X_test)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    df_probabilities["true_status"] = pd.Series(y_test.values)

    plt.figure(figsize=(10, 6))
    plt.hist(df_probabilities[df_probabilities["true_status"] == 1]["1"], bins=20, alpha=0.4, label="True Variants")
    plt.hist(df_probabilities[df_probabilities["true_status"] == 0]["1"], bins=20, alpha=0.4, label="False Variants")

    # Add labels and legend
    plt.xlabel("probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.yscale("log")
    plt.savefig(out_figure_path)


def cross_validation(xgb_clf_es, X, y, cv=5, cv_training_diff_cutoff=0.1):  # noqa: N802, N803
    cv_scores = cross_val_score(xgb_clf_es, X, y, cv=cv)

    logger.debug(f"Cross-validation scores: {cv_scores}")
    logger.debug(f"Mean CV score: {np.mean(cv_scores):.4f}")
    logger.debug(f"Standard deviation of CV scores: {np.std(cv_scores):.4f}")

    # If CV scores are significantly lower than training score, it may indicate overfitting
    if xgb_clf_es.score(X, y) - np.mean(cv_scores) > cv_training_diff_cutoff:  # This threshold can be adjusted
        logger.debug("Model might be overfitting.")
    else:
        logger.debug("No clear sign of overfitting.")


def aggreagte_vcf_from_vcfeval_dir(
    vcfeval_dir: str,
    added_agg_features: dict,
    ppm_added_agg_features: dict,
    custom_info_fields: list[str],
    chromosome: str,
):
    tp_vcf = pjoin(vcfeval_dir, "tp.vcf.gz")
    fp_vcf = pjoin(vcfeval_dir, "fp.vcf.gz")

    # read vcf block to dataframe
    custom_info_fields = featuremap_xgb_prediction.default_custom_info_fields
    custom_info_fields.extend(featuremap_xgb_prediction.ppm_custom_info_fields)

    out_vcf = {}
    for tag, in_vcf in zip(["tp", "fp"], [tp_vcf, fp_vcf], strict=False):
        out_vcf_file = in_vcf.replace(".vcf.gz", f".{chromosome}.agg_params.vcf.gz")
        df_variants = vcftools.get_vcf_df(in_vcf, custom_info_fields=custom_info_fields)
        df_variants = featuremap_xgb_prediction.df_vcf_manual_aggregation(df_variants)

        with pysam.VariantFile(in_vcf) as vcfin:
            hdr = vcfin.header
            featuremap_xgb_prediction.add_agg_fields_to_header(hdr)
            with pysam.VariantFile(out_vcf_file, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    featuremap_xgb_prediction.process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params=True)
            vcfout.close()
            vcfin.close()
        pysam.tabix_index(out_vcf_file, preset="vcf", min_shift=0, force=True)
        out_vcf[tag] = out_vcf_file

    custom_info_fields.extend(list(added_agg_features))
    custom_info_fields.extend(list(ppm_added_agg_features))
    df_fp = vcftools.get_vcf_df(out_vcf["fp"], custom_info_fields=custom_info_fields, chromosome=chromosome)
    df_tp = vcftools.get_vcf_df(out_vcf["tp"], custom_info_fields=custom_info_fields, chromosome=chromosome)
    df_fp["label"] = "negative"
    df_tp["label"] = "positive"
    df_all_variants = pd.concat([df_fp, df_tp])
    return df_all_variants


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train xgboost model using pileup featuremap aggregated data",
        description=run.__doc__,
    )
    parser.add_argument(
        "-tp",
        "--featuremap_pileup_tp",
        type=str,
        required=True,
        help="""Featuremap pileup vcf file holding true positive variants""",
    )
    parser.add_argument(
        "-fp",
        "--featuremap_pileup_fp",
        type=str,
        required=True,
        help="""Featuremap pileup vcf file holding false positive variants""",
    )
    parser.add_argument(
        "-o",
        "--output_model",
        type=str,
        required=True,
        help="""Output model file""",
    )
    parser.add_argument(
        "-min_alt_reads",
        "--min_supporting_reads_cutoff",
        type=str,
        required=True,
        help="""minimal value of alternative supporting reads""",
    )
    parser.add_argument(
        "-max_alt_reads",
        "--max_supporting_reads_cutoff",
        type=str,
        required=True,
        help="""maximal value of alternative supporting reads""",
    )
    parser.add_argument(
        "-chr",
        "--chromosome",
        type=str,
        required=False,
        help="""Optional chromosome name to use for data loading""",
    )
    parser.add_argument(
        "-is_ppm",
        "--is_ppmSeq",
        action="store_true",
        help="""Wether the input featuremap_pileup is ppmeSeq""",
    )
    parser.add_argument(
        "-split_data_every_2nd_variant",
        "--split_data_every_2nd_variant",
        action="store_true",
        help=(
            "Wether to split the data every 2nd variant (resulting with 50% training and 50% testing). "
            "Overrides the default split_data function"
        ),
    )
    parser.add_argument(
        "-test_size",
        "--test_size",
        type=float,
        default=0.25,
        required=False,
        help=(
            "Optional test size fraction for the split_data function. "
            "Default is 0.25 (25% for testing and 75% for training)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        required=False,
        default=True,
        help="""Whether to print debug messages (default: True)""",
    )
    return parser.parse_args(argv[1:])


def run(argv):  # noqa: C901,PLR0912,PLR0915
    """train xgboost model using pileup featuremap aggregated data"""
    args_in = __parse_args(argv)

    if args_in.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    out_file = Path(args_in.output_model)
    if out_file.is_file():
        logger.debug(f"out file exists: {out_file}. please provide a different output file path")
        sys.exit(1)

    out_file = args_in.output_model
    if not out_file.endswith(".json"):
        logger.debug("adding .json suffix to the output vcf file")
        out_file = out_file + ".json"

    custom_info_fields = featuremap_xgb_prediction.default_custom_info_fields
    custom_info_fields.extend(featuremap_xgb_prediction.ppm_custom_info_fields)

    # aggregate vcf
    vcf_agg_file = {}
    for tag, sorted_featuremap in zip(
        ["tp", "fp"], [args_in.featuremap_pileup_tp, args_in.featuremap_pileup_fp], strict=False
    ):
        output_vcf = sorted_featuremap.replace(".vcf.gz", ".agg_params.vcf.gz")

        df_variants = vcftools.get_vcf_df(sorted_featuremap, custom_info_fields=custom_info_fields)
        df_variants = featuremap_xgb_prediction.df_vcf_manual_aggregation(df_variants)
        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            featuremap_xgb_prediction.add_agg_fields_to_header(hdr)
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    featuremap_xgb_prediction.process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params=True)
            vcfout.close()
            vcfin.close()
        pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
        vcf_agg_file[tag] = output_vcf

    logger.debug("finished writing aggregate params vcf files:")
    logger.debug(vcf_agg_file["fp"])
    logger.debug(vcf_agg_file["tp"])

    custom_info_fields.extend(list(added_agg_features))
    custom_info_fields.extend(list(ppm_added_agg_features))
    df_fp = vcftools.get_vcf_df(
        vcf_agg_file["fp"], custom_info_fields=custom_info_fields, chromosome=args_in.chromosome
    )
    df_tp = vcftools.get_vcf_df(
        vcf_agg_file["tp"], custom_info_fields=custom_info_fields, chromosome=args_in.chromosome
    )
    df_fp["label"] = "negative"
    df_tp["label"] = "positive"
    df_all_variants = pd.concat([df_fp, df_tp])
    logger.debug("finished reading pileup featurepmap vcf files")

    training_features = ["dp", "vaf", "chrom", "pos", "qual", "ref"]
    training_features.extend(list(added_agg_features))
    if args_in.is_ppmSeq:
        training_features.extend(list(ppm_added_agg_features))
    training_features = [f.lower() for f in training_features]

    lower_cutoff = int(args_in.min_supporting_reads_cutoff)  # noqa: F841
    upper_cutoff = int(args_in.max_supporting_reads_cutoff)  # noqa: F841

    X = df_all_variants.query("alt_reads>=@lower_cutoff and alt_reads<=@upper_cutoff")[training_features]  # noqa: N806
    y = df_all_variants.query("alt_reads>=@lower_cutoff and alt_reads<=@upper_cutoff")["label"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    featuremap_xgb_prediction.set_categorial_columns(X)
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    if args_in.split_data_every_2nd_variant:
        [X_train, X_test, y_train, y_test] = split_data_every_2nd_variant(X, y)  # noqa: N806
    elif args_in.test_size < 1 and args_in.test_size > 0:
        [X_train, X_test, y_train, y_test] = split_data(X, y, test_size=args_in.test_size)  # noqa: N806
    xgb_clf_es = XGBoost_train(X_train, y_train)

    logger.debug("Model evaluation on TEST data:")
    y_pred_es = XGBoost_test(xgb_clf_es, X_test)
    XGBoost_evaluate_model(xgb_clf_es, y_test, y_pred_es)

    out_figure_path = out_file.replace(".json", ".probability_histogram.png")
    variants_classification_vs_probability(xgb_clf_es, X_test, y_test, out_figure_path)

    logger.debug("Model evaluation on TRAIN data")
    y_pred_es = XGBoost_test(xgb_clf_es, X_train)
    XGBoost_evaluate_model(xgb_clf_es, y_train, y_pred_es)

    cross_validation(xgb_clf_es, X, y)

    # save model
    xgb_clf_es.save_model(out_file)
    logger.debug(f"out model is save in: {out_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
