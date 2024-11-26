import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from ugbio_core.logger import logger

# import ugbio_featuremap.featuremap_xgb_prediction as featuremap_xgb_prediction
from ugbio_featuremap.featuremap_utils import FeatureMapFields

sys.path.append("/data/Runs/proj/VariantCalling/ugvc/vcfbed")
import vcftools

# import ugvc.vcfbed.vcftools as vcftools  #move to ugbio_core
sys.path.append("/data/Runs/proj/ugbio-utils/src/featuremap/ugbio_featuremap")
import featuremap_xgb_prediction

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


def XGBoost_train(X_train, y_train):  # noqa: N802, N803
    # Create XGBoost classifier
    xgb_clf_es = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,  # Set a high number, early stopping will decide when to stop
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
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
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(xgb_clf_es, X, y, cv=cv)

    logger.debug(f"Cross-validation scores: {cv_scores}")
    logger.debug(f"Mean CV score: {np.mean(cv_scores):.4f}")
    logger.debug(f"Standard deviation of CV scores: {np.std(cv_scores):.4f}")

    # If CV scores are significantly lower than training score, it may indicate overfitting
    if xgb_clf_es.score(X, y) - np.mean(cv_scores) > cv_training_diff_cutoff:  # This threshold can be adjusted
        logger.debug("Model might be overfitting.")
    else:
        logger.debug("No clear sign of overfitting.")


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
        type=bool,
        required=True,
        help="""Wether the input featuremap_pileup is ppmeSeq""",
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

    if args_in.is_ppmSeq:
        custom_info_fields.extend(list(ppm_added_agg_features))

    training_features = ["dp", "vaf", "chrom", "pos", "qual", "ref"]
    training_features.extend(custom_info_fields)
    training_features = [f.lower() for f in training_features]

    # aggregate vcf
    vcf_agg_file = {}
    for tag, sorted_featuremap in zip(
        ["tp", "fp"], [args_in.featuremap_pileup_tp, args_in.featuremap_pileup_fp], strict=False
    ):
        output_vcf = sorted_featuremap.replace(".vcf.gz", ".agg_params.vcf.gz")
        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            # adding manual aggregation fields
            # for field, field_type, field_description in zip(
            for field in added_agg_features:
                field_type = added_agg_features[field][1]
                field_description = added_agg_features[field][0]
                hdr.info.add(field, 1, field_type, field_description)
            if "st" in hdr.info:
                for field in ppm_added_agg_features:
                    field_type = ppm_added_agg_features[field][1]
                    field_description = ppm_added_agg_features[field][0]
                    hdr.info.add(field, 1, field_type, field_description)
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    record_dict_for_xgb = featuremap_xgb_prediction.record_manual_aggregation(row)
                    for key in added_agg_features:
                        row.info[key] = record_dict_for_xgb[key]
                    if "st" in hdr.info:
                        for key in ppm_added_agg_features:
                            row.info[key] = record_dict_for_xgb[key]
                    vcfout.write(row)
            vcfout.close()
            vcfin.close()
        pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
        vcf_agg_file[tag] = output_vcf
    logger.debug("finishe writing aggregate params vcf files:")
    logger.debug(vcf_agg_file["fp"])
    logger.debug(vcf_agg_file["tp"])

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

    lower_cutoff = int(args_in.min_supporting_reads_cutoff)  # noqa: F841
    upper_cutoff = int(args_in.max_supporting_reads_cutoff)  # noqa: F841

    X = df_all_variants.query("alt_reads>=@lower_cutoff and alt_reads<=@upper_cutoff")[training_features]  # noqa: N806
    y = df_all_variants.query("alt_reads>=@lower_cutoff and alt_reads<=@upper_cutoff")["label"].apply(
        lambda x: 1 if x == "positive" else 0
    )

    featuremap_xgb_prediction.set_categorial_columns(X)
    [X_train, X_test, y_train, y_test] = split_data_every_2nd_variant(X, y)  # noqa: N806

    xgb_clf_es = XGBoost_train(X_train, y_train)
    y_pred_es = XGBoost_test(xgb_clf_es, X_test)

    XGBoost_evaluate_model(xgb_clf_es, y_test, y_pred_es)

    out_figure_path = out_file.replace(".json", ".probability_histogram.png")
    variants_classification_vs_probability(xgb_clf_es, X_test, y_test, out_figure_path)
    cross_validation(xgb_clf_es, X, y)

    # save model
    xgb_clf_es.save_model(out_file)
    logger.debug(f"out model is save in: {out_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
