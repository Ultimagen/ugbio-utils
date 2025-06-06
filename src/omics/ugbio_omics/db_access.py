"""
Command line access to PAPYRUS mongoDB.
See documentation in
https://ultimagen.atlassian.net/wiki/spaces/AG/pages/1428914739/Papyrus+Metrics+Infrastructure+Proof+of+Concept#Command-Line-Access
"""

import io
import json
import os
import warnings
from enum import Enum
from typing import Any

import pandas as pd
import pymongo


class Collections(Enum):
    CROMWELL = "pipelines"
    RUNS = "runs"
    SAMPLES = "samples"
    EXECUTIONS = "executions"
    APPLICATION_QC = "application_qc"


def set_papyrus_access():
    """Disables/Enables papyrus access depends on PAPYRUS_ACCESS_STRING env"""
    disable_papyrus_access = False
    collections = {}
    if "PAPYRUS_ACCESS_STRING" in os.environ:
        my_client = initialize_client()
        my_db = my_client["pipelines"]
        collections[Collections.CROMWELL] = my_db["pipelines"]
        collections[Collections.RUNS] = my_db["runs"]
        collections[Collections.EXECUTIONS] = my_db["runs.executions"]
        collections[Collections.SAMPLES] = my_db["runs.executions.samples"]
        collections[Collections.APPLICATION_QC] = my_db["runs.executions.samples.applicationQc"]
    else:
        warnings.warn(
            "Define PAPYRUS_ACCESS_STRING environmental variable to enable access to Papyrus",
            stacklevel=2,
        )
        warnings.warn(
            "Example: export PAPYRUS_ACCESS_STRING=mongodb+srv://[user]:[passwd]@testcluster.jm2x3.mongodb.net/test",
            stacklevel=2,
        )
        disable_papyrus_access = True
    return disable_papyrus_access, collections


def initialize_client() -> pymongo.MongoClient:
    """Initializes pymongo client with the access string that is read from PAPYRUS_ACCESS_STRING
    environmental variable.

    Parameters:
    -----------
    None

    Returns
    -------
    pymongo.MongoClient
    """

    myclient = pymongo.MongoClient(os.environ["PAPYRUS_ACCESS_STRING"])
    return myclient


def query_database(query: dict, collection: str = "pipelines", **kwargs: Any) -> list:
    """Querying pipelines database. For easy access
    Define PAPYRUS_ACCESS_STRING environmental variable
    Example: export PAPYRUS_ACCESS_STRING=mongodb+srv://[user]:[passwd]@testcluster.jm2x3.mongodb.net/test

    Parameters
    ----------
    query : dict
        Pymongo query (dictionary)
    collection: str
        Supported - 'pipelines' (default), 'runs', 'executions', samples', 'ppmseq'
    **kwargs: Any
        kwargs to pass to pymongo.find

    Returns
    -------
    list
        List of documents
    """
    disable_papyrus_access, collections = set_papyrus_access()
    if disable_papyrus_access:
        raise ValueError("Database access not available through PAPYRUS_ACCESS_STRING")
    collection_values = [x.value for x in Collections]
    if collection not in collection_values:
        raise ValueError(
            f"Collection \"{collection}\" not supported, supported collections are:\n{', '.join(collection_values)}"
        )
    return list(collections[Collections(collection)].find(query, **kwargs))


DEFAULT_METRICS_TO_REPORT = [
    "AlignmentSummaryMetrics",
    "Contamination",
    "DuplicationMetrics",
    "GcBiasDetailMetrics",
    "GcBiasSummaryMetrics",
    "QualityYieldMetrics",
    "RawWgsMetrics",
    "WgsMetrics",
    "stats_coverage",
    "short_report_/all_data",
    "short_report_/all_data_gt",
]

SV_METRICS_TO_REPORT = [
    "short_report_/fp_counts_per_length_and_type",
    "short_report_/length_by_type_counts",
    "short_report_/length_counts",
    "short_report_/parameters",
    "short_report_/recall_per_length_and_type",
    "short_report_/recall_per_type",
    "short_report_/type_counts",
]


def metrics2df(doc: dict, metrics_to_report: list | None = None) -> pd.DataFrame:
    """Converts metrics document to pandas dataframe

    Parameters
    ----------
    doc: dict
        Single document from mongodb (dictionary, output of query_database)
    metrics_to_report: list, optional
        which metrics should be reported (default {", ".join(DEFAULT_METRICS_TO_REPORT)})

    Returns
    -------
    pd.DataFrame
        1xn dataframe with two level index on columns: (metric_type, metric_name)
    """

    if metrics_to_report is None:
        metrics_to_report = DEFAULT_METRICS_TO_REPORT

    if "workflowEntity" not in doc["metadata"]:  # omics documents
        metadata = pd.DataFrame((pd.DataFrame(_cleanup_metadata(doc["metadata"]))).loc["submission"]).T

    else:
        metadata = pd.DataFrame(
            (pd.DataFrame(_cleanup_metadata(doc["metadata"])))
            .query('(workflowEntity=="sample") | (workflowEntity=="Sample") | (workflowEntity=="Unknown")')
            .loc["entityType"]
        ).T
    metadata.index = pd.Index([0])
    metadata = pd.concat({"metadata": metadata}, axis=1)
    result = [
        (x, pd.read_json(io.StringIO(json.dumps(doc["metrics"][x])), orient="table"))
        for x in doc["metrics"]
        if x in metrics_to_report
    ]
    result = [x for x in result if x[1].shape[0] == 1]
    result_df = pd.concat((metadata, pd.concat(dict(result), axis=1)), axis=1).set_index(("metadata", "workflowId"))
    result_df.index = result_df.index.rename("workflowId")
    return result_df


def inputs2df(doc: dict) -> pd.DataFrame:
    """Returns a dataframe of inputs and outputs metadata

    Parameters
    ----------
    doc: dict
        Single document from mongoDB

    Returns
    -------
    pd.DataFrame
        Dataframe of inputs and outputs combined (single row)
    """
    if "workflowEntity" not in doc["metadata"]:  # omics documents
        metadata = pd.DataFrame(_cleanup_metadata(doc["metadata"])).loc["submission"]
    else:
        metadata = (
            pd.DataFrame(_cleanup_metadata(doc["metadata"]))
            .query('(workflowEntity=="sample") | ' + '(workflowEntity=="Sample") | ' + '(workflowEntity=="Unknown")')
            .loc["entityType"]
        )
    inputs = pd.Series(doc["inputs"])
    outputs = pd.Series(doc["outputs"])
    return pd.DataFrame(pd.concat((metadata, inputs, outputs))).T.set_index("workflowId")


def nexus_metrics_to_df(input_dict: dict) -> pd.DataFrame:
    """Returns a dataframe of nexus metrics from run/execution/sample documentation

    Parameters
    ----------
    input_dict: dict
        Dictionary from mongoDB

    Returns
    -------
    pd.DataFrame
        Dataframe of parameters
    """
    s = pd.Series(input_dict).drop("_id")
    values = s.to_numpy()
    index = pd.MultiIndex.from_tuples([x.split("_", maxsplit=1) for x in s.index])
    name = s["metadata_sequencingRunId"]
    return pd.DataFrame(columns=[name], index=index, data=values).T


def _cleanup_metadata(input_dict: dict) -> dict:
    """Cleans up metadata - removes empty lists

    Parameters
    ----------
    input_dict: dict
        Input metadata dict

    Returns
    -------
    dict
        Same dictionary without values of type lists (confuses dataframe conversion)
    """
    for k in list(input_dict.keys()):
        if isinstance(input_dict[k], list):
            del input_dict[k]
    return input_dict
