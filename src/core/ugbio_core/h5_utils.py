import json
import re

import h5py
import pandas as pd
from pandas import DataFrame
from ugbio_core.logger import logger


def get_h5_keys(h5_filename: str):
    with pd.HDFStore(h5_filename, "r") as store:
        keys = store.keys()
        return keys


def should_skip_h5_key(key: str, ignored_h5_key_substring: str):
    if ignored_h5_key_substring is None:
        return None
    return ignored_h5_key_substring in key


def preprocess_h5_key(key: str):
    result = key
    if result[0] == "/":
        result = result[1:]
    return result


def preprocess_columns(dataframe):
    """Handle multiIndex/ hierarchical .h5 - concatenate the columns for using it as single string in JSON."""

    def flatten_multi_index(col, separator):
        flat = separator.join(col)
        flat = re.sub(f"{separator}$", "", flat)
        return flat

    if hasattr(dataframe, "columns"):
        if isinstance(dataframe.columns, pd.core.indexes.multi.MultiIndex):
            dataframe.columns = [flatten_multi_index(col, "___") for col in dataframe.columns.to_numpy()]


def convert_h5_to_json(
    input_h5_filename: str,
    root_element: str,
    ignored_h5_key_substring: str = None,
    output_json: str = None,
):
    """Convert an .h5 metrics file to .json with control over the root element and the processing

    Parameters
    ----------
    input_h5_filename: str
        Input h5 file name

    root_element: str
        Root element of the returned json

    ignored_h5_key_substring: str, optional
        A way to filter some of the keys using substring match, if None (default) none are filtered

    output_json : str, optional
        Output json file name to create if not None (default)

    Returns
    -------
    str
        The result json string includes the schema (the types) of the metrics as well as the metrics themselves.

    """

    new_json_dict = {root_element: {}}
    h5_keys = get_h5_keys(input_h5_filename)
    for h5_key in h5_keys:
        if should_skip_h5_key(h5_key, ignored_h5_key_substring):
            logger.warning("Skipping: %s", h5_key)
            continue
        logger.info("Processing: %s", h5_key)
        data_frame = read_hdf(input_h5_filename, h5_key)
        preprocess_columns(data_frame)
        data_frame_to_json = data_frame.to_json(orient="table")
        json_dict = json.loads(data_frame_to_json)
        new_json_dict[root_element][preprocess_h5_key(h5_key)] = json_dict

    if output_json:
        with open(output_json, "w", encoding="utf-8") as outfile:
            json.dump(new_json_dict, outfile, indent=4)
    json_string = json.dumps(new_json_dict, indent=4)
    return json_string


def read_hdf(  # noqa: C901 #TODO: refactor. too complex
    file_name: str,
    key: str = "all",
    skip_keys: list[str] | None = None,
    columns_subset: list[str] | None = None,
) -> DataFrame:
    """
    Read data-frame or data-frames from an h5 file

    Parameters
    ----------
    file_name: str
        path of local file
    key: str
        hdf key to the data-frame.
        Special keys:
        1. all - read all data-frames from the file and concat them
        2. all_human_chrs - read chr1, ..., chr22, chrX keys, and concat them
        3. all_somatic_chrs - chr1, ..., chr22
    skip_keys: Iterable[str]
        collection of keys to skip from reading the H5 (e.g. concordance, input_args ... )
    columns_subset: list[str], optional
        select a subset of columns

    Returns
    -------
    data-frame or concat data-frame read from the h5 file according to key
    """
    if skip_keys is None:
        skip_keys = []
    if key == "all":
        with h5py.File(file_name, "r") as h5_file:
            keys = list(h5_file.keys())
        for k in skip_keys:
            if k in keys:
                keys.remove(k)
        dfs = []
        for k in keys:
            tmpdf: DataFrame = DataFrame(pd.read_hdf(file_name, key=k))
            if columns_subset is not None:
                tmpdf = tmpdf[[x for x in columns_subset if x in tmpdf.columns]]
            if tmpdf.shape[0] > 0:
                dfs.append(tmpdf)
        return pd.concat(dfs)
    if key == "all_human_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=f"chr{x}")) for x in list(range(1, 23)) + ["X", "Y"]]
        return pd.concat(dfs)
    if key == "all_hg19_human_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=x)) for x in list(range(1, 23)) + ["X", "Y"]]
        return pd.concat(dfs)
    if key == "all_somatic_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=f"chr{x}")) for x in list(range(1, 23))]
        return pd.concat(dfs)
    # If not one of the special keys:
    return DataFrame(pd.read_hdf(file_name, key=key))
