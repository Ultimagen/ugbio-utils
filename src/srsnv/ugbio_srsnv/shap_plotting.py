#!/env/python
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Standalone SHAP plotting utilities for XGBoost models trained with cross-validation
# CHANGELOG in reverse chronological order

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib import cm

# Constants for plot formatting and colorbar display
MAX_LABEL_LENGTH_FOR_HORIZONTAL = 5  # Maximum character length before rotating colorbar labels


class SHAPPlotter:
    """
    Standalone SHAP plotting utilities for XGBoost models trained with cross-validation.

    This class provides functionality to calculate and plot SHAP values for XGBoost models
    without requiring the full SRSNVReport infrastructure. It's designed to work with
    cross-validation trained models and supports plotting on data subsets.

    The class assumes:
    - Models were trained using cross-validation with XGBoost's "objective": "multi:softprob"
      which returns 3D SHAP arrays with shape (n_samples, n_classes, n_features+1)
    - Data contains a fold_id column indicating which fold each sample belongs to
    - fold_id can be NaN for samples not used in training (test data) - these can use any model
    - Both categorical and numerical features are present
    - Features information is provided either in srsnv_metadata format or as separate lists/dicts

    Categorical Feature Handling:
    - Categorical features are assumed to be correctly encoded in the input data
    - The srsnv_metadata.json file contains the categorical encodings with proper value mappings
    - XGBoost models are trained with enable_categorical=True using these pre-encoded features
    - For SHAP visualization, categorical features are converted to integers purely for display
      purposes so that SHAP's beeswarm plots can apply color gradients (otherwise they remain grey)
    - This integer conversion is NOT for model inference - the original categorical encodings
      from metadata are preserved and used correctly by the trained XGBoost models

    Note:
    - TODO: Add support for standard binary classification models that return 2D SHAP arrays
    """

    def __init__(  # noqa: PLR0913
        self,
        models: list[xgb.XGBClassifier],
        data: pd.DataFrame,
        fold_id_col: str = "fold_id",
        features_metadata: list[dict] = None,
        feature_names: list[str] = None,
        categorical_features_dict: dict[str, list] = None,
        label_col: str = "label",
        random_state: int = 42,
    ):
        """
        Initialize the SHAPPlotter.

        Parameters
        ----------
        models : list[xgb.XGBClassifier]
            List of trained XGBoost models, one for each CV fold.
        data : pd.DataFrame
            DataFrame containing features, labels, and fold assignments.
            Must contain the fold_id_col and label_col columns.
        fold_id_col : str, default "fold_id"
            Name of the column containing fold assignments. Values should be integers
            from 0 to len(models)-1 for training data, or NaN for test data.
            Test data (NaN fold_id) can be used with any model.
        features_metadata : list[dict], optional
            List of feature metadata dictionaries in srsnv_metadata format. Each dict
            should have 'name', 'type', and optionally 'values' keys. If provided,
            this takes precedence over feature_names and categorical_features_dict.
        feature_names : list[str], optional
            List of feature names in the order they were used for training.
            Only used if features_metadata is not provided.
        categorical_features_dict : dict[str, list], optional
            Dictionary mapping categorical feature names to their possible values.
            Only used if features_metadata is not provided.
        label_col : str, default "label"
            Name of the column containing the target labels.
        random_state : int, default 42
            Random state for reproducible sampling.

        Raises
        ------
        ValueError
            If the number of models doesn't match the number of unique fold IDs,
            if required columns are missing from the data, or if feature information
            is not provided in any valid format.
        """
        self.models = models
        self.data = data
        self.fold_id_col = fold_id_col
        self.label_col = label_col
        self.rng = np.random.default_rng(random_state)

        # Validate inputs
        if fold_id_col not in data.columns:
            raise ValueError(f"fold_id_col '{fold_id_col}' not found in data columns")
        if label_col not in data.columns:
            raise ValueError(f"label_col '{label_col}' not found in data columns")

        # Process features information
        if features_metadata is not None:
            # Extract features from srsnv_metadata format
            self.feature_names, self.categorical_features_dict = self._parse_features_metadata(features_metadata)
        elif feature_names is not None:
            # Use provided feature names and categorical dict
            self.feature_names = feature_names
            self.categorical_features_dict = categorical_features_dict or {}
        elif hasattr(models[0], "feature_names_in_"):
            # Try to get feature names from the first model
            self.feature_names = list(models[0].feature_names_in_)
            self.categorical_features_dict = categorical_features_dict or {}
        else:
            raise ValueError(
                "Feature information must be provided either through features_metadata, "
                "feature_names, or the model must have feature_names_in_ attribute"
            )

        # Check that number of models matches number of folds
        fold_values = data[fold_id_col].dropna().unique()
        if len(models) != len(fold_values):
            raise ValueError(f"Number of models ({len(models)}) doesn't match number of folds ({len(fold_values)})")

    def _parse_features_metadata(self, features_metadata: list[dict]) -> tuple[list[str], dict[str, list]]:
        """
        Parse features metadata from srsnv_metadata format.

        This method extracts categorical feature encodings from the metadata. The categorical
        features in the srsnv_metadata.json file already contain the correct encodings that
        were used during model training with XGBoost's enable_categorical=True.

        Example from srsnv_metadata.json:
        {
          "name": "REF",
          "type": "c",
          "values": {"A": 0, "C": 1, "G": 2, "T": 3}
        }

        These encodings ensure that:
        1. XGBoost models were trained with consistent categorical representations
        2. The same encodings are used for SHAP value calculation
        3. The integer conversion for visualization preserves the correct order/mapping

        Parameters
        ----------
        features_metadata : list[dict]
            List of feature metadata dictionaries from srsnv_metadata.json.
            Each dict should have 'name', 'type', and optionally 'values' keys.
            For categorical features (type='c'), 'values' contains the string-to-integer
            mapping that was used during model training.

        Returns
        -------
        tuple[list[str], dict[str, list]]
            Feature names list and categorical features dictionary mapping feature names
            to their ordered categorical values (sorted by the integer codes from metadata).
        """
        feature_names = []
        categorical_features_dict = {}

        for feature_info in features_metadata:
            name = feature_info["name"]
            feature_names.append(name)

            # Extract categorical feature encodings from metadata
            if feature_info.get("type") == "c" and "values" in feature_info:
                # Sort by the numeric values to preserve the encoding order used in training
                values_dict = feature_info["values"]
                sorted_values = sorted(values_dict.items(), key=lambda x: x[1])
                categorical_features_dict[name] = [item[0] for item in sorted_values]

        return feature_names, categorical_features_dict

    def plot_feature_importance(  # noqa: PLR0913
        self,
        data_subset: pd.DataFrame = None,
        fold_id: int = 0,
        n_sample: int = 10_000,
        n_features: int = 15,
        xlims: tuple[float, float] = None,
        output_filename: str = None,
        figsize: tuple[int, int] = (20, 10),
        *,
        shap_values: np.ndarray = None,
        x_sample: pd.DataFrame = None,
    ) -> tuple[np.ndarray, pd.DataFrame, plt.Figure]:
        """
        Plot SHAP feature importance using a bar plot.

        This method calculates SHAP values for a sample of data and creates a bar plot
        showing the mean absolute SHAP value for each feature, indicating their
        relative importance to the model.

        Parameters
        ----------
        data_subset : pd.DataFrame, optional
            Subset of data to use for SHAP calculation. If None, uses data from
            the specified fold_id. This allows plotting SHAP values for specific
            subsets of interest (e.g., specific mutation types, quality ranges, etc.).
        fold_id : int, default 0
            Which fold's model to use for SHAP calculation. Only used if data_subset
            is None.
        n_sample : int, default 10_000
            Number of samples to use for SHAP calculation. Larger values give more
            stable results but take longer to compute.
        n_features : int, default 15
            Number of top features to display in the plot.
        xlims : tuple[float, float], optional
            X-axis limits for the plot. If None, automatically determined.
        output_filename : str, optional
            Path to save the plot. If None, plot is displayed but not saved.
        figsize : tuple[int, int], default (20, 10)
            Figure size as (width, height) in inches.
        shap_values : np.ndarray, optional
            Pre-calculated SHAP values with shape (n_samples, n_classes, n_features+1).
            If provided, skips SHAP calculation. Must be provided together with x_sample.
        x_sample : pd.DataFrame, optional
            Sampled features corresponding to the shap_values. Must be provided
            together with shap_values.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame, plt.Figure]
            A tuple containing:
            - shap_values: Array of SHAP values with shape (n_samples, n_classes, n_features+1)
            - X_sample: DataFrame of the sampled data used for SHAP calculation
            - fig: The matplotlib figure object

        Notes
        -----
        - For binary classification, uses the difference between class 1 and class 0 SHAP values
        - Categorical features are automatically converted to integer representation for display
        - Features are ranked by mean absolute SHAP value
        """
        # Use pre-calculated SHAP values if provided, otherwise calculate them
        if shap_values is not None and x_sample is not None:
            if shap_values.shape[0] != len(x_sample):
                raise ValueError(
                    f"Shape mismatch: shap_values has {shap_values.shape[0]} samples "
                    f"but x_sample has {len(x_sample)} samples"
                )
        elif shap_values is not None or x_sample is not None:
            raise ValueError("Both shap_values and x_sample must be provided together, or neither")
        else:
            # Calculate SHAP values
            shap_values, x_sample, _ = self.sample_and_calculate_shap_values(
                data_subset=data_subset, fold_id=fold_id, n_sample=n_sample
            )

        # Prepare data for plotting
        # x_sample_plot = self._prepare_data_for_plotting(x_sample)
        x_sample_plot = x_sample.copy()

        # Create SHAP explanation object
        base_values_plot = shap_values[0, 1, -1] - shap_values[0, 0, -1]
        shap_values_plot = shap_values[:, 1, :-1] - shap_values[:, 0, :-1]
        explanation = shap.Explanation(
            values=shap_values_plot,
            base_values=base_values_plot,
            feature_names=x_sample_plot.columns,
            data=x_sample_plot,
        )

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        plt.sca(ax)
        shap.plots.bar(
            explanation,
            max_display=n_features,
            show=False,
        )
        ax.set_xlabel("")
        if xlims is not None:
            ax.set_xlim(xlims)
        ticklabels = ax.get_ymajorticklabels()
        ax.set_yticklabels(ticklabels)
        ax.grid(visible=True, axis="x", linestyle=":", linewidth=1)

        # Save or show the plot
        if output_filename:
            fig.savefig(output_filename, bbox_inches="tight", dpi=300)

        return shap_values, x_sample, fig

    def plot_beeswarm(  # noqa: PLR0913 C901 PLR0912 PLR0915
        self,
        data_subset: pd.DataFrame = None,
        fold_id: int = 0,
        n_sample: int = 10_000,
        n_features: int = 10,
        nplot_sample: int = None,
        cmap: str = "brg",
        xlims: tuple[float, float] = None,
        output_filename: str = None,
        figsize: tuple[int, int] = (20, 10),
        *,
        shap_values: np.ndarray = None,
        x_sample: pd.DataFrame = None,
        show_colorbar: bool = True,
        include_range: bool = True,
        show_other_features: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame, plt.Figure]:
        """
        Plot SHAP beeswarm plot showing individual SHAP values and feature interactions.

        This method creates a beeswarm plot where each dot represents a single sample's
        SHAP value for a feature. The x-axis shows the SHAP value (impact on model output),
        the y-axis shows the features, and the color represents the feature value.

        Parameters
        ----------
        data_subset : pd.DataFrame, optional
            Subset of data to use for SHAP calculation. If None, uses data from
            the specified fold_id.
        fold_id : int, default 0
            Which fold's model to use for SHAP calculation. Only used if data_subset
            is None.
        n_sample : int, default 10_000
            Number of samples to use for SHAP calculation.
        n_features : int, default 10
            Number of top features to display in the plot.
        nplot_sample : int, optional
            Number of samples to actually plot (subset of n_sample). If None,
            plots all samples. Useful for reducing visual clutter in dense plots.
        cmap : str, default "brg"
            Colormap name for the plot. Used to color points by feature value.
        xlims : tuple[float, float], optional
            X-axis limits for the plot. If None, automatically determined.
        output_filename : str, optional
            Path to save the plot. If None, plot is displayed but not saved.
        figsize : tuple[int, int], default (20, 10)
            Figure size as (width, height) in inches.
        shap_values : np.ndarray, optional
            Pre-calculated SHAP values with shape (n_samples, n_classes, n_features+1).
            If provided, skips SHAP calculation. Must be provided together with x_sample.
        x_sample : pd.DataFrame, optional
            Sampled features corresponding to the shap_values. Must be provided
            together with shap_values.
        show_colorbar : bool, default True
            Whether to show colorbars for categorical features. Colorbars help
            interpret the meaning of colors for categorical features.
        include_range : bool, default True
            Whether to include all categories between the first and last present category
            when filtering categorical features. If True, avoids color normalization
            mismatches by ensuring continuous range of category indices. If False,
            includes only actually present categories. NaN values (mapped to -1 by XGBoost)
            are always included when present and labeled as "NA" in the colorbar.
        show_other_features : bool, default True
            Whether to add an "Other features" row at the bottom of the beeswarm plot.
            This row shows the sum of SHAP values for all features not in the top n_features,
            displayed in grey without coloring. If all features are being plotted
            (n_features >= total features), this row is not added regardless of this parameter.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame, plt.Figure]
            A tuple containing:
            - shap_values: Array of SHAP values with shape (n_samples, n_classes, n_features+1)
            - X_sample: DataFrame of the sampled data used for SHAP calculation
            - fig: The matplotlib figure object

        Notes
        -----
        - Categorical features are grouped by their category values and displayed with
          separate colorbars when show_colorbar=True
        - Features are ranked by mean absolute SHAP value
        - The plot shows both the magnitude (x-position) and direction (color intensity)
          of feature impacts
        - When show_other_features=True, an additional grey row is added showing the sum
          of SHAP values for features not in the top n_features
        """
        # Validate and calculate SHAP values
        if shap_values is not None and x_sample is not None:
            # Use pre-calculated values - validate consistency
            if len(shap_values) != len(x_sample):
                raise ValueError(
                    f"Length mismatch: shap_values has {len(shap_values)} samples "
                    f"but x_sample has {len(x_sample)} samples"
                )
        elif shap_values is not None or x_sample is not None:
            raise ValueError("Both shap_values and x_sample must be provided together, or both must be None")
        else:
            # Calculate SHAP values
            shap_values, x_sample, _ = self.sample_and_calculate_shap_values(
                data_subset=data_subset, fold_id=fold_id, n_sample=n_sample
            )

        # Get filtered categorical features based on actual data present
        filtered_categorical_features = self._get_present_categorical_features(x_sample, include_range=include_range)

        # Prepare data for plotting and group categorical features
        grouped_features = self._group_categorical_features(
            categorical_features_dict={col: val[1] for col, val in filtered_categorical_features.items()}
        )
        x_sample_plot = self._prepare_data_for_plotting(
            x_sample, filtered_categorical_features=filtered_categorical_features
        )

        # Get top features by SHAP importance
        total_features = shap_values.shape[2] - 1  # Exclude bias term
        top_features = np.abs(shap_values[:, 1, :-1] - shap_values[:, 0, :-1]).mean(axis=0).argsort()[::-1][:n_features]

        # Filter grouped_features to only include groups represented in top features
        top_feature_names = [x_sample_plot.columns[i] for i in top_features]

        filtered_grouped_features = {}
        for group_key, feature_list in grouped_features.items():
            # Check if any feature in this group is in the top features
            if any(feature in top_feature_names for feature in feature_list):
                filtered_grouped_features[group_key] = feature_list

        # Rename features to include group information
        x_sample_plot = x_sample_plot.rename(
            columns={
                feature: f"{feature} [{group_idx}]"
                for group_idx, features in enumerate(filtered_grouped_features.values(), start=1)
                for feature in features
            }
        )

        # Determine if we should add "other features" row
        other_features = np.setdiff1d(np.arange(total_features), top_features)
        add_other_features = show_other_features and len(other_features) > 0

        # Sample for plotting if needed
        inds_for_plot = np.arange(x_sample_plot.shape[0])
        if nplot_sample is not None and nplot_sample < x_sample_plot.shape[0]:
            inds_for_plot = self.rng.choice(inds_for_plot, size=nplot_sample, replace=False)

        # Prepare SHAP values and data
        base_values_plot = shap_values[0, 1, -1] - shap_values[0, 0, -1]

        if add_other_features:
            # SHAP values for top features
            shap_top = (
                shap_values[inds_for_plot.reshape((-1, 1)), 1, top_features]
                - shap_values[inds_for_plot.reshape((-1, 1)), 0, top_features]
            )
            # SHAP values for "other features" (sum across other features)
            shap_other = (
                shap_values[inds_for_plot, 1, :][:, other_features]
                - shap_values[inds_for_plot, 0, :][:, other_features]
            ).sum(axis=1, keepdims=True)
            # Concatenate
            shap_values_plot = np.hstack([shap_top, shap_other])

            # Data for top features + dummy data for "Other features" (categorical 0 for grey coloring)
            other_data = pd.DataFrame({"Other features": [0] * len(inds_for_plot)}, index=inds_for_plot).astype(
                "category"
            )
            data_for_plot = pd.concat(
                [
                    x_sample_plot.iloc[inds_for_plot, top_features].reset_index(drop=True),
                    other_data.reset_index(drop=True),
                ],
                axis=1,
            )

            # Feature names including "other features"
            feature_names = list(x_sample_plot.columns[top_features]) + [f"Other features ({len(other_features)})"]

            # Calculate mean absolute SHAP values for ordering
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
            # Force "Other features" (last column) to appear at the bottom
            order = np.argsort(np.concatenate([mean_abs_shap[:-1], [mean_abs_shap[-1] - 1e6]]))[::-1]
            # Move "Other features" to the end
            order = np.concatenate([order[order != shap_values_plot.shape[1] - 1], [shap_values_plot.shape[1] - 1]])
        else:
            # No other features - standard processing
            shap_values_plot = (
                shap_values[inds_for_plot.reshape((-1, 1)), 1, top_features]
                - shap_values[inds_for_plot.reshape((-1, 1)), 0, top_features]
            )
            data_for_plot = x_sample_plot.iloc[inds_for_plot, top_features]
            feature_names = list(x_sample_plot.columns[top_features])

            # Order by decreasing mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
            order = np.argsort(mean_abs_shap)[::-1]

        explanation = shap.Explanation(
            values=shap_values_plot,
            base_values=base_values_plot,
            feature_names=feature_names,
            data=data_for_plot,
        )

        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize)
        if xlims is None:
            xlims = [
                np.floor(shap_values_plot.min() - base_values_plot),
                np.ceil(shap_values_plot.max() - base_values_plot),
            ]
        plt.sca(ax)
        shap_plot_kwargs = {
            "color": plt.get_cmap(cmap),
            "max_display": 30,
            "alpha": 0.2,
            "show": False,
            "plot_size": 0.4,
            "color_bar": False,
        }
        if order is not None:
            shap_plot_kwargs["order"] = order

        shap.plots.beeswarm(explanation, **shap_plot_kwargs)
        ax.set_xlabel("SHAP value", fontsize=12)
        ax.set_xlim(xlims)
        fig.tight_layout()

        # Add colorbars if requested
        if show_colorbar and grouped_features:
            self._add_colorbars_to_plot(fig, filtered_grouped_features, cmap)

        # Save or show the plot
        if output_filename:
            fig.savefig(output_filename, bbox_inches="tight", dpi=300)

        return shap_values, x_sample, fig

    def sample_and_calculate_shap_values(
        self,
        data_subset: pd.DataFrame = None,
        fold_id: int = 0,
        n_sample: int = 10_000,
    ) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
        """
        Calculate SHAP values for a given dataset and model.

        This is the core method that computes SHAP values using XGBoost's built-in
        SHAP calculation. It can be used independently or called by the plotting methods.

        The method assumes that categorical features are already correctly encoded
        according to the mappings stored in srsnv_metadata.json. XGBoost models
        were trained with enable_categorical=True using these same encodings, ensuring
        consistency between training and SHAP calculation phases.

        Parameters
        ----------
        data_subset : pd.DataFrame, optional
            Subset of data to use for SHAP calculation. If None, uses data from
            the specified fold_id plus any test data (NaN fold_id). Data should
            contain categorical features already encoded as they were during training.
            NOTE: If data_subset is provided, it overrides the fold_id logic, which
            might result in using the model with data used in its training.
        fold_id : int, default 0
            Which fold's model to use for SHAP calculation. Test data (NaN fold_id)
            can be processed by any model since it wasn't used in training.
        n_sample : int, default 10_000
            Number of samples to use for SHAP calculation. Samples are randomly
            selected from the provided data.

        Returns
        -------
        tuple[np.ndarray, pd.DataFrame, pd.Series]
            A tuple containing:
            - shap_values: Array of SHAP values with shape (n_samples, n_classes, n_features+1)
                          The last column contains the base values.
            - X_sample: DataFrame of the sampled features used for SHAP calculation
                       (contains categorical features with their original encodings)
            - y_sample: Series of the corresponding labels for the sampled data

        Notes
        -----
        - Uses XGBoost's pred_contribs=True for efficient SHAP calculation
        - Categorical features are used directly with their metadata-based encodings
        - Handles both numerical and categorical features automatically
        - Ensures reproducible sampling using the class's random_state
        - Test data (NaN fold_id) can be processed by any model
        - Assumes XGBoost models trained with "objective": "multi:softprob" which returns
          3D SHAP arrays with shape (n_samples, n_classes, n_features+1)
        - TODO: Add support for standard binary classification models that return 2D arrays
        """
        # Determine which data to use
        if data_subset is not None:
            data_to_use = data_subset
        else:
            # Use data from the specified fold, but if fold_id=NaN in data, that's test data
            # For test data (NaN fold_id), we can use any model since it wasn't used in training
            fold_mask = self.data[self.fold_id_col] == fold_id
            test_mask = self.data[self.fold_id_col].isna()
            data_to_use = self.data[fold_mask | test_mask]

        if len(data_to_use) == 0:
            raise ValueError(f"No data available for fold {fold_id}")

        # Get the model for this fold
        if fold_id < 0 or fold_id >= len(self.models):
            raise ValueError(f"fold_id {fold_id} out of range [0, {len(self.models)-1}]")
        model = self.models[fold_id]

        # Sample the data (using all features from the model)
        x_sample, y_sample = self._sample_data(data_to_use, n_sample, self.feature_names)

        # Calculate SHAP values using XGBoost's built-in method
        if not hasattr(model, "best_ntree_limit"):
            try:
                model.best_ntree_limit = model.best_iteration + 1
            except AttributeError:
                # best_iteration is only available with early stopping
                # Use all trees if early stopping wasn't used
                model.best_ntree_limit = model.n_estimators

        # Create DMatrix with categorical features properly encoded
        # x_sample contains categorical features with their original metadata-based encodings
        # enable_categorical=True tells XGBoost to handle them as categorical (not ordinal)
        x_sample_dm = xgb.DMatrix(data=x_sample, label=y_sample, enable_categorical=True)
        shap_values = model.get_booster().predict(
            x_sample_dm, pred_contribs=True, iteration_range=(0, model.best_ntree_limit)
        )

        return shap_values, x_sample, y_sample

    def get_feature_importance_summary(
        self,
        data_subset: pd.DataFrame = None,
        fold_id: int = 0,
        n_sample: int = 10_000,
    ) -> pd.Series:
        """
        Get a summary of feature importance based on mean absolute SHAP values.

        Parameters
        ----------
        data_subset : pd.DataFrame, optional
            Subset of data to use for SHAP calculation.
        fold_id : int, default 0
            Which fold's model to use for SHAP calculation.
        n_sample : int, default 10_000
            Number of samples to use for SHAP calculation.

        Returns
        -------
        pd.Series
            Series with feature names as index and mean absolute SHAP values as values,
            sorted in descending order of importance.
        """
        # Calculate SHAP values
        shap_values, x_sample, _ = self.sample_and_calculate_shap_values(
            data_subset=data_subset, fold_id=fold_id, n_sample=n_sample
        )

        # Calculate mean absolute SHAP values for feature importance
        # For binary classification, use difference between class 1 and class 0
        mean_abs_shap_scores = pd.Series(
            np.abs(shap_values[:, 1, :-1] - shap_values[:, 0, :-1]).mean(axis=0), index=x_sample.columns
        ).sort_values(ascending=False)

        return mean_abs_shap_scores

    def _get_present_categorical_features(  # noqa: C901
        self, x_sample: pd.DataFrame, nan_value: str = "NA", *, include_range: bool = True
    ) -> dict[str, list[str]]:
        """
        Create filtered categorical features dict with only present categories.

        This method analyzes the actual data to determine which categorical values
        are present and creates a filtered version of the categorical features dictionary.

        Parameters
        ----------
        x_sample : pd.DataFrame
            Sample data to analyze for present categories.
        nan_value : str, default "NA"
            Value to use for NaN entries in categorical features. This will be included
            in the filtered categories if NaN values are present in the data.
        include_range : bool, default True
            If True, include all categories between the first and last present category
            to ensure consistent color mapping. If False, include only actually present categories.

        Returns
        -------
        dict[str, list[str]]
            Filtered categorical features dictionary containing only relevant categories
            in their original metadata ordering. NaN values are represented as 'NA'.

        Notes
        -----
        - Preserves original category ordering from metadata
        - NaN values are included as 'NA' when present in the data
        - When include_range=True, fills gaps between min and max present categories
        """
        filtered_categorical_features = {}

        for feature_name, original_categories in self.categorical_features_dict.items():
            if feature_name not in x_sample.columns:
                continue

            # Get unique values, including NaN
            present_values = x_sample[feature_name].unique()

            # Find which category indices are present
            present_indices = []
            has_nan = False

            for val in present_values:
                if pd.isna(val):
                    has_nan = True
                elif str(val) in original_categories:
                    present_indices.append(original_categories.index(str(val)))

            if not present_indices and not has_nan:
                continue

            # Determine which indices to include
            if include_range and present_indices:
                # Include all categories between min and max present indices
                min_idx, max_idx = min(present_indices), max(present_indices)
                category_indices = list(range(min_idx, max_idx + 1))
            else:
                # Include only actually present indices
                category_indices = sorted(present_indices)

            # Build filtered categories list
            filtered_categories = []
            if has_nan:
                filtered_categories.append(nan_value)
            for idx in category_indices:
                filtered_categories.append(original_categories[idx])

            if filtered_categories:
                filtered_categorical_features[feature_name] = (has_nan, filtered_categories)

        return filtered_categorical_features

    def _prepare_data_for_plotting(
        self, x_sample: pd.DataFrame, filtered_categorical_features: dict[str, tuple[bool, list[str]]] = None
    ) -> pd.DataFrame:
        """
        Prepare data for SHAP plotting by converting categorical features to integer codes.

        This method converts categorical features to integer representations PURELY FOR VISUALIZATION
        purposes. The categorical features in the input data are already correctly encoded according
        to the mappings stored in srsnv_metadata.json and used by XGBoost during training.

        The integer conversion here serves only to enable SHAP's beeswarm plot to apply color
        gradients to categorical features. Without this conversion, categorical features would
        appear as grey dots in the beeswarm plot because SHAP cannot apply colors to non-numeric data.

        IMPORTANT: This conversion is NOT for model inference. The XGBoost models continue to use
        the original categorical encodings from the metadata. This method only facilitates proper
        visualization of SHAP values with meaningful colors and colorbars.

        Parameters
        ----------
        x_sample : pd.DataFrame
            Sample data with original feature values (already properly encoded for XGBoost).
        filtered_categorical_features : dict[str, tuple[bool, list[str]]], optional
            Filtered categorical features dictionary (output of
            self._get_present_categorical_features), containing only the categories
            actually present in the data. If None, uses self.categorical_features_dict
            and assumes has_nan=False.

        Returns
        -------
        pd.DataFrame
            Data with categorical features converted to integer codes for visualization.
            NaN values are mapped to 0, and other categories to their indices in the
            filtered category list. This maintains color consistency in SHAP plots.

        Notes
        -----
        - Integer codes are used directly for color mapping (no normalization to [0, 1])
        - This ensures proper colorbar alignment regardless of which categories are present
        - The conversion preserves the order of categories as defined in the metadata
        - NaN values consistently map to index 0 across all categorical features
        """
        if filtered_categorical_features is None:
            filtered_categorical_features = {col: (False, vals) for col, vals in self.categorical_features_dict.items()}
        x_sample_display = x_sample.copy()

        # Convert categorical features to integer codes for SHAP visualization
        for col, (_has_nan, filtered_categories) in filtered_categorical_features.items():
            if col not in x_sample_display.columns:
                continue
            value_to_int_map = {cv: i for i, cv in enumerate(filtered_categories)}
            x_sample_display[col] = x_sample_display[col].astype(object).map(value_to_int_map)
            x_sample_display[col] = x_sample_display[col].fillna(0).astype(int)

        return x_sample_display

    def _group_categorical_features(
        self, categorical_features_dict: dict[str, list[str]] = None
    ) -> dict[tuple, list[str]]:
        """
        Group categorical features by their present category values for colorbar grouping.

        Features are grouped based on the actual set of categories present in the data (including NaN as 'NA'),
        not the full set of possible categories. This ensures that features with different present categories
        (including different NaN presence) are grouped separately for colorbar display.

        Parameters
        ----------
        categorical_features_dict : dict[str, list[str]], optional
            Categorical features dictionary to use for grouping. If None, uses
            self.categorical_features_dict. This should be the filtered dictionary
            that contains only actually present categories.

        Returns
        -------
        dict[tuple, list[str]]
            Dictionary mapping tuples of present category values to lists of feature names
            that have those categories. Features with different sets of present
            categories (including different NaN presence) will be in separate groups.

        Notes
        -----
        - Groups are based on the exact set of categories that will appear in the plot.
        - Features with NaN vs without NaN will be in different groups.
        - Features with different subsets of categories will be in different groups.

        TODO: shap's _beeswarm.py (with the functions that draw the beeswarm plot) does the
        following when plotting numerical features: it filters out the bottom and top 5%-iles
        of the data, and only then plots (so that outliers don't affect the feature value colors).
        This is a problem for us, because it could be that the first of last categories have less than
        5% of the data, and thus they will not be shown in the beeswarm plot at all. What is worse, this
        will not be reflected in the colorbars, which will show the full range of categories.
        We should implement a similar filtering here, so that the colorbars match the actual categories
        shown in the beeswarm plot.
        """
        if categorical_features_dict is None:
            categorical_features_dict = self.categorical_features_dict

        # Create a defaultdict to hold lists of feature names for each tuple of values
        grouped_features = defaultdict(list)

        for feature, values in categorical_features_dict.items():
            # Convert the list of values to a tuple (so it can be used as a dictionary key)
            # This now uses the FILTERED categories, ensuring proper grouping
            value_tuple = tuple(values)
            # Append the feature name to the list corresponding to this tuple of values
            grouped_features[value_tuple].append(feature)

        # Convert the defaultdict to a regular dict before returning
        return dict(grouped_features)

    def _add_colorbars_to_plot(
        self, fig: plt.Figure, grouped_features: dict[tuple, list[str]], cmap: str = "brg", **kwargs
    ) -> None:
        """
        Add colorbars to a SHAP beeswarm plot for categorical features.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure to add colorbars to.
        grouped_features : dict[tuple, list[str]]
            Grouped categorical features from _group_categorical_features.
        cmap : str, default "brg"
            Colormap name to use for the colorbars.
        **kwargs
            Additional arguments for colorbar positioning and styling.
        """
        total_width = kwargs.get("total_width", 0.3)
        vertical_padding = kwargs.get("vertical_padding", 0.2)
        start_y = kwargs.get("start_y", 0.9)
        rotated_padding_factor = kwargs.get("rotated_padding_factor", 8.0)

        group_labels = []
        for i, (cat_values, features) in enumerate(grouped_features.items()):
            n_cat = len(cat_values)
            cbar_length = total_width
            max_cat_val_length = max(len(f"{val}") for val in cat_values)
            rotation = 0 if max_cat_val_length <= MAX_LABEL_LENGTH_FOR_HORIZONTAL else 1
            ha = "center" if rotation == 0 else "right"

            group_label = f"[{i + 1}]"
            group_labels.append(f"{group_label}: {', '.join(features)}")

            # Calculate the rectangle for the colorbar axis
            cbar_ax = fig.add_axes([1.1, start_y - 0.03, cbar_length, 0.03])

            # Create the colorbar
            fig_cbar = fig.colorbar(
                cm.ScalarMappable(norm=None, cmap=plt.get_cmap(cmap, n_cat)), cax=cbar_ax, orientation="horizontal"
            )
            fig_cbar.set_label(group_label, fontsize=12)
            fig_cbar.set_ticks(np.arange(0, 1, 1 / n_cat) + 1 / (2 * n_cat))
            fig_cbar.set_ticklabels(cat_values, fontsize=12, rotation=rotation * 25, ha=ha)
            fig_cbar.outline.set_visible(False)

            # Update the start position for the next colorbar
            start_y -= vertical_padding + rotation / rotated_padding_factor

        # Add an extra colorbar for numerical features below the others
        cbar_num_ax = fig.add_axes([1.1, start_y - 0.03, total_width, 0.03])
        fig_num_cbar = fig.colorbar(
            cm.ScalarMappable(norm=None, cmap=plt.get_cmap(cmap)), cax=cbar_num_ax, orientation="horizontal"
        )
        fig_num_cbar.set_label("Numerical features", fontsize=12)
        fig_num_cbar.set_ticks([0, 1])
        fig_num_cbar.set_ticklabels(["Low value", "High value"], fontsize=12)
        fig_num_cbar.outline.set_visible(False)

    def _get_model_for_fold(self, fold_id: int) -> xgb.XGBClassifier:
        """
        Get the model for a specific fold.

        Parameters
        ----------
        fold_id : int
            The fold ID to get the model for.

        Returns
        -------
        xgb.XGBClassifier
            The XGBoost model for the specified fold.

        Raises
        ------
        ValueError
            If fold_id is out of range.
        """
        if fold_id < 0 or fold_id >= len(self.models):
            raise ValueError(f"fold_id {fold_id} out of range [0, {len(self.models)-1}]")
        return self.models[fold_id]

    def _sample_data(
        self, data: pd.DataFrame, n_sample: int, features: list[str] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Sample data for SHAP calculation.

        Parameters
        ----------
        data : pd.DataFrame
            Data to sample from.
        n_sample : int
            Number of samples to take.
        features : list[str], optional
            Features to include. If None, uses all model features.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Sampled features and corresponding labels.
        """
        if features is None:
            features = self.feature_names

        # Ensure we don't sample more than available
        n_sample = min(n_sample, len(data))

        # Sample the data
        if n_sample < len(data):
            sampled_indices = self.rng.choice(data.index, size=n_sample, replace=False)
            data_sampled = data.loc[sampled_indices]
        else:
            data_sampled = data

        x_sample = data_sampled[features]
        y_sample = data_sampled[self.label_col]

        return x_sample, y_sample

    def plot_both(  # noqa: PLR0913
        self,
        data_subset: pd.DataFrame = None,
        fold_id: int = 0,
        n_sample: int = 10_000,
        *,
        n_features_importance: int = 15,
        n_features_beeswarm: int = 10,
        nplot_sample: int = None,
        cmap: str = "brg",
        xlims_importance: tuple[float, float] = None,
        xlims_beeswarm: tuple[float, float] = None,
        output_filename_importance: str = None,
        output_filename_beeswarm: str = None,
        figsize_importance: tuple[int, int] = (20, 10),
        figsize_beeswarm: tuple[int, int] = (20, 10),
        show_colorbar: bool = True,
        show_other_features: bool = True,
    ) -> tuple[plt.Figure, plt.Figure, np.ndarray, pd.DataFrame]:
        """
        Generate both feature importance and beeswarm plots efficiently.

        This method calculates SHAP values once and generates both plots,
        avoiding redundant computation. This is much more efficient than
        calling plot_feature_importance() and plot_beeswarm() separately.

        Parameters
        ----------
        data_subset : pd.DataFrame, optional
            Subset of data to use for SHAP calculation. If None, uses data from
            the specified fold_id.
        fold_id : int, default 0
            Which fold's model to use for SHAP calculation. Only used if data_subset
            is None.
        n_sample : int, default 10_000
            Number of samples to use for SHAP calculation.
        n_features_importance : int, default 15
            Number of top features to display in the feature importance plot.
        n_features_beeswarm : int, default 10
            Number of top features to display in the beeswarm plot.
        nplot_sample : int, optional
            Number of samples to actually plot in beeswarm (subset of n_sample).
            If None, plots all samples.
        cmap : str, default "brg"
            Colormap name for the beeswarm plot.
        xlims_importance : tuple[float, float], optional
            X-axis limits for the feature importance plot.
        xlims_beeswarm : tuple[float, float], optional
            X-axis limits for the beeswarm plot.
        output_filename_importance : str, optional
            Path to save the feature importance plot.
        output_filename_beeswarm : str, optional
            Path to save the beeswarm plot.
        figsize_importance : tuple[int, int], default (20, 10)
            Figure size for the feature importance plot.
        figsize_beeswarm : tuple[int, int], default (20, 10)
            Figure size for the beeswarm plot.
        show_colorbar : bool, default True
            Whether to show colorbars in the beeswarm plot.
        show_other_features : bool, default True
            Whether to add an "Other features" row to the beeswarm plot showing
            the sum of SHAP values for features not in the top n_features_beeswarm.

        Returns
        -------
        tuple[plt.Figure, plt.Figure, np.ndarray, pd.DataFrame]
            A tuple containing:
            - fig_importance: Feature importance plot figure
            - fig_beeswarm: Beeswarm plot figure
            - shap_values: Array of SHAP values used for both plots
            - x_sample: DataFrame of sampled data used for both plots
        """
        # Calculate SHAP values once
        shap_values, x_sample, _ = self.sample_and_calculate_shap_values(
            data_subset=data_subset, fold_id=fold_id, n_sample=n_sample
        )

        # Generate feature importance plot using pre-calculated values
        _, _, fig_importance = self.plot_feature_importance(
            n_features=n_features_importance,
            xlims=xlims_importance,
            output_filename=output_filename_importance,
            figsize=figsize_importance,
            shap_values=shap_values,
            x_sample=x_sample,
        )

        # Generate beeswarm plot using pre-calculated values
        _, _, fig_beeswarm = self.plot_beeswarm(
            n_features=n_features_beeswarm,
            nplot_sample=nplot_sample,
            cmap=cmap,
            xlims=xlims_beeswarm,
            output_filename=output_filename_beeswarm,
            figsize=figsize_beeswarm,
            show_colorbar=show_colorbar,
            show_other_features=show_other_features,
            shap_values=shap_values,
            x_sample=x_sample,
        )

        return fig_importance, fig_beeswarm, shap_values, x_sample


if __name__ == "__main__":
    import json

    base_path = "/data/Runs/data/srsnv/debug/new_pipeline/test_data/out/"
    data_df = pd.read_parquet(base_path + "416119_L7402.featuremap_df.parquet")
    print(data_df.shape)

    with open(base_path + "416119_L7402.srsnv_metadata.json") as f:
        metadata = json.load(f)

    # Load XGBoost models from the paths in metadata
    model_paths = metadata["model_paths"]
    models = []

    for fold, model_path in model_paths.items():  # noqa: B007
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        models.append(model)

    num_CV_folds = len(models)  # noqa: N816

    print(f"Successfully loaded {num_CV_folds} XGBoost models")

    shap_plotter = SHAPPlotter(
        models,
        data_df,
        features_metadata=metadata["features"],
    )

    shap_values, x_sample, fig = shap_plotter.plot_both(
        n_sample=1_000,
    )
    plt.show()
    print("done")
