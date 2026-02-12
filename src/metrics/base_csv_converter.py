from halib import *
from abc import ABC, abstractmethod
from typing import List, Optional
from src.common import GlobalConst


class BaseCSVConverter(ABC):
    """Base class for converting CSV/DataFrame columns to target formats."""

    @abstractmethod
    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        """
        Convert a single column. Must be implemented by subclasses.
        """
        pass

    @staticmethod
    def validate_col_labels(to_check_labels: np.ndarray, valid_labels: List[str]):
        """Validate that all labels in to_check_labels are in valid_labels.

        Args:
            to_check_labels (np.ndarray): The array of labels to validate.
            valid_labels (List[str]): The list of valid labels
        Raises:
            ValueError: If any label in to_check_labels is not in valid_labels.
        """
        for label in to_check_labels:
            if label not in valid_labels:
                raise ValueError(
                    f"Invalid label '{label}' found. Valid labels are: {valid_labels}"
                )

    def get_valid_input_label(self):
        """Validate input labels before conversion."""
        return []

    def get_valid_output_label(self):
        """Validate output labels after conversion."""
        return []

    def do_convert(
        self,
        df: pd.DataFrame,
        ls_target_cols: List[str],
        inplace=False,
        extra_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Run converting logic on specified target columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data to parse.
            ls_target_cols (List[str]): List of column names to apply converting logic on.
            extra_dict (Optional[dict]): Additional parameters for converting logic.

        Returns:
            pd.DataFrame: The DataFrame with converted columns.
        """
        rs_df = df if inplace else df.copy()
        pprint(f"{ls_target_cols=}, {inplace=}, {extra_dict=}")
        for target_col in ls_target_cols:
            valid_input_labels = self.get_valid_input_label()
            if valid_input_labels:
                self.validate_col_labels(
                    rs_df[target_col].to_numpy(), valid_input_labels
                )
            converted_array = self.convert_col(rs_df, target_col, extra_dict)
            valid_ouput_labels = self.get_valid_output_label()
            if valid_ouput_labels:
                self.validate_col_labels(converted_array, valid_ouput_labels)
            rs_df[target_col] = converted_array
        return rs_df


class FireSmokeLabelConverter(BaseCSVConverter):
    """
    Standard converter that helps normalizing fire/smoke labels to 'firesmoke' or 'none'.
    Used as a pre-processing step for metrics or other analyses.
    """

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        assert target_col in df.columns, (
            f"Target column '{target_col}' not found in DataFrame"
        )
        is_label_column = False
        if extra_dict is not None:
            is_label_column = extra_dict.get("is_label_column", False)
        # Apply normalization only to known label columns
        if target_col in [GlobalConst.COL_GT, GlobalConst.COL_PRED] or is_label_column:
            return (
                df[target_col]
                .str.lower()
                .apply(
                    lambda x: GlobalConst.FIRESMOKE_LABEL
                    if ("fire" in x or "smoke" in x)
                    else (GlobalConst.NONE_LABEL if "none" in x else x)
                )
                .to_numpy()
            )
        else:
            # no conversion applied (e.g., elapsed_time)
            return df[target_col].to_numpy()


class TorchMetricsConverter(FireSmokeLabelConverter):
    LABEL_NUM_MAPPING = {
        GlobalConst.FIRESMOKE_LABEL: 1,
        GlobalConst.NONE_LABEL: 0,
    }

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        # first apply base conversion (FireSmokeLabelConverter)
        converted_array = super().convert_col(df, target_col, extra_dict)

        if target_col not in [
            GlobalConst.COL_GT,
            GlobalConst.COL_PRED,
        ]:  # only convert GT/PRED labels to integers
            return converted_array

        return np.array([self.LABEL_NUM_MAPPING[label] for label in converted_array])

    def do_convert(
        self,
        df: pd.DataFrame,
        ls_target_cols: List[str],
        inplace=False,
        extra_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        metric_mode = extra_dict.get(  # ty:ignore[possibly-missing-attribute]
            "metric_mode", GlobalConst.METRIC_PER_FRAME
        )
        if metric_mode == GlobalConst.METRIC_PER_VIDEO:
            temp_df = super().do_convert(df, ls_target_cols, inplace, extra_dict)
            gt = temp_df[GlobalConst.COL_GT].to_numpy().tolist()
            type_in_gt = list(set(gt))
            final_video_gt = GlobalConst.NONE_LABEL
            if self.LABEL_NUM_MAPPING[GlobalConst.FIRESMOKE_LABEL] in type_in_gt:
                final_video_gt = GlobalConst.FIRESMOKE_LABEL

            temp_df[GlobalConst.COL_GT] = [
                self.LABEL_NUM_MAPPING[final_video_gt]
            ] * len(temp_df)

            return temp_df.head(
                1
            )  # only single row (first row) is needed for per-video metric

        else:
            return super().do_convert(df, ls_target_cols, inplace, extra_dict)
