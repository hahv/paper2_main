from halib import *
from abc import ABC
from typing import List, Optional
from src.common import GlobalConst


class BaseCSVConverter(ABC):
    """This converter is responsible for converting CSV field values to target formats."""

    # ! default, can be overridden by subclasses
    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        assert target_col in df.columns, (
            f"Target column '{target_col}' not found in DataFrame"
        )
        if target_col in [GlobalConst.COL_GT, GlobalConst.COL_PRED]:
            return (
                df[target_col]
                .str.lower()
                .apply(
                    lambda x: GlobalConst.FIRESMOKE_LABEL
                    if ("fire" in x or "smoke" in x)
                    else GlobalConst.NONE_LABEL
                )
                .to_numpy()
            )
        else:
            return df[
                target_col
            ].to_numpy()  # no conversion applied (e.g., elapsed_time)

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
        for target_col in ls_target_cols:
            converted_array = self.convert_col(rs_df, target_col, extra_dict)
            rs_df[target_col] = converted_array
        return rs_df


class TorchMetricsConverter(BaseCSVConverter):
    LABEL_NUM_MAPPING = {
        GlobalConst.FIRESMOKE_LABEL: 1,
        GlobalConst.NONE_LABEL: 0,
    }

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        # first apply base conversion
        converted_array = super().convert_col(df, target_col, extra_dict)
        if target_col not in [
            GlobalConst.COL_GT,
            GlobalConst.COL_PRED,
        ]:  # only convert GT/PRED labels to integers
            return np.array(
                [self.LABEL_NUM_MAPPING[label] for label in converted_array]
            )
        # for other columns, no further conversion
        return converted_array

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
