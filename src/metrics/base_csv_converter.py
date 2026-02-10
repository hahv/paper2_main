from halib import *
from abc import ABC
from typing import List, Optional
from src.metrics.loaders.base_csv_loader import BaseVideoCsvLoader


class BaseCSVConverter(ABC):
    """This converter is responsible for converting CSV field values to target formats."""

    # ! default, can be overridden by subclasses
    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        assert target_col in df.columns, (
            f"Target column '{target_col}' not found in DataFrame"
        )
        suppported_col = [BaseVideoCsvLoader.COL_GT, BaseVideoCsvLoader.COL_PRED]
        assert target_col in suppported_col, (
            f"Conversion for column '{target_col}' is not supported"
        )
        return (
            df[target_col]
            .str.lower()
            .apply(
                lambda x: BaseVideoCsvLoader.FIRESMOKE_LABEL
                if ("fire" in x or "smoke" in x)
                else BaseVideoCsvLoader.NONE_LABEL
            )
            .to_numpy()
        )

    def do_convert(
        self,
        df: pd.DataFrame,
        ls_target_cols: List[str],
        inplace=True,
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
    METRIC_MODE_PER_FRAME = "per_frame"
    METRIC_MODE_PER_VIDEO = "per_video"
    LABEL_NUM_MAPPING = {
        BaseVideoCsvLoader.FIRESMOKE_LABEL: 1,
        BaseVideoCsvLoader.NONE_LABEL: 0,
    }

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        # first apply base conversion
        converted_array = super().convert_col(df, target_col, extra_dict)

        mapped_array = np.array(
            [self.LABEL_NUM_MAPPING[label] for label in converted_array]
        )
        return mapped_array

    def do_convert(
        self,
        df: pd.DataFrame,
        ls_target_cols: List[str],
        inplace=True,
        extra_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        metric_mode = extra_dict.get(  # ty:ignore[possibly-missing-attribute]
            "metric_mode", TorchMetricsConverter.METRIC_MODE_PER_FRAME
        )
        if metric_mode == TorchMetricsConverter.METRIC_MODE_PER_VIDEO:
            temp_df = super().do_convert(df, ls_target_cols, inplace, extra_dict)
            gt = temp_df[BaseVideoCsvLoader.COL_GT].to_numpy().tolist()
            type_in_gt = list(set(gt))
            final_video_gt = BaseVideoCsvLoader.NONE_LABEL
            if self.LABEL_NUM_MAPPING[BaseVideoCsvLoader.FIRESMOKE_LABEL] in type_in_gt:
                final_video_gt = BaseVideoCsvLoader.FIRESMOKE_LABEL

            temp_df[BaseVideoCsvLoader.COL_GT] = [
                self.LABEL_NUM_MAPPING[final_video_gt]
            ] * len(temp_df)

            return temp_df.head(
                1
            )  # only single row (first row) is needed for per-video metric

        else:
            return super().do_convert(df, ls_target_cols, inplace, extra_dict)
