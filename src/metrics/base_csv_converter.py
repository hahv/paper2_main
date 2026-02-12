from IPython.testing.decorators import f
from halib import *
from abc import ABC, abstractmethod
from typing import List, Optional, Any
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
    def do_validate_lbs(to_check_labels: np.ndarray, valid_labels: Optional[List[str]]):
        """Validate that all labels in to_check_labels are in valid_labels.

        Args:
            to_check_labels (np.ndarray): The array of labels to validate.
            valid_labels (List[str]): The list of valid labels
        Raises:
            ValueError: If any label in to_check_labels is not in valid_labels.
        """
        if valid_labels is None:
            return
        for label in to_check_labels:
            if label not in valid_labels:
                assert False, (
                    f"Invalid label '{label}' found. Valid labels are: {valid_labels}"
                )

    # --- Configuration Properties ---
    @property
    def valid_in_lbs(self) -> Optional[List[Any]]:
        """Override to provide a default list of valid input labels."""
        return None

    @property
    def valid_out_lbs(self) -> Optional[List[Any]]:
        """Override to provide a default list of valid output labels."""
        return None

    @staticmethod
    # ! The order of cols_convert_tuple_ls matters for chain conversion
    def do_convert_chain(
        df: pd.DataFrame,
        cols_convert_tuple_ls: List[tuple[str, "BaseCSVConverter"]],
        inplace=True,
        extra_dict: Optional[dict] = None,
        context="",  # for debugging purpose
    ) -> pd.DataFrame:
        """Run converting logic on specified target columns with different converters (with possible chain conversion).
        E.g: [('col1', ConverterA), ('col1', ConverterB)] means first apply ConverterA on 'col1', then apply ConverterB on the result.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data to parse.
            cols_convert_tuple_ls (List[tuple[str, "BaseCSVConverter"]]): List of tuples containing column names and their respective converters.
            Note that each converter can have its own logic, and we can do the chain conversion if needed.
            extra_dict (Optional[dict]): Additional parameters for converting logic.

        Returns:
            pd.DataFrame: The DataFrame with converted columns.
        """
        rs_df = df if inplace else df.copy()
        if context:
            console.rule(f"<do_convert_chain>: {context}")
            pprint(f"{inplace=}, {extra_dict=}")
            for target_col, converter in cols_convert_tuple_ls:
                pprint(f"col:{target_col} => converter: {converter.__class__.__name__}")
        for target_col, converter in cols_convert_tuple_ls:
            converter.do_validate_lbs(
                rs_df[target_col].to_numpy(), converter.valid_in_lbs
            )
            converted_array = converter.convert_col(rs_df, target_col, extra_dict)
            converter.do_validate_lbs(converted_array, converter.valid_out_lbs)
            rs_df[target_col] = converted_array
            if context:
                console.rule(
                    f"Converted column '{target_col}' using {converter.__class__.__name__}"
                )
                pprint(
                    f"Unique values after conversion: {rs_df[target_col].unique().tolist()}"
                )
                csvfile.fn_display_df(rs_df.head(3))
                console.rule(f"End of conversion for column '{target_col}'")
        if context:
            console.rule(f"End of <do_convert_chain>: {context}")
        return rs_df

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
        pprint(f"{ls_target_cols=}, {inplace=}, {extra_dict=}")
        for target_col in ls_target_cols:
            self.do_validate_lbs(rs_df[target_col].to_numpy(), self.valid_in_lbs)
            converted_array = self.convert_col(rs_df, target_col, extra_dict)
            self.do_validate_lbs(converted_array, self.valid_out_lbs)
            rs_df[target_col] = converted_array
        return rs_df


class FireSmokeLabelConverter(BaseCSVConverter):
    """
    Standard converter that helps normalizing fire/smoke labels to 'firesmoke' or 'none'.
    Used as a pre-processing step for metrics or other analyses.
    """

    @property
    def valid_out_lbs(self) -> Optional[List[Any]]:
        """Override to provide a default list of valid output labels."""
        return [GlobalConst.FIRESMOKE_LABEL, GlobalConst.NONE_LABEL]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        target_col_values = df[target_col].unique().tolist()
        # check if any 'fire' or 'smoke' or 'none' in any case of target col values
        has_fire_smoke_none = any(
            [
                ("fire" in str(val).lower())
                or ("smoke" in str(val).lower())
                or ("none" in str(val).lower())
                for val in target_col_values
            ]
        )
        if has_fire_smoke_none:
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


class TorchMetricsConverter(BaseCSVConverter):
    LABEL_NUM_MAPPING = {
        GlobalConst.FIRESMOKE_LABEL: 1,
        GlobalConst.NONE_LABEL: 0,
    }

    @property
    def valid_in_lbs(self) -> Optional[List[Any]]:
        return [GlobalConst.FIRESMOKE_LABEL, GlobalConst.NONE_LABEL]

    @property
    def valid_out_lbs(self) -> Optional[List[Any]]:
        return [1, 0]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        return df[target_col].apply(lambda x: self.LABEL_NUM_MAPPING[x]).to_numpy()

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
        assert metric_mode in [
            GlobalConst.METRIC_PER_FRAME,
            GlobalConst.METRIC_PER_VIDEO,
        ], f"Unknown metric_mode: {metric_mode}"
        if metric_mode == GlobalConst.METRIC_PER_FRAME:
            # do normal conversion (per-frame), do not aggregate
            return super().do_convert(df, ls_target_cols, inplace, extra_dict)
        else:
            # METRIC_PER_VIDEO = do per-video conversion (aggregate to single row)
            df_converted = super().do_convert(
                df, ls_target_cols, inplace=False, extra_dict=extra_dict
            )
            rs_df = df_converted.copy()
            rs_df[ls_target_cols] = df_converted.groupby(GlobalConst.COL_VIDEO)[
                ls_target_cols
            ].transform("max")

            return rs_df
