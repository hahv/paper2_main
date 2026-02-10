"""This converter is responsible for converting CSV field values to target formats."""
from halib import *
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseCSVConverter(ABC):
    """
    Base class for converting logic.
    'timeline_type' determines which color schema to validate against.
    """

    @abstractmethod
    def convert_logic(self, df: pd.DataFrame, target_col: str, extra_dict:Optional[dict] = None) -> np.ndarray:
        """Pure logic implementation.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data to parse.
            target_col (str): The column name to apply converting logic on.
            extra_dict (Optional[dict]): Additional parameters for converting logic. E.g: mode = ['per_video', 'per_frame'] when calculating video-level metrics or frame-level metrics.
        """
        pass

    def run(self, df: pd.DataFrame, ls_target_cols: List[str], extra_dict:Optional[dict] = None) -> dict:
        """Run converting logic on specified target columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data to parse.
            ls_target_cols (List[str]): List of column names to apply converting logic on.
            extra_dict (Optional[dict]): Additional parameters for converting logic.

        Returns:
            dict: A dictionary with target column names as keys and converted numpy arrays as values.
        """
        result = {}
        for target_col in ls_target_cols:
            converted_array = self.convert_logic(df, target_col, extra_dict)
            result[target_col] = converted_array
        return result