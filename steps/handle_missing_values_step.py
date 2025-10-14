from zenml import step
from src.handle_missing_data_values import MissingValueHandler, DropMissingValuesStrategy, FillMissingValuesStrategy
import pandas as pd

@step
def handle_missing_values_step(df: pd.DataFrame, strategy) -> pd.DataFrame:
    """Handles missing values in the DataFrame using the specified strategy.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing missing values.
    
    strategy(str): The strategy to handle missing values. Options are 'drop' or 'fill'
    
    Returns: pd.DataFrame: The DataFrame with missing values handled.
    """
    if strategy =="drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy =="fill":
        handler = MissingValueHandler(FillMissingValuesStrategy())