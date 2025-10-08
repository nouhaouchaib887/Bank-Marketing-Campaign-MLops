import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
        pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass


# Concrete Strategy for Z-Score Based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        """"
        # Threshold:

        # ----------

        # A threshold is a cutoff value used to decide when a condition is met.

        # It separates what is considered normal from what is abnormal, or one class from another.

        # The threshold can be:

        # - Set manually (e.g., z-score > 3 for outliers)

        # - Derived statistically from data (e.g., mean + 3*std or a percentile)

        # - Optimized automatically (e.g., maximizing accuracy or F1-score)

        # Choosing an appropriate threshold depends on the context, data distribution,

        # and the balance between false positives and false negatives.

        """
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        """
        A z-score measures how many standard deviations a data point is from the mean.
        It is computed as: z = (x - mean) / std
        This transformation standardizes the data so that it has:
        - Mean = 0
        - Standard deviation = 1
        Z-scores are useful for comparing values from different distributions,
        detecting outliers, and preparing data for algorithms sensitive to scale
        """
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers


# Concrete Strategy for IQR Based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        """
        IQR (Interquartile Range) Method:
        ---------------------------------
        Detects outliers based on data spread.
        IQR = Q3 - Q1, where Q1 and Q3 are the 25th and 75th percentiles.
        Values below (Q1 - 1.5IQR) or above (Q3 + 1.5IQR) are considered outliers.
        Works well for non-normal and skewed data.
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outliers detected using the IQR method.")
        return outliers


# Context Class for Outlier Detection and Handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        # Capping outliers means limiting (or “winsorizing”) extreme values so they stay
        # within a defined range — instead of removing them.
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv("../extracted_data/AmesHousing.csv")
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    # outliers = outlier_detector.detect_outliers(df_numeric)
    # df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # print(df_cleaned.shape)
    # # Visualize outliers in specific features
    # # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
    pass


#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    2000000       10000


#    SalePrice  Gr Liv Area
# 0     200000        1500
# 1     300000        2000
# 2     400000        2500
# 3     500000        3000
# 4    1000000        4000
# 5    1000000        4000
