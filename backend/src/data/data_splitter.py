import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Class responsible for splitting data into training and testing sets.

    This class encapsulates the train-test split logic to ensure
    reproducibility and consistency across the pipeline.
    """

    def __init__(self, test_size: float = 0.25, random_state: int = 42) -> None:
        """
        Initialize the data splitter.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the test split (default = 0.25).
        random_state : int, optional
            Seed used for random number generation to ensure reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state , stratify=y)