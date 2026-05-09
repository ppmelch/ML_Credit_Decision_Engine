import pandas as pd


class DataPreparation:
    """
    Class responsible for preparing raw credit data for modeling.

    This class handles the preprocessing pipeline required before
    model training, including:

    - Separating the target variable from predictor variables.
    - Removing non-predictive or unnecessary columns.
    - Encoding categorical variables using one-hot encoding.
    - Selecting the final subset of features used for modeling.

    Attributes
    ----------
    data : pd.DataFrame
        Raw input dataset copied from the original source.

    SELECTED_FEATURES : list[str]
        List of final predictor variables selected for model training.
    """

    SELECTED_FEATURES = [
        "pa_cv_remesa",
        "pa_pendiente_norm",
        "pa_n_meses_sin_envio",
        "pa_max_racha_sin_envio",
        "pa_fuerza_estacional",
        "pa_antiguedad_meses",
        "pa_ratio_remesa_cuota",

        "pb_edad",
        "pb_n_dependientes",
        "pb_escolaridad_ord",
        "pb_vivienda_propia",

        "pc_score_regional",
        "pc_sub_capacidad_pago"
    ]

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the DataPreparation object.

        Parameters
        ----------
        data : pd.DataFrame
            Raw dataset containing borrower information and loan outcomes.
        """
        self.data = data.copy()

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Execute the full preprocessing pipeline for model input.

        This method performs the following steps:

        1. Extracts the target variable ('default_12m').
        2. Removes irrelevant or non-feature columns.
        3. Encodes categorical variables into dummy/indicator variables.
        4. Filters the dataset to retain only selected model features.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            A tuple containing:

            - X : Processed feature matrix ready for modeling.
            - y : Target variable representing loan default status.

        Raises
        ------
        ValueError
            If one or more selected features are not found after preprocessing.
        """
        # Target
        y = self.data['default_12m']

        # Features
        X = self.data.drop(
            columns=["default_12m"]
        )

        # Encoding
        X = pd.get_dummies(X, drop_first=True)

        # Validate selected features
        missing = set(self.SELECTED_FEATURES) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing selected features after preprocessing: {missing}")

        # Feature selection
        X = X[self.SELECTED_FEATURES]

        return X, y