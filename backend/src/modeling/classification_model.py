import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from backend.src.utils.utils import compute_scale_pos_weight
from backend.src.modeling.config import MODEL_CONFIG
from backend.src.modeling.base_model import BaseModel

class ClassificationModel(BaseModel):
    """
    Wrapper class for classification models with support for multiple algorithms.

    This class provides a unified interface for training, prediction,
    probability estimation, and model persistence. The model configuration
    is retrieved from a centralized configuration dictionary.

    Supported models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Additional models can be added by extending the initialization logic and updating the configuration.
    """

    def __init__(self, model_name: str , **kwargs) -> None:
        """
        Initialize the classification model based on the specified model name.

        Parameters
        ----------
        model_name : str
            - Name of the model to initialize. 
            
            - Supported values are:
            'logistic', 'random_forest', 'xgboost', 'lightgbm'.

        Raises
        ------
        ValueError
            If the provided model_name is not supported.
        """
        self.model_name = model_name
        self.config = MODEL_CONFIG.get(model_name, {}).copy()
        self.config.update(kwargs)

        if model_name == "logistic":
            self.model = LogisticRegression(**self.config)

        elif model_name == "random_forest":
            self.model = RandomForestClassifier(**self.config)

        elif model_name == "xgboost":
            self.model = XGBClassifier(**self.config)
            
        elif model_name == "lightgbm":
            self.model = LGBMClassifier(**self.config)

        else:
            raise ValueError("Modelo no soportado")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the classification model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used for training.
        y : pd.Series
            Target variable (binary classification).
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate class predictions for the input data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        pd.Series
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predicted probabilities for the positive class (PD).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        pd.Series
            Probability of the positive class (default probability).
        """
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, filename: str, models_dir) -> None:
        """
        Save the trained model to disk using joblib.

        The method stores the model along with metadata such as the model name
        and configuration.

        Parameters
        ----------
        filename : str
            Name of the file to save the model (e.g., 'model.pkl').
        models_dir : Path
            Directory where the model will be stored.
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / filename

        joblib.dump({
            "model": self.model,
            "model_name": self.model_name,
            "config": self.config
        }, path)

    def load_model(self, filename: str, models_dir) -> None:
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        filename : str
            Name of the file containing the saved model.
        models_dir : Path
            Directory where the model is stored.
        """
        path = models_dir / filename
        data = joblib.load(path)

        self.model = data["model"]
        self.model_name = data["model_name"]
        self.config = data["config"]