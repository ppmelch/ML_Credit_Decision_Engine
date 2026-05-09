from backend.src.modeling.classification_model import ClassificationModel
from backend.src.modeling.config import MODEL_CONFIG
from backend.src.utils.utils import compute_scale_pos_weight


class Model:

    @staticmethod
    def get_model(
        task_type: str,
        model_name: str,
        y_train=None
    ):

        if task_type == "classification":

            allowed_models = [
                "logistic",
                "random_forest",
                "xgboost",
                "lightgbm"
            ]

            if model_name not in allowed_models:
                raise ValueError(
                    f"Model '{model_name}' not supported"
                )

            model_config = MODEL_CONFIG[model_name].copy()

            # Dynamic runtime parameters
            if model_name == "xgboost" and y_train is not None:

                model_config["scale_pos_weight"] = (
                    compute_scale_pos_weight(y_train)
                )

            return ClassificationModel(
                model_name,
                **model_config
            )

        else:
            raise ValueError("Invalid model type")