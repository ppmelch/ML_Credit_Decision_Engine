from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix


class ModelEvaluation:
    """
    Class for evaluating machine learning models.

    This class supports evaluation for classification tasks using
    common performance metrics such as ROC-AUC, accuracy, precision,
    recall, F1-score, and confusion matrix.

    The design allows future extension to regression tasks.
    """

    def __init__(self, task_type: str = "classification") -> None:
        """
        Initialize the evaluation object.

        Parameters
        ----------
        task_type : str, optional
            Type of task to evaluate. Currently supports:
            - 'classification'
            Default is 'classification'.
        """
        self.task_type = task_type

    def evaluate(self, y_true, y_pred, y_pred_proba=None) -> dict:
        """
        Evaluate model predictions based on the specified task type.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted class labels.
        y_pred_proba : array-like, optional
            Predicted probabilities for the positive class.
            Required for ROC-AUC calculation.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.

        Raises
        ------
        ValueError
            If the task_type is not supported.
        """
        if self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred, y_pred_proba)

        # Future extension for regression
        # elif self.task_type == "regression":
        #     return self._evaluate_regression(y_true, y_pred)

        else:
            raise ValueError("Type of evaluation not supported")

    def _evaluate_classification(self, y_true, y_pred, y_pred_proba) -> dict:
        """
        Compute classification performance metrics.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        y_pred_proba : array-like
            Predicted probabilities for the positive class.

        Returns
        -------
        dict
            Dictionary with the following metrics:
            - roc_auc
            - accuracy
            - precision
            - recall
            - f1_score
            - confusion_matrix
        """
        results = {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        return results

    # Regression
        '''
    def _evaluate_regression(self, y_true, y_pred) -> dict:
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        results = {
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "r2": r2_score(y_true, y_pred)
        }
        
        return results
        '''
        
    def evaluate_full(self, y_train,y_train_pred ,y_train_proba, y_test, y_test_pred, y_test_proba):
        """
        Perform full train/test evaluation and return structured results.

        Parameters
        ----------
        y_train : pd.Series
            True training labels.

        y_train_pred : np.ndarray
            Predicted training labels.

        y_train_proba : np.ndarray
            Predicted training probabilities.

        y_test : pd.Series
            True testing labels.

        y_test_pred : np.ndarray
            Predicted testing labels.

        y_test_proba : np.ndarray
            Predicted testing probabilities.

        Returns
        -------
        dict
            Dictionary containing train/test metrics and raw predictions.
        """

        results = {}

        test_metrics = self._evaluate_classification(
            y_test,
            y_test_pred,
            y_test_proba
        )

        results.update({
            f"test_{k}": v for k, v in test_metrics.items()
        })

        train_metrics = self._evaluate_classification(
            y_train,
            y_train_pred,
            y_train_proba
        )

        results.update({
            f"train_{k}": v for k, v in train_metrics.items()
        })

        results["y_test"] = y_test
        results["y_pred"] = y_test_pred
        results["y_prob"] = y_test_proba

        results["y_train"] = y_train
        results["y_train_pred"] = y_train_pred
        results["y_train_prob"] = y_train_proba

        return results