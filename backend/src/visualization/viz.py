import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, roc_curve, confusion_matrix as sk_confusion_matrix

from backend.src.modeling.geospatial_risk import BASE_DIR, FRONTEND_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", palette="Greys_r")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


class Visualization:
    """
    Visualization utilities for evaluating and analyzing credit risk models.

    Provides methods for plotting model performance metrics, distributions,
    and relationships between key variables such as PD, expected loss, and interest rate.
    """

    def plot_roc_curve(self, y_test, y_prob, name="Test Set"):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        y_test : array-like
            True binary labels.

        y_prob : array-like
            Predicted probabilities for the positive class.

        name : str, optional
            Label for the dataset (e.g., "Train Set", "Test Set").
        """
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#313131")
        plt.plot([0, 1], [0, 1], linestyle='--', color="#580000")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred , name = ""):
        """
        Plot a confusion matrix with labeled cells.

        Parameters
        ----------
        y_test : array-like
            True labels.

        y_pred : array-like
            Predicted class labels.
        """
        cm = sk_confusion_matrix(y_test, y_pred)

        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        annotated = np.empty_like(cm).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotated[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

        cmap = LinearSegmentedColormap.from_list(
            "custom",
            ["#d6d6d6", "#101010"]
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=annotated, fmt="", cmap=cmap, cbar=False)

        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, data, hue, x, y):
        """
        Plot a scatter plot for two variables with optional hue grouping.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        hue : str
            Column name used for color grouping.

        x : str
            Column name for x-axis.

        y : str
            Column name for y-axis.
        """
        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=["#313131", "#505050"],
            marker="^",
            s=70,
            alpha=0.8
        )

        plt.xlabel(x.replace('_', ' ').title())
        plt.ylabel(y.replace('_', ' ').title())
        plt.title(
            f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}")
        plt.legend(title=hue)
        plt.grid(alpha=0.2)
        plt.show()

    def plot_bar(self, data, x, y):
        """
        Plot a bar chart of average values grouped by a categorical variable.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        x : str
            Grouping variable.

        y : str
            Numerical variable to average.
        """
        data.groupby(x)[y].mean().plot(kind='bar')

        plt.title(
            f"Average {y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
        plt.xlabel(x.replace('_', ' ').title())
        plt.ylabel(f"Average {y.replace('_', ' ').title()}")
        plt.xticks(rotation=0)
        plt.grid(alpha=0.2)
        plt.show()

    def plot_distribution(self, x, dataset_name, var_name):
        """
        Plot the distribution (KDE) of a numerical variable.

        Parameters
        ----------
        x : array-like
            Data values.

        dataset_name : str
            Dataset label.

        var_name : str
            Variable name for labeling.
        """
        plt.figure(figsize=(10, 5))

        sns.kdeplot(
            x=x,
            fill=True,
            alpha=0.3,
            linewidth=2
        )

        plt.title(f"{dataset_name} {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Density")
        plt.grid(alpha=0.2)
        plt.show()

    def plot_boxplot(self, data, x, y, hue=None, order=None):
        """
        Plot a boxplot for a numerical variable grouped by categories.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        x : str
            Categorical variable.

        y : str
            Numerical variable.

        hue : str, optional
            Additional grouping variable.

        order : list, optional
            Order of categories for x-axis.
        """

        data = data.copy()

        if x == "risk_bucket":
            bucket_labels = [
                "Low Risk",
                "Medium-Low Risk",
                "Medium Risk",
                "High Risk",
                "Very High Risk"
            ]

            data[x] = data[x].astype(str)

            unique_buckets = sorted(data[x].unique())

            mapping = dict(zip(unique_buckets, bucket_labels))

            data[x] = data[x].map(mapping)

            order = bucket_labels

        plt.figure(figsize=(10, 5))

        ax = sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            palette=["#303030", "#868686"],
            showfliers=False,
            linewidth=1.2
        )

        ax.set_title(
            f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
        )

        ax.set_xlabel(x.replace('_', ' ').title())
        ax.set_ylabel(y.replace('_', ' ').title())

        plt.grid(axis='y', alpha=0.2)

        sns.despine()

        plt.tight_layout()

        plt.show()

    def plot_probability_density(self , y_test , proba, model_name: str = "Modelo", class_names: tuple[str, str, str] = ("Denied","Approved")) -> None:
        """
        Plot kernel density estimation of predicted probabilities.

        This visualization shows how probability estimates differ
        across the true classes.

        Parameters
        ----------
        y_test : array-like
            True class labels.
        proba : np.ndarray
            Predicted class probabilities.
        model_name : str
            Model name used in the plot title.
        class_names : tuple
            Names of the classes.
        """
        y_test = np.array(y_test)

        plt.figure()

        if proba.ndim == 1:

            sns.kdeplot(
                proba[y_test == 0],
                fill=True,
                alpha=0.3,
                linewidth=2,
                label=class_names[0]
            )

            sns.kdeplot(
                proba[y_test == 1],
                fill=True,
                alpha=0.3,
                linewidth=2,
                label=class_names[1]
            )

        plt.title(f"Probability Density - {model_name}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
        
        
    def plot_real_vs_predicted_pd(self,
            predicted_pd: np.ndarray,
            true_labels: pd.Series | np.ndarray,
            predicted_labels: np.ndarray,
            dataset_name: str = "Train"
        ) -> None:
        """
        Compare real and predicted PD distributions for Approved/Denied classes.

        Parameters
        ----------
        predicted_pd : np.ndarray
            Predicted probability of default values.
        true_labels : array-like
            True borrower labels.
        predicted_labels : np.ndarray
            Predicted borrower labels.
        dataset_name : str
            Dataset identifier (Train/Test).
        """

        results_df = pd.DataFrame({
            "Predicted_PD": predicted_pd,
            "Real_Class": true_labels,
            "Predicted_Class": predicted_labels
        })

        label_names = {
            0: "Approved",
            1: "Denied"
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for class_id in range(2):

            real_dist = results_df[
                results_df["Real_Class"] == class_id
            ]["Predicted_PD"]

            pred_dist = results_df[
                results_df["Predicted_Class"] == class_id
            ]["Predicted_PD"]

            sns.kdeplot(
                real_dist,
                ax=axes[class_id],
                label="Real",
                fill=True,
                alpha=0.3
            )

            sns.kdeplot(
                pred_dist,
                ax=axes[class_id],
                label="Predicted",
                linestyle="--",
                linewidth=2
            )

            axes[class_id].set_title(
                f"{label_names[class_id]}"
                )

            axes[class_id].set_xlabel("Predicted PD")
            axes[class_id].set_ylabel("Density")
            axes[class_id].legend()

        plt.tight_layout()
        plt.show()
        
        

    def plot_calibration_curve(
        self,
        y_true,
        predicted_pd,
        dataset_name="Test Set",
        n_bins=50
    ):
        """
        Plot calibration curve comparing predicted PD vs observed default rate.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        predicted_pd : array-like
            Predicted probabilities of default.
        dataset_name : str
            Dataset label.
        n_bins : int
            Number of bins for calibration.
        """

        prob_true, prob_pred = calibration_curve(
            y_true,
            predicted_pd,
            n_bins=n_bins,
            strategy='uniform'
        )

        plt.figure(figsize=(7, 5))

        plt.plot(
            prob_pred,
            prob_true,
            marker='o',
            linewidth=2,
            color="#313131",
            label="Model Calibration"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color="#580000",
            label="Perfect Calibration"
        )

        plt.xlabel("Mean Predicted PD")
        plt.ylabel("Observed Default Rate")
        plt.title(f"Calibration Curve - {dataset_name}")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()



    def plot_all(self, results, data):
        """
        Generate a full set of visualizations for model evaluation and analysis.
        """

        # ==========================================
        # 1. MODEL DISCRIMINATION PERFORMANCE
        # ==========================================
        
        # ROC Curves
        self.plot_roc_curve(
            results['y_test'],
            results['y_prob'],
            name="Test Set"
        )

        self.plot_roc_curve(
            results['y_train'],
            results['y_train_prob'],
            name="Train Set"
        )

        # Confusion Matrices
        self.plot_confusion_matrix(
            results['y_test'],
            results['y_pred'],
            name="Test"
        )

        self.plot_confusion_matrix(
            results['y_train'],
            results['y_train_pred'],
            name="Train"
        )

        # ==========================================
        # 2. PROBABILITY / PD VALIDATION
        # ==========================================

        # Calibration Curves
        self.plot_calibration_curve(
            results['y_train'],
            results['y_train_prob'],
            dataset_name="Train Set"
        )

        self.plot_calibration_curve(
            results['y_test'],
            results['y_prob'],
            dataset_name="Test Set"
        )

        # PD Density by Class
        self.plot_probability_density(
            results['y_train'],
            results['y_train_prob'],
            model_name="Train Set - Predicted PD",
            class_names=("Approved", "Denied")
        )

        self.plot_probability_density(
            results['y_test'],
            results['y_prob'],
            model_name="Test Set - Predicted PD",
            class_names=("Approved", "Denied")
        )

        # Real vs Predicted PD Alignment
        self.plot_real_vs_predicted_pd(
            predicted_pd=results['y_train_prob'],
            true_labels=results['y_train'],
            predicted_labels=results['y_train_pred'],
            dataset_name="Train Set"
        )

        self.plot_real_vs_predicted_pd(
            predicted_pd=results['y_prob'],
            true_labels=results['y_test'],
            predicted_labels=results['y_pred'],
            dataset_name="Test Set"
        )

        # ==========================================
        # 3. RISK SEGMENTATION / FINANCIAL OUTPUTS
        # ==========================================
        '''
        # Risk Bucket Means
        self.plot_bar(
            data,
            x='risk_bucket',
            y='predicted_pd'
        )

        self.plot_bar(
            data,
            x='risk_bucket',
            y='expected_loss'
        )

        # Risk Bucket Distributions
        self.plot_boxplot(
            data,
            x='risk_bucket',
            y='predicted_pd'
        )

        self.plot_boxplot(
            data,
            x='risk_bucket',
            y='expected_loss'
        )

        self.plot_boxplot(
            data,
            x='risk_bucket',
            y='interest_rate_model'
        )
        '''
    def export_dashboard_data(self, results, data):

        fpr_train, tpr_train, _ = roc_curve(
            results["y_train"],
            results["y_train_prob"]
        )

        fpr_test, tpr_test, _ = roc_curve(
            results["y_test"],
            results["y_prob"]
        )

        cm_train = sk_confusion_matrix(
            results["y_train"],
            results["y_train_pred"]
        )

        cm_test = sk_confusion_matrix(
            results["y_test"],
            results["y_pred"]
        )

        risk_bucket = (
            data["risk_bucket"]
            .value_counts()
            .sort_index()
        )

        dashboard_data = {

            "roc_train": {
                "fpr": fpr_train.tolist(),
                "tpr": tpr_train.tolist()
            },

            "roc_test": {
                "fpr": fpr_test.tolist(),
                "tpr": tpr_test.tolist()
            },

            "cm_train": cm_train.tolist(),

            "cm_test": cm_test.tolist(),

            "density_train": {
                "approved":
                    results["y_train_prob"][
                        results["y_train"] == 0
                    ].tolist(),

                "denied":
                    results["y_train_prob"][
                        results["y_train"] == 1
                    ].tolist()
            },

            "density_test": {
                "approved":
                    results["y_prob"][
                        results["y_test"] == 0
                    ].tolist(),

                "denied":
                    results["y_prob"][
                        results["y_test"] == 1
                    ].tolist()
            },

            "risk_bucket": {
                "labels": risk_bucket.index.tolist(),
                "values": risk_bucket.values.tolist()
            }

        }

        output_path = (
            FRONTEND_DIR / "dashboard_data.json"
        )

        with open(output_path, "w") as f:

            json.dump(dashboard_data, f)