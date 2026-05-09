import pandas as pd


class PrintUtils:
    """
    Utility class for displaying model outputs, risk metrics,
    and portfolio analyses in a structured and readable format.

    This class centralizes all printing logic to ensure consistency
    across evaluation, risk segmentation, and pricing analysis.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the PrintUtils class.

        Parameters
        ----------
        data : pd.DataFrame
            Final dataset containing model outputs and risk metrics,
            such as predicted PD, expected loss, risk buckets, and decisions.
        """
        self.data = data

    def print_model_results(self, results: dict) -> None:
        print("\n=== Model Evaluation ===")

        metrics_to_show = [
            "test_roc_auc",
            "train_roc_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1_score",
        ]

        for key in metrics_to_show:
            print(f"{key}: {results[key]}")


    def print_sample(self) -> None:
        """
        Display a sample of the dataset with key outputs.

        Notes
        -----
        Shows the first rows including:
        - predicted_pd : estimated probability of default
        - expected_loss : calculated expected loss
        - decision : credit approval decision
        """
        print("\n=== Sample Results ===")
        print(self.data[['predicted_pd', 'expected_loss', 'decision']].head())

    def print_risk_analysis(self) -> None:
        """
        Display average risk metrics by risk bucket.

        Notes
        -----
        Aggregates the following metrics by risk segment:
        - predicted_pd
        - expected_loss

        Useful for understanding how risk is distributed across segments.
        """
        print("\n=== Risk Buckets (PD & Expected Loss) ===")
        print(
            self.data.groupby('risk_bucket')[
                ['predicted_pd', 'expected_loss']].mean()
        )

    def print_pricing_analysis(self) -> None:
        """
        Display relationship between risk and pricing.

        Notes
        -----
        Aggregates:
        - predicted_pd
        - interest_rate

        by risk bucket to analyze whether pricing is aligned with risk levels.
        """
        print("\n=== PD vs Interest Rate ===")
        print(
            self.data.groupby('risk_bucket')[
                ['predicted_pd', 'interest_rate_model']].mean()
        )

    def print_all(self, results: dict) -> None:
        """
        Execute all available print methods.

        Parameters
        ----------
        results : dict
            Model evaluation metrics.

        Notes
        -----
        This method provides a complete overview of:
        - Model performance
        - Sample predictions
        - Risk segmentation
        - Pricing consistency
        """
        self.print_model_results(results)
        #self.print_sample()
        #self.print_risk_analysis()
        #self.print_pricing_analysis()