import pandas as pd


class BusinessLogic:
    """
    Implements core business rules for credit decisioning,
    customer risk segmentation, and pricing-related logic.

    Attributes
    ----------
    threshold : float
        Probability of Default (PD) cutoff used to approve or reject applications.
        Clients with PD below this threshold are approved.
    """

    def __init__(self, threshold=0.5, LGD=0.45 , rf=0.069971,
                 spread_fondeo=0.03, operating_cost=0.035, capital_cost=0.0189, profit_margin=0.015):
        self.threshold = threshold
        self.LGD = LGD
        self.rf = rf
        self.sfondeo = spread_fondeo
        self.coper = operating_cost
        self.ccapital = capital_cost
        self.m = profit_margin

    def credit_decision(self, pd_values: pd.Series) -> pd.Series:
        """
        Generate binary credit decisions based on PD values.

        Parameters
        ----------
        pd_values : pd.Series
            Series containing Probability of Default (PD) estimates for each client.

        Returns
        -------
        pd.Series
            Binary decision for each client:
            - 1 : Approved (PD < threshold)
            - 0 : Rejected (PD >= threshold)

        Notes
        -----
        This rule represents a simple risk-based cutoff strategy.
        In real-world applications, this threshold may vary depending on
        portfolio strategy, regulatory constraints, or economic conditions.
        """
        return (pd_values < self.threshold).astype(int)

    def risk_buckets(self, pd_values: pd.Series, q=5) -> pd.Series:
        """
        Segment clients into risk buckets based on PD quantiles.

        Parameters
        ----------
        pd_values : pd.Series
            Series containing Probability of Default (PD) values.
        q : int, default=5
            Number of quantile-based buckets (e.g., 5 = quintiles).

        Returns
        -------
        pd.Series
            Categorical series indicating the risk bucket assigned to each client.

        Notes
        -----
        - Buckets are created using quantiles (equal-sized groups).
        - Lower buckets correspond to lower risk (lower PD).
        - Useful for portfolio segmentation, pricing strategies, and monitoring.
        - If there are too many duplicate PD values, pd.qcut may raise an error.
        """
        return pd.qcut(pd_values, q=q)
    
    def calculate_interest_rate(self, pd_values: pd.Series) -> pd.Series:
        """
        Calculate interest rate based on risk (PD).

        Higher PD -> higher interest rate.
        """
        
        risk_premium = pd_values * self.LGD 
        
        # rf + spread_fondeo + risk_premium + costos_operativos + costo_capital + Margen de utilidad
        
        return self.rf + self.sfondeo + risk_premium + self.coper + self.ccapital + self.m
    
    
    
