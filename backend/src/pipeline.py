from backend.src.modeling.model import Model
from backend.src.modeling.config import MODELS_DIR
from backend.src.data.data_splitter import DataSplitter
from backend.src.data.data_preparation import DataPreparation
from backend.src.modeling.business_logic import BusinessLogic
from backend.src.modeling.geospatial_risk import GeospatialRisk
from backend.src.modeling.risk_calculator import RiskCalculator
from backend.src.modeling.model_evaluation import ModelEvaluation


class CreditPipeline:
    '''Pipeline for credit risk modeling.'''
    def __init__(self, data, model_name="random_forest"):
        self.data = data
        self.model_name = model_name
    
    def run(self):
        
        # 1. Data ingestion and preprocessing
        prep = DataPreparation(self.data)
        X, y = prep.prepare_data()
        
        # 3. Train-test split
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(X, y)
    
        # 4. Model selection and instantiation
        model = Model.get_model("classification", self.model_name, y_train=y_train) 
        
        # 4.1 Model training
        model.train(X_train, y_train)
        
        #4.1.1 Model predictions (train & test)
                # === TRAIN ===
        y_train_proba = model.predict_proba(X_train)
        y_train_pred = model.predict(X_train)
        
                # === TEST ===
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        # 4.2 Model evaluation
        evaluator = ModelEvaluation()
        
        results = evaluator.evaluate_full(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_train_proba=y_train_proba,
        y_test=y_test,
        y_test_pred=y_test_pred,
        y_test_proba=y_test_proba
        )
        
        
        # 5. Risk metrics calculation and business logic application
        risk = RiskCalculator(lgd=0.45)
        pd_values = risk.calculate_pd(model, X)
        self.data["predicted_pd"] = pd_values

        # 5.1 Risk metrics
        ead = risk.calculate_ead(self.data)
        lgd = risk.calculate_lgd(self.data)

        self.data["expected_loss"] = risk.calculate_expected_loss(
            pd_values, lgd, ead
        )

        # 5.2 Business logic
        logic = BusinessLogic(threshold=0.4, LGD=0.45 , rf=0.069971, 
                 spread_fondeo=0.03, operating_cost=0.035, capital_cost=0.0189, profit_margin=0.015)
        self.data["decision"] = logic.credit_decision(pd_values)
        self.data["risk_bucket"] = logic.risk_buckets(pd_values)
        self.data["interest_rate_model"] = logic.calculate_interest_rate(pd_values)

                
        # Save Model
        model.save_model(f"{self.model_name}.pkl", MODELS_DIR)
        
        
        # Geospatial analysis (if applicable)
        geo = GeospatialRisk()
        municipality_risk = (geo.build_municipality_risk(self.data))
        
        # 6. Deployment (if applicable)
        
        
        # 7. Inference
        
        
        # 8. Monitoring and maintenance
            
        return results, self.data
    
    
    
    
    

