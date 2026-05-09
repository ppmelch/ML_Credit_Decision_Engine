from backend.src.modeling.model import Model
from backend.src.modeling.config import MODEL_CONFIG, MODELS_DIR
from backend.src.data.data_splitter import DataSplitter
from backend.src.data.data_preparation import DataPreparation
from backend.src.modeling.model_evaluation import ModelEvaluation
import warnings

warnings.filterwarnings("ignore")



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
        
        
        # LOGICA de negocio para decidir si el modelo es lo suficientemente bueno para ser guardado y desplegado
  
        
        # Save Model
        model.save_model(f"{self.model_name}.pkl", MODELS_DIR)
        
        # 6. Deployment (if applicable)
        
        
        # 7. Inference
        
        
        # 8. Monitoring and maintenance
            
        return results, self.data
    
    
    
    
    

