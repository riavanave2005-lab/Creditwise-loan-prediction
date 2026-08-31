import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def main():
    print("="*60)
    print("🏦 CreditWise - Loan Prediction System")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    print("\n📂 Loading Data...")
    data = preprocessor.load_data('data/loan_data.csv')
    
    print("\n🧹 Cleaning Data...")
    data_cleaned = preprocessor.clean_data(data)
    
    print("\n🔧 Feature Engineering...")
    data_engineered = engineer.engineer_features(data_cleaned)
    
    print("\n🔢 Encoding Categorical Variables...")
    data_encoded = preprocessor.encode_categorical(data_engineered)
    
    print("\n🎯 Preparing Features and Target...")
    X, y = preprocessor.prepare_features(data_encoded, 'Loan_Status')
    
    print("\n📊 Splitting Data...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print("\n📏 Scaling Features...")
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    models = trainer.train_models(X_train_scaled, y_train)
    models = trainer.hyperparameter_tuning(X_train_scaled, y_train)
    best_model, best_model_name = trainer.select_best_model(X_train_scaled, y_train)
    best_name = evaluator.evaluate_all_models(models, X_test_scaled, y_test)
    
    print("\n📈 Generating Visualizations...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_model_comparison()
    
    print("\n💾 Saving Models...")
    trainer.save_model(best_model)
    trainer.save_preprocessors(preprocessor, engineer)
    
    print("\n" + "="*60)
    print("✅ CREDITWISE - COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
