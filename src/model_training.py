import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import time

class ModelTrainer:
    def __init__(self):
        self.models = {
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB()
        }
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def train_models(self, X_train, y_train):
        print("\n=== Model Training ===")
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            self.models[name] = model
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            training_time = time.time() - start_time
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            print(f"✓ {name} trained in {training_time:.2f} seconds")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return self.models
    
    def hyperparameter_tuning(self, X_train, y_train):
        print("\n=== Hyperparameter Tuning ===")
        print("\nTuning KNN...")
        knn_params = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        self.models['KNN'] = knn_grid.best_estimator_
        print(f"✓ Best KNN parameters: {knn_grid.best_params_}")
        print(f"  Best score: {knn_grid.best_score_:.4f}")
        
        print("\nTuning Logistic Regression...")
        lr_params = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
        lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=5, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_grid.best_estimator_
        print(f"✓ Best Logistic Regression parameters: {lr_grid.best_params_}")
        print(f"  Best score: {lr_grid.best_score_:.4f}")
        return self.models
    
    def select_best_model(self, X_train, y_train):
        print("\n=== Model Selection ===")
        best_score = 0
        best_name = None
        for name, model in self.models.items():
            score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
            if score > best_score:
                best_score = score
                best_name = name
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f"🏆 Best Model: {best_name}")
        print(f"   Cross-validation Accuracy: {best_score:.4f}")
        return self.best_model, self.best_model_name
    
    def save_model(self, model, filepath='models/best_model.pkl'):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the model
        joblib.dump(model, filepath)
        print(f"\n✓ Model saved to {filepath}")
        # Verify the file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"  ✓ File size: {file_size} bytes")
        else:
            print(f"  ❌ ERROR: File was not saved!")
    
    def save_preprocessors(self, preprocessor, engineer, filepath='models/'):
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(preprocessor, f'{filepath}/preprocessor.pkl')
        joblib.dump(engineer, f'{filepath}/feature_engineer.pkl')
        print(f"✓ Preprocessors saved to {filepath}")
        # Verify files were saved
        if os.path.exists(f'{filepath}/preprocessor.pkl'):
            print(f"  ✓ preprocessor.pkl saved")
        if os.path.exists(f'{filepath}/feature_engineer.pkl'):
            print(f"  ✓ feature_engineer.pkl saved")
