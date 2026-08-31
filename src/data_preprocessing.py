import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} records")
        print(f"Features: {data.columns.tolist()}")
        return data
    
    def clean_data(self, data):
        df = data.copy()
        
        # Check for missing values before cleaning
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            print(f"\n⚠️ Missing values found before cleaning:")
            print(missing_before[missing_before > 0])
        
        # Handle missing values for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  ✓ Filled missing values in {col}")
        
        # Handle missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  ✓ Filled missing values in {col}")
        
        # Final check
        missing_after = df.isnull().sum()
        if missing_after.sum() > 0:
            print(f"\n⚠️ Still has missing values: {missing_after[missing_after > 0]}")
        else:
            print(f"\n✅ No missing values after cleaning")
        
        return df
    
    def encode_categorical(self, data):
        df = data.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Loan_Status':  # Target variable
                le = LabelEncoder()
                # Convert to string and handle NaN
                df[col] = df[col].astype(str).fillna('Unknown')
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col}")
        
        return df
    
    def prepare_features(self, data, target_col='Loan_Status'):
        if target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Encode target if it's categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.label_encoders[target_col] = le
            
            self.feature_names = X.columns.tolist()
            print(f"\n✓ Prepared {len(X.columns)} features and target")
            return X, y
        return data, None
    
    def scale_features(self, X_train, X_test=None):
        # Get numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            print(f"  ✓ Scaling {len(numerical_cols)} numerical features")
            
            # Handle any remaining NaN before scaling
            X_train_clean = X_train.copy()
            X_train_clean[numerical_cols] = X_train_clean[numerical_cols].fillna(0)
            
            # Scale
            X_train_scaled = X_train_clean.copy()
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train_clean[numerical_cols])
            
            if X_test is not None:
                X_test_clean = X_test.copy()
                X_test_clean[numerical_cols] = X_test_clean[numerical_cols].fillna(0)
                X_test_scaled = X_test_clean.copy()
                X_test_scaled[numerical_cols] = self.scaler.transform(X_test_clean[numerical_cols])
                return X_train_scaled, X_test_scaled
            
            return X_train_scaled
        
        print("  ⚠️ No numerical features to scale")
        return X_train
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\n📊 Data Split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Training approval rate: {y_train.mean()*100:.1f}%")
        print(f"  Test approval rate: {y_test.mean()*100:.1f}%")
        return X_train, X_test, y_train, y_test
