import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.created_features = []
    
    def create_ratio_features(self, df):
        data = df.copy()
        
        # Income to Loan ratio - handle division by zero
        if 'ApplicantIncome' in data.columns and 'LoanAmount' in data.columns:
            # Add small epsilon to avoid division by zero
            data['Income_Loan_Ratio'] = data['ApplicantIncome'] / (data['LoanAmount'] + 1e-6)
            # Replace inf with 0 and NaN with 0
            data['Income_Loan_Ratio'] = data['Income_Loan_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
            self.created_features.append('Income_Loan_Ratio')
            print("✓ Created Income_Loan_Ratio")
        
        # Coapplicant income ratio
        if 'CoapplicantIncome' in data.columns and 'ApplicantIncome' in data.columns:
            data['Coapp_Income_Ratio'] = data['CoapplicantIncome'] / (data['ApplicantIncome'] + 1e-6)
            data['Coapp_Income_Ratio'] = data['Coapp_Income_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
            self.created_features.append('Coapp_Income_Ratio')
            print("✓ Created Coapp_Income_Ratio")
        
        # Total Income
        if 'ApplicantIncome' in data.columns and 'CoapplicantIncome' in data.columns:
            data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
            self.created_features.append('Total_Income')
            print("✓ Created Total_Income")
        
        # Loan Amount to Total Income ratio
        if 'Total_Income' in data.columns and 'LoanAmount' in data.columns:
            data['Loan_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1e-6)
            data['Loan_Income_Ratio'] = data['Loan_Income_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
            self.created_features.append('Loan_Income_Ratio')
            print("✓ Created Loan_Income_Ratio")
        
        return data
    
    def apply_log_transformation(self, df):
        data = df.copy()
        
        # Log transform for skewed features - handle negative/zero values
        skewed_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        
        for feature in skewed_features:
            if feature in data.columns:
                # Shift values to be positive if needed (add 1 to avoid log(0))
                # Using log1p which is log(1+x), safer for positive values
                data[f'Log_{feature}'] = np.log1p(data[feature].clip(lower=0))
                # Replace any remaining NaN or inf
                data[f'Log_{feature}'] = data[f'Log_{feature}'].replace([np.inf, -np.inf], 0).fillna(0)
                self.created_features.append(f'Log_{feature}')
                print(f"✓ Created Log_{feature}")
        
        return data
    
    def create_interaction_features(self, df):
        data = df.copy()
        
        # Credit History and Loan Amount interaction
        if 'Credit_History' in data.columns and 'LoanAmount' in data.columns:
            data['Credit_Loan_Interaction'] = data['Credit_History'] * data['LoanAmount']
            data['Credit_Loan_Interaction'] = data['Credit_Loan_Interaction'].replace([np.inf, -np.inf], 0).fillna(0)
            self.created_features.append('Credit_Loan_Interaction')
            print("✓ Created Credit_Loan_Interaction")
        
        return data
    
    def engineer_features(self, df):
        print("\n=== Feature Engineering ===")
        data = df.copy()
        
        # Step 1: Create ratio features
        data = self.create_ratio_features(data)
        
        # Step 2: Apply log transformation
        data = self.apply_log_transformation(data)
        
        # Step 3: Create interaction features
        data = self.create_interaction_features(data)
        
        # Final cleanup: Replace any remaining NaN or inf with 0
        # Get all numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace inf with 0
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            # Replace NaN with 0
            data[col] = data[col].fillna(0)
        
        print(f"\n✓ Created {len(self.created_features)} new features")
        print(f"✓ Total features: {len(data.columns)}")
        
        return data
