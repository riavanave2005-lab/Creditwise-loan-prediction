import pandas as pd
import numpy as np
def create_loan_dataset():
    """Create a synthetic loan dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'ApplicantIncome': np.random.normal(5000, 2000, n_samples),
        'CoapplicantIncome': np.random.normal(2000, 1000, n_samples),
        'LoanAmount': np.random.normal(150, 50, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 60], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable
    df['Loan_Status'] = ((df['Credit_History'] == 1) & 
                         (df['ApplicantIncome'] > 3000) & 
                         (df['LoanAmount'] < 200)).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1*n_samples), replace=False)
    df.loc[noise_idx, 'Loan_Status'] = 1 - df.loc[noise_idx, 'Loan_Status']
    
    # Save to CSV
    df.to_csv('data/loan_data.csv', index=False)
    print(f"✅ Dataset created with {n_samples} samples")
    print(f"📊 Approval rate: {df['Loan_Status'].mean()*100:.1f}%")
    print(f"📁 Saved to: data/loan_data.csv")
    return df

if __name__ == "__main__":
    create_loan_dataset()
