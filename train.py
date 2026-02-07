"""
Wine Quality Prediction - Training Script
Author: Bhanu Reddy (2022bcd0026)
Course: MLOps Lab 2 - Automated Training with GitHub Actions
Based on Lab 1: Experiment Tracking

This script trains a machine learning model on Wine Quality dataset.
Modify hyperparameters, preprocessing, and model type directly in this file
for each experiment, then commit and push to trigger automated training.
"""

import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# EXPERIMENT CONFIGURATION
# Modify these parameters for each experiment
# ============================================================================

# Model Selection: 'LinearRegression', 'Ridge', 'Lasso', or 'RandomForest'
MODEL_TYPE = 'Ridge'

# Train-Test Split
TEST_SIZE = 0.20

# Preprocessing
USE_SCALING = True  # True to apply StandardScaler

# Feature Selection
FEATURE_SELECTION = 6  # None for all features, or integer (e.g., 6) for top K features

# Hyperparameters for Ridge/Lasso
ALPHA = 1.0  # Regularization strength for Ridge/Lasso

# Hyperparameters for RandomForest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None  # None for unlimited
RF_RANDOM_STATE = 1

# ============================================================================
# DATASET
# ============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def load_data():
    """Load Wine Quality dataset"""
    print("=" * 70)
    print("STEP 1: Loading Dataset")
    print("=" * 70)
    df = pd.read_csv(DATASET_URL, sep=';')
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def preprocess_data(df):
    """Apply preprocessing and feature selection"""
    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing")
    print("=" * 70)
    
    # Separate features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]
    
    # Apply scaling if enabled
    if USE_SCALING:
        print("✓ Applying StandardScaler")
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        print("○ Scaling: Disabled")
    
    # Apply feature selection if enabled
    if FEATURE_SELECTION is not None:
        print(f"✓ Applying feature selection: Top {FEATURE_SELECTION} features")
        selector = SelectKBest(score_func=f_regression, k=FEATURE_SELECTION)
        X = selector.fit_transform(X, y)
    else:
        print("○ Feature Selection: Disabled (using all features)")
    
    return X, y

def train_model(X_train, y_train):
    """Train the selected model"""
    print("\n" + "=" * 70)
    print("STEP 3: Training Model")
    print("=" * 70)
    
    if MODEL_TYPE == 'LinearRegression':
        print("Model: Linear Regression")
        model = LinearRegression()
    elif MODEL_TYPE == 'Ridge':
        print(f"Model: Ridge Regression (alpha={ALPHA})")
        model = Ridge(alpha=ALPHA, random_state=1)
    elif MODEL_TYPE == 'Lasso':
        print(f"Model: Lasso Regression (alpha={ALPHA})")
        model = Lasso(alpha=ALPHA, random_state=1)
    elif MODEL_TYPE == 'RandomForest':
        print(f"Model: Random Forest (n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH})")
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    model.fit(X_train, y_train)
    print("✓ Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute metrics"""
    print("\n" + "=" * 70)
    print("STEP 4: Model Evaluation")
    print("=" * 70)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2

def save_artifacts(model, mse, r2):
    """Save trained model and results"""
    print("\n" + "=" * 70)
    print("STEP 5: Saving Artifacts")
    print("=" * 70)
    
    # Save trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved to: model.pkl")
    
    # Save results to JSON
    results = {
        "student": "Bhanu Reddy",
        "roll_number": "2022bcd0026",
        "lab": "Lab 2 - Automated Training with GitHub Actions",
        "experiment_config": {
            "model_type": MODEL_TYPE,
            "test_size": TEST_SIZE,
            "use_scaling": USE_SCALING,
            "feature_selection": FEATURE_SELECTION,
            "alpha": ALPHA if MODEL_TYPE in ['Ridge', 'Lasso'] else None,
            "rf_n_estimators": RF_N_ESTIMATORS if MODEL_TYPE == 'RandomForest' else None,
            "rf_max_depth": RF_MAX_DEPTH if MODEL_TYPE == 'RandomForest' else None
        },
        "metrics": {
            "MSE": round(mse, 4),
            "R2_Score": round(r2, 4)
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to: results.json")
    
    return results

def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("WINE QUALITY PREDICTION - AUTOMATED TRAINING")
    print("=" * 70)
    print(f"Student: Bhanu Reddy")
    print(f"Roll Number: 2022bcd0026")
    print(f"Lab: Lab 2 - GitHub Actions CI/CD")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Train-test split
    print(f"\nTrain-Test Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=1
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    results = save_artifacts(model, mse, r2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"MSE: {mse:.4f} | R² Score: {r2:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
