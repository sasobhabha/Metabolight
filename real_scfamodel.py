"""
Updated SCFA-Neurotransmission Model using REAL literature-based data with Ridge regression to handle collinearity
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import json

def load_real_data():
    """
    Load the real-data based dataset we created from literature sources
    """
    print("Loading real literature-based dataset...")
    
    try:
        # Try to load our created dataset
        data = pd.read_csv('scfa_neurotransmission_realdata.csv')
        print(f"Loaded real dataset with {len(data)} samples")
        print(f"Features: {list(data.columns)}")
        return data
    except FileNotFoundError:
        print("Real dataset not found, creating it now...")
        # Import and run the dataset creation
        import create_real_dataset
        data = create_real_dataset.add_metadata_and_save()
        return data

def train_model(data, target_neurotransmitter='serotonin'):
    """
    Train a model to predict neurotransmitter levels from SCFA and bacterial features
    """
    print(f"\nTraining model for {target_neurotransmitter} prediction...")
    
    # Prepare features and target - updated bacterial features
    feature_cols = ['acetate', 'propionate', 'butyrate', 
                   'lactobacillus', 'bifidobacterium', 'clostridia',
                   'bacteroides', 'veillonella', 'akkermansia']
    
    X = data[feature_cols]
    y = data[target_neurotransmitter]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    # Use RidgeCV to automatically select alpha and handle collinearity
    model_ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    models = {
        'ridge': model_ridge,
        'random_forest': model_rf
    }
    
    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        # Get feature importances or coefficients
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            # For ridge, use absolute coefficients
            feature_importance = np.abs(model.coef_)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'feature_importance': dict(zip(feature_cols, feature_importance))
        }
        
        print("{}:".format(name))
        print("  Train R²: {:.3f}".format(train_r2))
        print("  Test R²: {:.3f}".format(test_r2))
        print("  Test MSE: {:.3f}".format(test_mse))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print("  Top features:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:3]
            for i in indices:
                print("    {}: {:.3f}".format(feature_cols[i], importances[i]))
        else:
            print("  Top features (coefficient magnitude):")
            coefs = np.abs(model.coef_)
            indices = np.argsort(coefs)[::-1][:3]
            for i in indices:
                print("    {}: {:.3f}".format(feature_cols[i], coefs[i]))
    
    # Select best model based on test R²
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print("\nBest model: {}".format(best_model_name))
    
    # Save model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target': target_neurotransmitter,
        'results': results
    }
    
    joblib.dump(model_data, 'real_model_{}.pkl'.format(target_neurotransmitter))
    
    # Save metadata
    metadata = {
        'target_neurotransmitter': target_neurotransmitter,
        'best_model': best_model_name,
        'performance': {
            'test_r2': results[best_model_name]['test_r2'],
            'test_mse': results[best_model_name]['test_mse']
        },
        'feature_importance': results[best_model_name]['feature_importance'],
        'data_shape': {
            'n_samples': len(data),
            'n_features': len(feature_cols)
        },
        'data_source': 'Real literature-based dataset'
    }
    
    with open('real_model_{}_metadata.json'.format(target_neurotransmitter), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_data

def predict_neurotransmitter(model_data, scfa_data):
    """
    Make predictions for new SCFA/bacterial data
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    
    # Prepare input
    if isinstance(scfa_data, dict):
        # Convert single sample dict to array
        X = np.array([[scfa_data[col] for col in feature_cols]])
    else:
        # Assume already formatted as array or DataFrame
        if hasattr(scfa_data, 'cols'):
            X = scfa_data[feature_cols].values
        else:
            X = scfa_data
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    
    return prediction[0] if len(prediction) == 1 else prediction

def main():
    """
    Main function to demonstrate the modeling workflow with real data
    """
    print("=" * 60)
    print("REAL-DATA SCFA-Neurotransmission Model Training")
    print("=" * 60)
    
    # Load real data
    data = load_real_data()
    
    # Train models for each neurotransmitter
    neurotransmitters = ['serotonin', 'dopamine', 'gaba']
    models = {}
    
    for nt in neurotransmitters:
        print(f"\nTraining model for {nt.upper()}")
        print("-" * 40)
        model_data = train_model(data, nt)
        models[nt] = model_data
    
    # Example prediction with REALISTIC values from the dataset
    # Using values that reflect acetate dominance as per literature
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    # Get mean values from the dataset to ensure biological realism
    sample_input = {
        'acetate': 61.1,      # mM - acetate dominant (~54% of total SCFAs)
        'propionate': 25.9,   # mM
        'butyrate': 25.1,     # mM
        'lactobacillus': 0.074,
        'bifidobacterium': 0.074,
        'clostridia': 0.231,
        'bacteroides': 0.594,
        'veillonella': 0.014,
        'akkermansia': 0.014
    }
    
    print("Input values (based on dataset means):")
    for key, value in sample_input.items():
        if key in ['acetate', 'propionate', 'butyrate']:
            print(f"  {key}: {value} mM")
        else:
            print(f"  {key}: {value} (relative abundance)")
    
    print("\nPredictions:")
    print("-" * 15)
    for nt in neurotransmitters:
        if nt in models:
            pred = predict_neurotransmitter(models[nt], sample_input)
            print(f"{nt.capitalize():<10}: {pred:.3f}")
    
    print("\nModel training complete!")
    print("Files saved:")
    for nt in neurotransmitters:
        print("  - real_model_{}.pkl".format(nt))
        print("  - real_model_{}_metadata.json".format(nt))

if __name__ == "__main__":
    main()