"""
SCFA-Neurotransmission Model
Placeholder for human data model linking SCFAs, gut bacteria, and neurotransmitters
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import json

def load_human_data():
    """
    Placeholder function to load human data from literature.
    In practice, this would extract quantitative data from:
    - Fecal SCFA concentrations (acetate, propionate, butyrate)
    - Neurotransmitter levels (serotonin, dopamine, GABA)
    - Bacterial taxa abundances
    - Clinical phenotypes
    """
    # This is a placeholder - replace with actual data extraction
    print("Loading human data from literature sources...")
    
    # Simulated dataset structure based on literature
    np.random.seed(42)
    n_samples = 100
    
    # Features: SCFA concentrations and bacterial abundances
    acetate = np.random.lognormal(mean=3, sigma=0.5, size=n_samples)  # mM
    propionate = np.random.lognormal(mean=2, sigma=0.4, size=n_samples)  # mM
    butyrate = np.random.lognormal(mean=1.5, sigma=0.3, size=n_samples)  # mM
    
    # Bacterial abundances (relative)
    lactobacillus = np.random.beta(2, 5, size=n_samples)
    bifidobacterium = np.random.beta(1.5, 4, size=n_samples)
    clostridia = np.random.beta(3, 4, size=n_samples)
    
    # Derived features
    total_scfa = acetate + propionate + butyrate
    butyrate_ratio = butyrate / (total_scfa + 1e-8)
    acetate_propionate_ratio = acetate / (propionate + 1e-8)
    
    # Target: Neurotransmitter levels (simulated based on literature relationships)
    # Based on studies showing SCFAs influence neurotransmitter production
    serotonin = (
        0.3 * np.log(acetate + 1) +
        0.2 * np.log(propionate + 1) +
        0.4 * np.log(butyrate + 1) +
        0.2 * lactobacillus +
        0.1 * bifidobacterium -
        0.1 * clostridia +
        np.random.normal(0, 0.1, n_samples)
    )
    
    dopamine = (
        0.25 * np.log(acetate + 1) +
        0.3 * np.log(propionate + 1) +
        0.35 * np.log(butyrate + 1) +
        0.15 * lactobacillus +
        0.05 * bifidobacterium +
        0.1 * clostridia +
        np.random.normal(0, 0.1, n_samples)
    )
    
    gaba = (
        0.2 * np.log(acetate + 1) +
        0.25 * np.log(propionate + 1) +
        0.4 * np.log(butyrate + 1) +
        0.3 * lactobacillus +
        0.2 * bifidobacterium -
        0.05 * clostridia +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'acetate': acetate,
        'propionate': propionate,
        'butyrate': butyrate,
        'lactobacillus': lactobacillus,
        'bifidobacterium': bifidobacterium,
        'clostridia': clostridia,
        'total_scfa': total_scfa,
        'butyrate_ratio': butyrate_ratio,
        'acetate_propionate_ratio': acetate_propionate_ratio,
        'serotonin': serotonin,
        'dopamine': dopamine,
        'gaba': gaba
    })
    
    return data

def train_model(data, target_neurotransmitter='serotonin'):
    """
    Train a model to predict neurotransmitter levels from SCFA and bacterial features
    """
    print("Training model for {} prediction...".format(target_neurotransmitter))
    
    # Prepare features and target
    feature_cols = ['acetate', 'propionate', 'butyrate', 
                   'lactobacillus', 'bifidobacterium', 'clostridia',
                   'total_scfa', 'butyrate_ratio', 'acetate_propionate_ratio']
    
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
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
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
            # For linear regression, use absolute coefficients
            feature_importance = np.abs(model.coef_)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'feature_importance': dict(zip(feature_cols, feature_importance))
        }
        
        print("{}:".format(name))
        print("  Train R^2: {:.3f}".format(train_r2))
        print("  Test R^2: {:.3f}".format(test_r2))
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
    
    # Select best model based on test R^2
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
    
    joblib.dump(model_data, 'model_{}.pkl'.format(target_neurotransmitter))
    
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
        }
    }
    
    with open('model_{}_metadata.json'.format(target_neurotransmitter), 'w') as f:
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
    Main function to demonstrate the modeling workflow
    """
    print("=" * 50)
    print("SCFA-Neurotransmission Model Training")
    print("=" * 50)
    
    # Load data
    data = load_human_data()
    print("Loaded dataset with {} samples".format(len(data)))
    print("Features: {}".format(list(data.columns)))
    print("")
    
    # Train models for each neurotransmitter
    neurotransmitters = ['serotonin', 'dopamine', 'gaba']
    models = {}
    
    for nt in neurotransmitters:
        print("Training model for {}".format(nt.upper()))
        print("-" * 30)
        model_data = train_model(data, nt)
        models[nt] = model_data
        print("")
    
    # Example prediction
    print("Example Prediction:")
    print("-" * 20)
    sample_input = {
        'acetate': 50.0,      # mM
        'propionate': 20.0,   # mM
        'butyrate': 15.0,     # mM
        'lactobacillus': 0.3,
        'bifidobacterium': 0.25,
        'clostridia': 0.15,
        'total_scfa': 85.0,
        'butyrate_ratio': 0.176,
        'acetate_propionate_ratio': 2.5
    }
    
    for nt in neurotransmitters:
        pred = predict_neurotransmitter(models[nt], sample_input)
        print("Predicted {}: {:.3f}".format(nt, pred))
    
    print("\nModel training complete!")
    print("Files saved:")
    for nt in neurotransmitters:
        print("  - model_{}.pkl".format(nt))
        print("  - model_{}_metadata.json".format(nt))

if __name__ == "__main__":
    main()