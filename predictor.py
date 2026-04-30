#!/usr/bin/env python3
"""
Simple frontend for SCFA-Neurotransmission models
"""

import numpy as np
import pandas as pd
import joblib
import json

def load_model(neurotransmitter):
    """Load a trained model and its metadata"""
    model_data = joblib.load(f'model_{neurotransmitter}.pkl')
    with open(f'model_{neurotransmitter}_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model_data, metadata

def predict_neurotransmitter(model_data, scfa_input):
    """Make prediction using loaded model"""
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    
    # Convert input to array
    if isinstance(scfa_input, dict):
        X = np.array([[scfa_input[col] for col in feature_cols]])
    else:
        X = scfa_input
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    
    return prediction[0]

def main():
    print("SCFA-Neurotransmission Prediction Interface")
    print("=" * 45)
    
    # Load all models
    neurotransmitters = ['serotonin', 'dopamine', 'gaba']
    models = {}
    metadata = {}
    
    for nt in neurotransmitters:
        try:
            models[nt], metadata[nt] = load_model(nt)
            print(f"Loaded {nt} model: {metadata[nt]['best_model']} (R² = {metadata[nt]['performance']['test_r2']:.3f})")
        except FileNotFoundError:
            print(f"Warning: Model files for {nt} not found")
            continue
    
    if not models:
        print("No models loaded. Please train models first.")
        return
    
    print("\nEnter SCFA and bacterial data for prediction:")
    print("(Enter 'quit' to exit)")
    
    while True:
        print("\n" + "-" * 30)
        try:
            # Get input values
            acetate = input("Acetate (mM) [default: 50]: ").strip()
            if acetate.lower() == 'quit':
                break
            acetate = float(acetate) if acetate else 50.0
            
            propionate = input("Propionate (mM) [default: 20]: ").strip()
            if propionate.lower() == 'quit':
                break
            propionate = float(propionate) if propionate else 20.0
            
            butyrate = input("Butyrate (mM) [default: 15]: ").strip()
            if butyrate.lower() == 'quit':
                break
            butyrate = float(butyrate) if butyrate else 15.0
            
            lactobacillus = input("Lactobacillus abundance [default: 0.3]: ").strip()
            if lactobacillus.lower() == 'quit':
                break
            lactobacillus = float(lactobacillus) if lactobacillus else 0.3
            
            bifidobacterium = input("Bifidobacterium abundance [default: 0.25]: ").strip()
            if bifidobacterium.lower() == 'quit':
                break
            bifidobacterium = float(bifidobacterium) if bifidobacterium else 0.25
            
            clostridia = input("Clostridia abundance [default: 0.15]: ").strip()
            if clostridia.lower() == 'quit':
                break
            clostridia = float(clostridia) if clostridia else 0.15
            
            # Calculate derived features
            total_scfa = acetate + propionate + butyrate
            butyrate_ratio = butyrate / (total_scfa + 1e-8)
            acetate_propionate_ratio = acetate / (propionate + 1e-8)
            
            # Create input dictionary
            scfa_input = {
                'acetate': acetate,
                'propionate': propionate,
                'butyrate': butyrate,
                'lactobacillus': lactobacillus,
                'bifidobacterium': bifidobacterium,
                'clostridia': clostridia,
                'total_scfa': total_scfa,
                'butyrate_ratio': butyrate_ratio,
                'acetate_propionate_ratio': acetate_propionate_ratio
            }
            
            # Make predictions
            print("\nPREDICTIONS:")
            print("-" * 15)
            for nt in neurotransmitters:
                if nt in models:
                    pred = predict_neurotransmitter(models[nt], scfa_input)
                    print(f"{nt.capitalize():<10}: {pred:.3f}")
            
            # Show input summary
            print("\nINPUT SUMMARY:")
            print("-" * 15)
            print(f"SCFAs: Acetate={acetate:.1f}, Propionate={propionate:.1f}, Butyrate={butyrate:.1f} mM")
            print(f"Bacteria: Lacto={lactobacillus:.2f}, Bifido={bifidobacterium:.2f}, Clost={clostridia:.2f}")
            
        except ValueError:
            print("Please enter valid numbers or 'quit' to exit")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the SCFA-Neurotransmission predictor!")

if __name__ == "__main__":
    main()