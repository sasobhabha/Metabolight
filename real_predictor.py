#!/usr/bin/env python3
"""
Updated frontend for REAL-DATA SCFA-Neurotransmission models
"""

import numpy as np
import pandas as pd
import joblib
import json

def load_real_model(neurotransmitter):
    """Load a trained real-data model and its metadata"""
    try:
        model_data = joblib.load(f'real_model_{neurotransmitter}.pkl')
        with open(f'real_model_{neurotransmitter}_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model_data, metadata
    except FileNotFoundError:
        print(f"Model files for {neurotransmitter} not found. Please train models first.")
        return None, None

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
    print("REAL-DATA SCFA-Neurotransmission Prediction Interface")
    print("=" * 55)
    
    # Load all models
    neurotransmitters = ['serotonin', 'dopamine', 'gaba']
    models = {}
    metadata = {}
    
    for nt in neurotransmitters:
        model_data, meta = load_real_model(nt)
        if model_data is not None:
            models[nt] = model_data
            metadata[nt] = meta
            print(f"Loaded {nt} model: {meta['best_model']} (R² = {meta['performance']['test_r2']:.3f})")
        else:
            print(f"Failed to load {nt} model")
    
    if not models:
        print("No models loaded. Please train models first using real_scfamodel.py")
        return
    
    print("\nEnter SCFA and bacterial data for prediction:")
    print("(Enter 'quit' to exit, or just press Enter for typical healthy values)")
    
    while True:
        print("\n" + "-" * 40)
        try:
            # Get input values with realistic defaults based on dataset means
            acetate_input = input("Acetate (mM) [default: 27.0]: ").strip()
            if acetate_input.lower() == 'quit':
                break
            acetate = float(acetate_input) if acetate_input else 27.0
            
            propionate_input = input("Propionate (mM) [default: 48.0]: ").strip()
            if propionate_input.lower() == 'quit':
                break
            propionate = float(propionate_input) if propionate_input else 48.0
            
            butyrate_input = input("Butyrate (mM) [default: 30.0]: ").strip()
            if butyrate_input.lower() == 'quit':
                break
            butyrate = float(butyrate_input) if butyrate_input else 30.0
            
            lactobacillus_input = input("Lactobacillus abundance [default: 0.05]: ").strip()
            if lactobacillus_input.lower() == 'quit':
                break
            lactobacillus = float(lactobacillus_input) if lactobacillus_input else 0.05
            
            bifidobacterium_input = input("Bifidobacterium abundance [default: 0.05]: ").strip()
            if bifidobacterium_input.lower() == 'quit':
                break
            bifidobacterium = float(bifidobacterium_input) if bifidobacterium_input else 0.05
            
            clostridia_input = input("Clostridia abundance [default: 0.15]: ").strip()
            if clostridia_input.lower() == 'quit':
                break
            clostridia = float(clostridia_input) if clostridia_input else 0.15
            
            bacteroides_input = input("Bacteroides abundance [default: 0.40]: ").strip()
            if bacteroides_input.lower() == 'quit':
                break
            bacteroides = float(bacteroides_input) if bacteroides_input else 0.40
            
            veillonella_input = input("Veillonella abundance [default: 0.01]: ").strip()
            if veillonella_input.lower() == 'quit':
                break
            veillonella = float(veillonella_input) if veillonella_input else 0.01
            
            akkermansia_input = input("Akkermansia abundance [default: 0.01]: ").strip()
            if akkermansia_input.lower() == 'quit':
                break
            akkermansia = float(akkermansia_input) if akkermansia_input else 0.01
            
            # Validate inputs
            if acetate < 0 or propionate < 0 or butyrate < 0:
                print("Error: SCFA concentrations cannot be negative")
                continue
            for name, val in [('lactobacillus', lactobacillus), ('bifidobacterium', bifidobacterium), 
                              ('clostridia', clostridia), ('bacteroides', bacteroides), 
                              ('veillonella', veillonella), ('akkermansia', akkermansia)]:
                if not (0 <= val <= 1):
                    print(f"Error: {name} abundance must be between 0 and 1")
                    break
            else:
                # All abundances valid
                pass
            
            # Create input dictionary
            scfa_input = {
                'acetate': acetate,
                'propionate': propionate,
                'butyrate': butyrate,
                'lactobacillus': lactobacillus,
                'bifidobacterium': bifidobacterium,
                'clostridia': clostridia,
                'bacteroides': bacteroides,
                'veillonella': veillonella,
                'akkermansia': akkermansia
            }
            
            # Make predictions
            print("\nPREDICTIONS (based on real literature data):")
            print("-" * 45)
            for nt in neurotransmitters:
                if nt in models:
                    pred = predict_neurotransmitter(models[nt], scfa_input)
                    print(f"{nt.capitalize():<10}: {pred:.3f}")
            
            # Show input summary
            print("\nINPUT SUMMARY:")
            print("-" * 15)
            print(f"SCFAs: Acetate={acetate:.1f}, Propionate={propionate:.1f}, Butyrate={butyrate:.1f} mM")
            print(f"Bacteria: Lacto={lactobacillus:.2f}, Bifido={bifidobacterium:.2f}, Clost={clostridia:.2f}, "
                  f"Bact={bacteroides:.2f}, Veillo={veillonella:.2f}, Akk={akkermansia:.2f}")
            
            # Provide interpretation
            print("\nINTERPRETATION:")
            print("-" * 15)
            total_scfa = acetate + propionate + butyrate
            if total_scfa > 80:
                scfa_level = "High"
            elif total_scfa > 40:
                scfa_level = "Moderate"
            else:
                scfa_level = "Low"
            print(f"SCFA Level: {scfa_level} (Total: {total_scfa:.1f} mM)")
            
            butyrate_ratio = butyrate / (total_scfa + 1e-8)
            if butyrate_ratio > 0.25:
                butyrate_status = "High (neuroprotective)"
            elif butyrate_ratio > 0.15:
                butyrate_status = "Moderate"
            else:
                butyrate_status = "Low"
            print(f"Butyrate Ratio: {butyrate_status} ({butyrate_ratio:.1%})")
            
            # Bacteroides ratio (propionate producer)
            total_bacteria = lactobacillus + bifidobacterium + clostridia + bacteroides + veillonella + akkermansia
            if total_bacteria > 0:
                bacteroides_ratio = bacteroides / total_bacteria
                if bacteroides_ratio > 0.25:
                    bacteroides_status = "Good (propionate support)"
                elif bacteroides_ratio > 0.15:
                    bacteroides_status = "Moderate"
                else:
                    bacteroides_status = "Low"
                print(f"Bacteroides Ratio: {bacteroides_status} ({bacteroides_ratio:.1%})")
            else:
                print("Bacteroides Ratio: N/A (no bacteria measured)")
            
            # Clostridia ratio (butyrate producer)
            if total_bacteria > 0:
                clostridia_ratio = clostridia / total_bacteria
                if clostridia_ratio > 0.20:
                    clostridia_status = "Good (butyrate support)"
                elif clostridia_ratio > 0.10:
                    clostridia_status = "Moderate"
                else:
                    clostridia_status = "Low"
                print(f"Clostridia Ratio: {clostridia_status} ({clostridia_ratio:.1%})")
            
            # Biological sanity check: propionate should not exceed acetate in healthy gut
            if propionate > acetate:
                print("\n⚠️  BIOLOGICAL WARNING: Propionate exceeds acetate.")
                print("   In healthy guts, acetate is typically the dominant SCFA (≥50% of total).")
                print("   Propionate > acetate may indicate dysbiosis or measurement issue.")
            
        except ValueError:
            print("Please enter valid numbers or 'quit' to exit")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Real-Data SCFA-Neurotransmission predictor!")

if __name__ == "__main__":
    main()