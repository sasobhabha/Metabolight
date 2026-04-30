#!/usr/bin/env python3
"""
Flask GUI for SCFA-Neurotransmission prediction model.
Deployable on Render as a web service.
"""

import numpy as np
import pandas as pd
import joblib
import json
from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# Load models and metadata at startup
def load_model(neurotransmitter):
    try:
        model_data = joblib.load(f'real_model_{neurotransmitter}.pkl')
        with open(f'real_model_{neurotransmitter}_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model_data, metadata
    except FileNotFoundError:
        return None, None

def predict_neurotransmitter(model_data, scfa_input):
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']
    if isinstance(scfa_input, dict):
        X = np.array([[scfa_input[col] for col in feature_cols]])
    else:
        X = scfa_input
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return prediction[0]

# Load all models
neurotransmitters = ['serotonin', 'dopamine', 'gaba']
models = {}
metadata = {}
for nt in neurotransmitters:
    m, meta = load_model(nt)
    if m is not None:
        models[nt] = m
        metadata[nt] = meta

# HTML template as a string for simplicity (no separate templates folder)
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>SCFA-Neurotransmission Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="number"] { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #2980b9; }
        .result { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 4px; }
        .warning { color: #e74c3c; font-weight: bold; }
        .info { color: #2c3e50; }
    </style>
</head>
<body>
<div class="container">
    <h1>SCFA-Neurotransmission Predictor</h1>
    <p>Enter your gut microbiome test results to predict neurotransmitter levels.</p>
    <form method="post" action="{{ url_for('predict') }}">
        <div class="form-group">
            <label for="acetate">Acetate (mM)</label>
            <input type="number" id="acetate" name="acetate" step="0.1" value="{{ acetate|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="propionate">Propionate (mM)</label>
            <input type="number" id="propionate" name="propionate" step="0.1" value="{{ propionate|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="butyrate">Butyrate (mM)</label>
            <input type="number" id="butyrate" name="butyrate" step="0.1" value="{{ butyrate|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="lactobacillus">Lactobacillus abundance (0-1)</label>
            <input type="number" id="lactobacillus" name="lactobacillus" step="0.01" min="0" max="1" value="{{ lactobacillus|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="bifidobacterium">Bifidobacterium abundance (0-1)</label>
            <input type="number" id="bifidobacterium" name="bifidobacterium" step="0.01" min="0" max="1" value="{{ bifidobacterium|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="clostridia">Clostridia abundance (0-1)</label>
            <input type="number" id="clostridia" name="clostridia" step="0.01" min="0" max="1" value="{{ clostridia|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="bacteroides">Bacteroides abundance (0-1)</label>
            <input type="number" id="bacteroides" name="bacteroides" step="0.01" min="0" max="1" value="{{ bacteroides|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="veillonella">Veillonella abundance (0-1)</label>
            <input type="number" id="veillonella" name="veillonella" step="0.01" min="0" max="1" value="{{ veillonella|default:'' }}" required>
        </div>
        <div class="form-group">
            <label for="akkermansia">Akkermansia abundance (0-1)</label>
            <input type="number" id="akkermansia" name="akkermansia" step="0.01" min="0" max="1" value="{{ akkermansia|default:'' }}" required>
        </div>
        <button type="submit">Predict Neurotransmitter Levels</button>
    </form>

    {% if prediction %}
    <div class="result">
        <h2>Prediction Results</h2>
        <p class="info"><strong>Serotonin:</strong> {{ prediction.serotonin|round(3) }}</p>
        <p class="info"><strong>Dopamine:</strong> {{ prediction.dopamine|round(3) }}</p>
        <p class="info"><strong>GABA:</strong> {{ prediction.gaba|round(3) }}</p>
        
        <h3>Input Summary</h3>
        <p class="info"><strong>SCFAs:</strong> Acetate={{ acetate }} mM, Propionate={{ propionate }} mM, Butyrate={{ butyrate }} mM</p>
        <p class="info"><strong>Bacteria:</strong> Lacto={{ lactobacillus|round(3) }}, Bifido={{ bifidobacterium|round(3) }}, Clost={{ clostridia|round(3) }}, Bact={{ bacteroides|round(3) }}, Veillo={{ veillonella|round(3) }}, Akk={{ akkermansia|round(3) }}</p>
        
        <h3>Interpretation</h3>
        <p class="info"><strong>SCFA Level:</strong> {{ scfa_level }}</p>
        <p class="info"><strong>Butyrate Ratio:</strong> {{ butyrate_status }} ({{ butyrate_ratio|round(1) }}%)</p>
        <p class="info"><strong>Bacteroides Ratio:</strong> {{ bacteroides_status }} ({{ bacteroides_ratio|round(1) }}%)</p>
        <p class="info"><strong>Clostridia Ratio:</strong> {{ clostridia_status }} ({{ clostridia_ratio|round(1) }}%)</p>
        
        {% if warning %}
        <p class="warning">{{ warning }}</p>
        {% endif %}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        acetate = float(request.form['acetate'])
        propionate = float(request.form['propionate'])
        butyrate = float(request.form['butyrate'])
        lactobacillus = float(request.form['lactobacillus'])
        bifidobacterium = float(request.form['bifidobacterium'])
        clostridia = float(request.form['clostridia'])
        bacteroides = float(request.form['bacteroides'])
        veillonella = float(request.form['veillonella'])
        akkermansia = float(request.form['akkermansia'])
    except (ValueError, KeyError):
        return redirect(url_for('index'))

    # Validate ranges
    if acetate < 0 or propionate < 0 or butyrate < 0:
        return render_template_string(HTML_TEMPLATE, error="SCFA concentrations cannot be negative")
    for name, val in [('lactobacillus', lactobacillus), ('bifidobacterium', bifidobacterium),
                      ('clostridia', clostridia), ('bacteroides', bacteroides),
                      ('veillonella', veillonella), ('akkermansia', akkermansia)]:
        if not (0 <= val <= 1):
            return render_template_string(HTML_TEMPLATE, error=f"{name} abundance must be between 0 and 1")

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
    prediction_dict = {}
    for nt in neurotransmitters:
        if nt in models:
            pred = predict_neurotransmitter(models[nt], scfa_input)
            prediction_dict[nt] = pred

    # Calculate interpretation values
    total_scfa = acetate + propionate + butyrate
    if total_scfa > 80:
        scfa_level = "High"
    elif total_scfa > 40:
        scfa_level = "Moderate"
    else:
        scfa_level = "Low"

    butyrate_ratio = butyrate / (total_scfa + 1e-8)
    if butyrate_ratio > 0.25:
        butyrate_status = "High (neuroprotective)"
    elif butyrate_ratio > 0.15:
        butyrate_status = "Moderate"
    else:
        butyrate_status = "Low"

    total_bacteria = lactobacillus + bifidobacterium + clostridia + bacteroides + veillonella + akkermansia
    if total_bacteria > 0:
        bacteroides_ratio = bacteroides / total_bacteria
        if bacteroides_ratio > 0.25:
            bacteroides_status = "Good (propionate support)"
        elif bacteroides_ratio > 0.15:
            bacteroides_status = "Moderate"
        else:
            bacteroides_status = "Low"
    else:
        bacteroides_ratio = 0
        bacteroides_status = "N/A"

    if total_bacteria > 0:
        clostridia_ratio = clostridia / total_bacteria
        if clostridia_ratio > 0.20:
            clostridia_status = "Good (butyrate support)"
        elif clostridia_ratio > 0.10:
            clostridia_status = "Moderate"
        else:
            clostridia_status = "Low"
    else:
        clostridia_ratio = 0
        clostridia_status = "N/A"

    # Biological warning
    warning = None
    if propionate > acetate:
        warning = "⚠️ BIOLOGICAL WARNING: Propionate exceeds acetate. In healthy guts, acetate is typically the dominant SCFA (≥50% of total). Propionate > acetate may indicate dysbiosis or measurement issue."

    # Prepare data for template
    template_data = {
        'acetate': acetate,
        'propionate': propionate,
        'butyrate': butyrate,
        'lactobacillus': lactobacillus,
        'bifidobacterium': bifidobacterium,
        'clostridia': clostridia,
        'bacteroides': bacteroides,
        'veillonella': veillonella,
        'akkermansia': akkermansia,
        'prediction': type('obj', (object,), prediction_dict)(),  # Simple object for attribute access
        'scfa_level': scfa_level,
        'butyrate_ratio': butyrate_ratio * 100,
        'butyrate_status': butyrate_status,
        'bacteroides_ratio': bacteroides_ratio * 100,
        'bacteroides_status': bacteroides_status,
        'clostridia_ratio': clostridia_ratio * 100,
        'clostridia_status': clostridia_status,
        'warning': warning
    }

    return render_template_string(HTML_TEMPLATE, **template_data)

if __name__ == '__main__':
    # For local testing; on Render, use gunicorn
    app.run(host='0.0.0.0', port=10000)