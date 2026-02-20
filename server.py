from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import os

# Initialize the Flask app (serve static files from ./static)
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Artifacts paths
MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'feature_columns.pkl'

_model = None
_scaler = None
_feature_columns = None

def load_artifacts():
    global _model, _scaler, _feature_columns
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as f:
            _feature_columns = pickle.load(f)

load_artifacts()


@app.route('/', methods=['GET'])
def health():
    return jsonify({'status':'ok','message':'Server is running. Use POST /predict to get predictions.'})


@app.route('/ui', methods=['GET'])
def ui():
    # Serve the UI page from static/index.html
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/features', methods=['GET'])
def features():
    if _feature_columns is None:
        return jsonify({'error':'feature_columns.pkl not found. Train model with app.py first.'}), 400
    return jsonify({'features': _feature_columns})


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_loaded': _model is not None,
        'scaler_loaded': _scaler is not None,
        'n_features': len(_feature_columns) if _feature_columns is not None else None
    })


def prepare_input(data_records):
    if _feature_columns is None:
        raise ValueError('feature_columns.pkl not available')
    df_raw = pd.DataFrame(data_records)
    df_enc = pd.get_dummies(df_raw, drop_first=True)
    df_reindexed = df_enc.reindex(columns=_feature_columns, fill_value=0)
    return df_reindexed


@app.route('/preview', methods=['POST'])
def preview():
    try:
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            payload = [payload]
        df_prepared = prepare_input(payload)
        return jsonify({'prepared': df_prepared.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    if _model is None or _scaler is None or _feature_columns is None:
        return jsonify({'error':'Model, scaler, or feature list not loaded. Run app.py to train first.'}), 400
    try:
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            payload = [payload]
        df_prepared = prepare_input(payload)
        X_scaled = _scaler.transform(df_prepared)
        preds = _model.predict(X_scaled)
        mapped = ['Yes' if int(p)==1 else 'No' for p in preds]
        return jsonify({'predictions': mapped})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # For quick local dev (use waitress for production on Windows)
    app.run(host='127.0.0.1', port=5000, debug=True)
