import json
import torch
import numpy as np
import joblib
import argparse

def load_best_model_info(path="models/best_batch.json"):
    with open(path, 'r') as f:
        info = json.load(f)
    return info

def load_model_and_scaler(model_path, scaler_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    scaler = joblib.load(scaler_path)
    return model, scaler

def infer(features, model, scaler, threshold=0.5):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.sigmoid(logits).item()
    decision = int(prob > threshold)
    return {"prob": prob, "decision": decision}

def main():
    parser = argparse.ArgumentParser(description="Live inference for best batch model.")
    parser.add_argument('--features', nargs='+', type=float, required=True, help='Feature vector for inference')
    parser.add_argument('--best-info', default='models/best_batch.json', help='Path to best_batch.json')
    args = parser.parse_args()
    info = load_best_model_info(args.best_info)
    # Load model (assume torch.load returns nn.Module)
    model = torch.load(info['model'], map_location=torch.device('cpu'))
    scaler = joblib.load(info['scaler'])
    with open(info['threshold'], 'r') as f:
        threshold = json.load(f).get('best_threshold', 0.5)
    result = infer(args.features, model, scaler, threshold)
    print(f"P(class=1): {result['prob']:.4f}, Decision: {result['decision']}")

if __name__ == "__main__":
    main() 