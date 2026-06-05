from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch, numpy as np, torch.nn as nn

# ── Arquitectura del modelo (debe ser idéntica a la del notebook) ──────────────
class FinancialLSTM_V2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.2, fc_hidden=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, fc_hidden), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(fc_hidden, output_size)
        )
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.head(out[:, -1, :])

# ── Esquemas Pydantic ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Ventana de 6 meses × n_features (lista aplanada, row-major)
    features: List[List[float]]   # shape: [seq_length, n_features]

class PredictResponse(BaseModel):
    predicted_returns: Dict[str, float]
    portfolio_weights: Dict[str, float]
    expected_portfolio_return: float

# ── Carga del artefacto y reconstrucción del modelo ───────────────────────────
print("Cargando artefacto MacroPulse...")
artifact = torch.load("financial_lstm_v2_checkpoint.pt", map_location="cpu")

bp  = artifact["best_params"]
mdl = FinancialLSTM_V2(
    artifact["input_size"], bp["hidden_size"], bp["num_layers"],
    artifact["output_size"], bp["dropout"], bp["fc_hidden"]
)
mdl.load_state_dict(artifact["model_state_dict"])
mdl.eval()

TICKERS      = artifact["asset_names"]
SCALER_MEAN  = artifact["scaler_mean"]
SCALER_SCALE = artifact["scaler_scale"]
OPT_WEIGHTS  = artifact["optimal_weights"]
SEQ_LEN      = artifact["seq_length"]
N_FEATURES   = artifact["input_size"]
print(f"Modelo listo: {len(TICKERS)} activos | {N_FEATURES} features | seq={SEQ_LEN}")

# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MacroPulse API",
    description="LSTM multi-output para predicción de retornos y optimización de portafolio",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "model": "MacroPulse LSTM v2",
            "assets": TICKERS, "seq_length": SEQ_LEN, "n_features": N_FEATURES}

@app.get("/assets")
def get_assets():
    return {"assets": TICKERS, "optimal_weights": OPT_WEIGHTS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    mat = np.array(req.features, dtype=np.float32)
    if mat.shape != (SEQ_LEN, N_FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f"Se esperaba shape ({SEQ_LEN}, {N_FEATURES}), se recibió {mat.shape}"
        )
    # Normalizar con el scaler del entrenamiento
    mat_scaled = (mat - SCALER_MEAN) / SCALER_SCALE

    tensor = torch.tensor(mat_scaled[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = mdl(tensor).numpy()[0]

    # Desnormalizar solo los retornos (primeras n columnas)
    n = len(TICKERS)
    pred_real = pred_scaled[:n] * SCALER_SCALE[:n] + SCALER_MEAN[:n]

    predicted_returns = {t: float(r) for t, r in zip(TICKERS, pred_real)}
    exp_port_ret = float(sum(pred_real[i] * OPT_WEIGHTS[t] for i, t in enumerate(TICKERS)))

    return PredictResponse(
        predicted_returns=predicted_returns,
        portfolio_weights=OPT_WEIGHTS,
        expected_portfolio_return=exp_port_ret
    )