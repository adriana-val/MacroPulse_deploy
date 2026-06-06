import os
import json
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn

# ── Configuración por entorno (con valores por defecto seguros) ────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "financial_lstm_v2_checkpoint.pt")
META_PATH = os.getenv("MODEL_META_PATH", "metadata.json")
# El checkpoint NO almacena seq_length; se asume una ventana de 6 pasos (6 meses)
# salvo que se indique lo contrario por variable de entorno o metadata.json.
DEFAULT_SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))


# ── Arquitectura del modelo (debe ser idéntica a la del notebook) ──────────────
class FinancialLSTM_V2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.2, fc_hidden=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
    # Ventana temporal × n_features (matriz, row-major)
    features: List[List[float]]   # shape: [seq_length, n_features]


class PredictResponse(BaseModel):
    predicted_returns: Dict[str, float]
    portfolio_weights: Dict[str, float]
    expected_portfolio_return: float
    scaler_applied: bool
    weights_source: str


# ── Carga del artefacto y reconstrucción del modelo ───────────────────────────
print(f"Cargando artefacto MacroPulse desde '{MODEL_PATH}'...")
# weights_only=False es necesario porque el checkpoint es un dict con metadatos.
artifact = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

bp = artifact["best_params"]
N_FEATURES = int(artifact["input_size"])     # 37
N_ASSETS = int(artifact["output_size"])      # 9

mdl = FinancialLSTM_V2(
    N_FEATURES, bp["hidden_size"], bp["num_layers"],
    N_ASSETS, bp["dropout"], bp["fc_hidden"]
)
mdl.load_state_dict(artifact["model_state_dict"])
mdl.eval()


# ── Metadatos de inferencia (no incluidos en el checkpoint) ───────────────────
# El checkpoint guardado solo contiene pesos y configuración de arquitectura.
# Los artefactos de inferencia (scaler, nombres de activos, pesos de portafolio,
# longitud de ventana) se cargan de forma opcional desde metadata.json o
# variables de entorno; si no existen, se usan valores por defecto razonables
# para garantizar que la API arranque y responda en Railway.
def _load_sidecar_meta(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            print(f"Metadata de inferencia cargada desde '{path}'.")
            return meta
        except Exception as exc:  # defensivo: nunca debe tumbar el arranque
            print(f"AVISO: no se pudo leer '{path}': {exc}")
    return {}


_meta = _load_sidecar_meta(META_PATH)


def _meta_get(key: str, default):
    """Prioridad: metadata.json -> artefacto .pt -> valor por defecto."""
    if key in _meta and _meta[key] is not None:
        return _meta[key]
    val = artifact.get(key) if isinstance(artifact, dict) else None
    return default if val is None else val


# Nombres de activos
TICKERS = _meta_get("asset_names", [f"ASSET_{i}" for i in range(N_ASSETS)])
if not isinstance(TICKERS, list) or len(TICKERS) != N_ASSETS:
    print("AVISO: 'asset_names' ausente o inconsistente; usando nombres genéricos.")
    TICKERS = [f"ASSET_{i}" for i in range(N_ASSETS)]

# Longitud de la ventana temporal
SEQ_LEN = int(_meta_get("seq_length", DEFAULT_SEQ_LEN))

# Scaler (estandarización de features). Si no está disponible se usa identidad
# (media 0, escala 1), de modo que la API funciona sin transformar las entradas.
_sm = _meta_get("scaler_mean", None)
_ss = _meta_get("scaler_scale", None)
HAS_SCALER = _sm is not None and _ss is not None
if HAS_SCALER:
    SCALER_MEAN = np.asarray(_sm, dtype=np.float32)
    SCALER_SCALE = np.asarray(_ss, dtype=np.float32)
    # Evitar divisiones por cero
    SCALER_SCALE = np.where(SCALER_SCALE == 0, 1.0, SCALER_SCALE)
else:
    print("AVISO: scaler no disponible; se usará identidad (sin normalización).")
    SCALER_MEAN = np.zeros(N_FEATURES, dtype=np.float32)
    SCALER_SCALE = np.ones(N_FEATURES, dtype=np.float32)

# Pesos óptimos de portafolio. Si no están, se reparte de forma equiponderada.
_ow = _meta_get("optimal_weights", None)
if isinstance(_ow, dict) and len(_ow) == N_ASSETS:
    OPT_WEIGHTS = {t: float(_ow.get(t, 0.0)) for t in TICKERS}
    WEIGHTS_SOURCE = "artifact"
else:
    print("AVISO: 'optimal_weights' ausente; usando pesos equiponderados (1/N).")
    OPT_WEIGHTS = {t: 1.0 / N_ASSETS for t in TICKERS}
    WEIGHTS_SOURCE = "equal_weight_fallback"

print(
    f"Modelo listo: {N_ASSETS} activos | {N_FEATURES} features | seq={SEQ_LEN} "
    f"| scaler={'sí' if HAS_SCALER else 'identidad'} | pesos={WEIGHTS_SOURCE}"
)


# ── App FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MacroPulse API",
    description="LSTM multi-output para predicción de retornos y optimización de portafolio",
    version="1.1.0"
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "MacroPulse LSTM v2",
        "assets": TICKERS,
        "seq_length": SEQ_LEN,
        "n_features": N_FEATURES,
        "scaler_applied": HAS_SCALER,
        "weights_source": WEIGHTS_SOURCE,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/assets")
def get_assets():
    return {"assets": TICKERS, "optimal_weights": OPT_WEIGHTS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    mat = np.array(req.features, dtype=np.float32)

    # Validación: la dimensión de features debe coincidir exactamente.
    if mat.ndim != 2 or mat.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperaba una matriz [seq_length, {N_FEATURES}]; "
                f"se recibió {tuple(mat.shape)}."
            ),
        )
    if mat.shape[0] < 1:
        raise HTTPException(
            status_code=422,
            detail="La ventana temporal debe contener al menos un paso.",
        )

    # Normalizar con el scaler del entrenamiento (identidad si no está disponible).
    mat_scaled = (mat - SCALER_MEAN) / SCALER_SCALE

    tensor = torch.tensor(mat_scaled[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = mdl(tensor).numpy()[0]

    # Desnormalizar los retornos predichos (se asume que las primeras N_ASSETS
    # columnas del scaler corresponden a los retornos de los activos).
    pred_real = pred_scaled[:N_ASSETS] * SCALER_SCALE[:N_ASSETS] + SCALER_MEAN[:N_ASSETS]

    predicted_returns = {t: float(r) for t, r in zip(TICKERS, pred_real)}
    exp_port_ret = float(
        sum(pred_real[i] * OPT_WEIGHTS[t] for i, t in enumerate(TICKERS))
    )

    return PredictResponse(
        predicted_returns=predicted_returns,
        portfolio_weights=OPT_WEIGHTS,
        expected_portfolio_return=exp_port_ret,
        scaler_applied=HAS_SCALER,
        weights_source=WEIGHTS_SOURCE,
    )