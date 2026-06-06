# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** MacroPulse LSTM v2 (`financial_lstm_v2_checkpoint.pt`)
- **Plataforma de despliegue:** [Railway](https://railway.app) — PaaS con soporte Nixpacks y despliegue desde GitHub
- **Requisitos técnicos:**
  - Python 3.10+
  - torch==2.2.2
  - fastapi==0.115.0
  - uvicorn==0.30.6
  - numpy==1.26.4
  - scikit-learn==1.4.2
  - pydantic>=2.0.0
  - RAM mínima recomendada: 1 GB (Railway Pro) debido al peso de PyTorch
- **Requisitos de seguridad:**
  - Variables de entorno sensibles (API keys, tokens) deben configurarse en el panel de Railway, nunca en el repositorio
  - El archivo `.pt` no debe contener datos personales de usuarios
  - Considerar autenticación por API key en `/predict` si la API es pública
- **Diagrama de arquitectura:**

  ```
  Cliente HTTP
      │
      ▼
  Railway (Nixpacks build)
      │
      ├── main.py  (FastAPI + uvicorn)
      │       │
      │       ├── GET  /           → estado del modelo
      │       ├── GET  /assets     → tickers y pesos óptimos
      │       └── POST /predict    → predicción de retornos y portafolio
      │
      └── financial_lstm_v2_checkpoint.pt
              │
              └── FinancialLSTM_V2 (LSTM multi-output)
  ```

---

## Código de despliegue

- **Archivo principal:** `main.py`
- **Rutas de acceso a los archivos:**

  | Archivo | Descripción |
  |---|---|
  | `main.py` | API FastAPI + carga del modelo |
  | `financial_lstm_v2_checkpoint.pt` | Checkpoint PyTorch con pesos, parámetros y escaladores |
  | `requirements.txt` | Dependencias Python |
  | `railway.json` | Configuración de build y arranque en Railway |

- **Claves esperadas en el checkpoint `.pt`:**

  | Clave | Tipo | Descripción |
  |---|---|---|
  | `best_params` | dict | `hidden_size`, `num_layers`, `dropout`, `fc_hidden` |
  | `input_size` | int | Número de features de entrada |
  | `output_size` | int | Número de activos (salidas del modelo) |
  | `model_state_dict` | OrderedDict | Pesos del modelo PyTorch |
  | `asset_names` | list[str] | Tickers de los activos |
  | `scaler_mean` | np.ndarray | Media del escalador de entrenamiento |
  | `scaler_scale` | np.ndarray | Desviación estándar del escalador |
  | `optimal_weights` | dict | Pesos óptimos del portafolio por ticker |
  | `seq_length` | int | Longitud de la ventana temporal |

- **Variables de entorno:** ninguna requerida por defecto. Railway inyecta `$PORT` automáticamente. Añadir en el panel de Railway si se incorpora autenticación u otras integraciones.

---

## Documentación del despliegue

### Instrucciones de instalación

1. Verificar que el checkpoint tiene las claves correctas antes de subir:

   ```python
   import torch
   ck = torch.load("financial_lstm_v2_checkpoint.pt", map_location="cpu")
   print(ck.keys())
   # Esperado: best_params, input_size, output_size, model_state_dict,
   #           asset_names, scaler_mean, scaler_scale, optimal_weights, seq_length
   ```

2. Hacer commit de todos los archivos:

   ```bash
   git add main.py requirements.txt railway.json financial_lstm_v2_checkpoint.pt
   git commit -m "Migrate model loading from pkl to pt checkpoint"
   ```

   > Si el archivo `.pt` supera 100 MB, usar Git LFS:
   > ```bash
   > git lfs install
   > git lfs track "*.pt"
   > git add .gitattributes
   > ```

3. Subir al repositorio remoto:

   ```bash
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

### Instrucciones de configuración

1. Ir a [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Autorizar Railway y seleccionar el repositorio
3. Railway detecta `railway.json` automáticamente; el build usa Nixpacks sin configuración adicional
4. Si se necesitan variables de entorno: ir al servicio → tab **Variables** → agregar clave/valor

El comando de arranque configurado en `railway.json` es:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Instrucciones de uso

Una vez desplegado, Railway asigna un dominio público (p. ej. `macropulse.railway.app`).

**Endpoints disponibles:**

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado de la API, activos disponibles, configuración del modelo |
| `GET` | `/assets` | Lista de tickers y pesos óptimos del portafolio |
| `POST` | `/predict` | Predicción de retornos y retorno esperado del portafolio |

**Ejemplo de llamada a `/predict`:**

```bash
curl -X POST https://TU_DOMINIO.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[...], [...]]   
  }'
```

El campo `features` debe tener shape `[seq_length, n_features]` (lista de listas de floats, ya escalada en la misma escala que el entrenamiento — la API aplica el escalador internamente).

**Respuesta esperada:**

```json
{
  "predicted_returns": {"AAPL": 0.012, "MSFT": 0.008, ...},
  "portfolio_weights": {"AAPL": 0.25, "MSFT": 0.30, ...},
  "expected_portfolio_return": 0.0095
}
```

### Instrucciones de mantenimiento

- **Monitoreo:** revisar el tab **Deployments** en Railway para ver logs en tiempo real. El mensaje `Modelo listo: X activos | Y features | seq=Z` al arranque confirma carga exitosa del `.pt`.
- **Actualizar el modelo:** reemplazar `financial_lstm_v2_checkpoint.pt` en el repositorio y hacer push; Railway redespliega automáticamente.
- **Escalar:** si hay alta demanda o el modelo requiere más RAM, cambiar al plan Pro de Railway (más memoria y CPU).
- **Troubleshooting frecuente:**

  | Error | Causa probable | Solución |
  |---|---|---|
  | `KeyError: 'best_params'` | Clave distinta en el `.pt` | Ajustar la clave en `main.py` |
  | `RuntimeError: size mismatch` | Arquitectura no coincide con el checkpoint | Verificar parámetros de `FinancialLSTM_V2` |
  | OOM / build lento | PyTorch usa mucha RAM | Usar plan Pro o `torch==2.2.2+cpu` |
  | Archivo `.pt` rechazado por GitHub | Archivo >100 MB | Configurar Git LFS |
  | `422 Unprocessable Entity` | Shape de features incorrecto | Verificar que `features` tenga shape `[seq_length, n_features]` |
