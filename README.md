# Proyecto Fase II — Clasificación inteligente de tickets

Continúa el baseline de Fase I: TF-IDF (1,2) + regresión logística balanceada sobre el dataset Kaggle `tobiasbueck/multilingual-customer-support-tickets`.

## Requisitos
Python 3.11, Docker Desktop, acceso a internet para descargar el dataset al entrenar. Los resultados reales pueden variar por versión del dataset y dependencias.

## Entrenar y exportar el modelo
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_export.py
pytest -q
```
**Importante:** el modelo `.joblib` no se incluye inicialmente; se genera al entrenar. No ejecutar `docker build` antes de generarlo. El archivo de modelo debe proceder de una fuente confiable.

## Ejecutar Docker
```bash
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"subject":"Password reset","body":"I cannot access my account"}'
```
En Windows PowerShell, utilizar `Invoke-RestMethod` en lugar de `curl` cuando sea necesario.

Frontend: http://localhost:8000/ · API: http://localhost:8000/docs

Detener: `docker compose down`.

## Comprobaciones
- `/health`: servicio activo y disponibilidad del archivo de modelo.
- `/predict`: clase estimada, confianza y señal de revisión humana.
- Texto vacío o faltante: HTTP 422.
- Modelo ausente: HTTP 503.
- Umbral de revisión: 0.50 (configurable).

## Límites
La confianza es una probabilidad del clasificador, no una garantía de acierto ni necesariamente calibrada. No se debe presentar la precisión sobre casos aceptados como exactitud global. El modelo clasifica cola/departamento; **no predice prioridad** en esta versión.

## Publicación
Crear un repositorio vacío en GitHub, ejecutar `git init`, `git add .`, `git commit -m "Fase II"`, configurar el remoto y hacer `git push`. No subir datos sensibles, secretos ni tickets reales.
