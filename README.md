# Cats vs Dogs MLOps Pipeline

End-to-end baseline MLOps implementation for Assignment 2:
- Model training + experiment tracking (MLflow)
- Artifact packaging and FastAPI inference service
- Docker image build and registry publishing
- GitHub Actions CI/CD with post-deploy smoke tests
- DVC pipeline definition for reproducible training

## Contributors

- ARYAMANN SINGH - 2024aa05025
- ANANTHAN P   - 2024aa05692
- BALAJI R  - 2024aa05844
- BALSURE ANIKET K  - 2024aa05296
- SAURAV BANSAL - 2023aa05710

## 1) Project structure

```text
src/
  data.py
  model.py
  train.py
  service/main.py
tests/
scripts/
.github/workflows/
dvc.yaml
params.yaml
Dockerfile
docker-compose.yml
```

## 2) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Dataset layout

Prepare preprocessed images in the following structure:

```text
data/processed/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
  test/
    cat/
    dog/
```

Images should be 224x224 RGB. You can use Kaggle Cats vs Dogs and split 80/10/10.

## 4) Train + track experiments

```bash
python -m src.train --data-dir data/processed --epochs 3 --batch-size 32 --learning-rate 0.001 --output-dir artifacts
mlflow ui --backend-store-uri ./mlruns
```

Outputs:
- `artifacts/model.pt`
- `artifacts/labels.json`
- `artifacts/confusion_matrix.png`
- `artifacts/loss_curve.png`

## 5) Run inference locally

```bash
uvicorn src.service.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@tests/assets/sample.png"
```

## 6) Docker

Build and run:

```bash
docker build -t cats-dogs-api:local .
docker run --rm -p 8000:8000 cats-dogs-api:local
```

## 7) CI/CD (GitHub Actions)

### CI (`.github/workflows/ci.yml`)
- Runs on push + PR
- Installs dependencies
- Executes unit tests (`pytest`)
- Builds Docker image
- Pushes to GHCR on push events

### CD (`.github/workflows/cd.yml`)
- Triggers after CI success on `main`
- Runs on a `self-hosted` GitHub Actions runner
- Pulls latest image from GHCR and deploys with Docker Compose
- Runs `scripts/smoke_test.sh`; fails pipeline if health/predict checks fail

## 8) Notes for GitHub setup

1. Ensure GitHub Packages permissions are enabled for GHCR.
2. Add a self-hosted runner on deployment host with Docker + Compose installed.
3. On deployment host, run workflows from this repository path.
4. Make sure model artifacts are available in the image before deployment.

## 9) DVC

Pipeline stage is declared in `dvc.yaml`. After installing DVC:

```bash
pip install dvc
# optional once per repo
# dvc init

dvc repro
```
