# Live-Spoofing-Detection

Repo minimal pour **fine-tuner un modèle ImageNet** (ResNet18) sur la classification visage:
- `spoof`
- `non-spoof`

Le repo inclut aussi une **petite interface Streamlit** pour tester facilement le modèle.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure attendue du dataset

Le script d'entraînement utilise `torchvision.datasets.ImageFolder`:

```text
data/
  train/
    spoof/
    non-spoof/
  val/
    spoof/
    non-spoof/
```

## Entraînement (fine-tuning ImageNet)

```bash
python train.py --data-dir data --epochs 5 --output checkpoints/best.pt
```

Options utiles:
- `--freeze-backbone` : n'entraîne que la tête de classification
- `--no-pretrained` : désactive les poids ImageNet

## Inference CLI

```bash
python infer.py --checkpoint checkpoints/best.pt --image path/to/face.jpg
```

## Interface Streamlit

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Ensuite, chargez une image visage depuis l'UI pour obtenir:
- la classe prédite
- le score de confiance
- les probabilités par classe

## Test rapide

```bash
python -m unittest tests/test_spoof_model.py
```
