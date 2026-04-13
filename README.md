# Diabetic Retinopathy Detector (Desktop)

Tkinter / CustomTkinter desktop app that loads a trained PyTorch classifier to predict diabetic retinopathy from retinal images. Includes lightweight login (JSON or optional MySQL), image upload, and a separate scan-view window for the analyzed image.

## Features
- Retina image classification with a ResNet18 backbone (`classifier.pt` weights).
- Clean UI with login/signup, image uploader, and result dialog.
- “Show plot scan image” button opens a dedicated tab showing the uploaded scan with prediction and confidence.
- User storage via local `users.json`; switches to MySQL if `DB_HOST`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` are set.

## Project structure
- `blindness.py` – GUI entrypoint.
- `model.py` – model build / load / predict helpers.
- `train.py` – training loop (if you want to retrain).
- `inference.py` – CLI inference helper.
- `classifier.pt` – trained weights (expected in project root).
- `config.yaml` – app / training config sample.
- `data/` – place your dataset here (not tracked).
- `artifacts/` – training outputs.
- `requirements.txt` – dependencies.

## Setup (Windows, Python 3.11)
```ps1
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Faster CPU-only install of PyTorch (optional)
If you don’t need GPU, use the lighter wheel:
```ps1
pip uninstall -y torch torchvision
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

## Running the app
```ps1
.\.venv\Scripts\Activate.ps1
python blindness.py
```
Upload a retinal image (jpg/png/tiff). After prediction, use “Show plot scan image” to view the uploaded scan in a separate tabbed window.

## MySQL (optional)
Set environment variables before running:
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
If set, the app stores users in a MySQL table `thegreat`; otherwise it falls back to `users.json`.

## Training / inference via CLI
- Retrain: `python train.py`
- Single image inference: `python inference.py --image path/to/img.jpg`

## GitHub push quickstart
```ps1
git init            # if not already
git add .
git commit -m "Add diabetic retinopathy desktop detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes
- Keep `classifier.pt` in the project root for inference.
- The app prefers GPU if available; otherwise CPU is used automatically.
- Make sure `bg.png` / `bgs.png` exists if you want the background art shown in the UI.

##Datasets
https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data
