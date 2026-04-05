# DyCDC — Competence-Aware Curriculum Framework for VQA

A curriculum learning framework that trains ViLT-based VQA models on the CLEVR dataset using a dynamic competence-driven training strategy (DyCDC), with an interactive web interface for inference and model comparison.

---

## Quick Start

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at **http://localhost:5173** — requires the backend to be running.

---

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Runs at **http://localhost:8000**

Place model checkpoints in `backend/vilt_models/`:
```
vilt_models/
├── checkpoint_best.pt   # Curriculum model
└── baseline_best.pt     # Baseline model
```

---

### Curriculum Training (Google Colab / SageMaker)

Run these steps in order:

1. **Download CLEVR dataset** — `utils/clevr_downloader.ipynb`
2. **Split into tiers L1–L5** — `utils/tier_division_clevr.ipynb`
3. **Downsample tiers** — `utils/downsampling_clevr.ipynb`
4. **Train curriculum model** — `trainer_sagemaker.ipynb`
5. **Train baseline model** — `baseline_trainer_sagemaker.ipynb`

---

## Project Structure

```
├── backend/              # FastAPI inference server
├── curriculum/
│   ├── data/             # Dataset loaders
│   ├── services/         # Competence tracker, tier sampler, self-paced sampler
│   ├── training/         # Training loop
│   └── utils/            # Data prep notebooks
└── frontend/             # React + Vite web interface
```

---

## Results

| Model      | Overall Accuracy |
|------------|-----------------|
| Baseline   | 71.33%          |
| DyCDC      | 82.65%          |
