# DyCDC - Competence-Aware Curriculum Framework for VQA

DyCDC (Dynamic Competence-Driven Curriculum) is an advanced curriculum learning framework designed to train Vision-and-Language Transformer (ViLT) models for Visual Question Answering (VQA). Unlike traditional training methods that present data randomly, DyCDC dynamically adapts the training data complexity based on the model's continuously assessed competence levels. Evaluated on the CLEVR dataset, this approach significantly accelerates convergence and improves reasoning capabilities. The project also features a modern, interactive React-based web interface and a FastAPI backend, enabling users to seamlessly perform model inference, visualize results, and compare the DyCDC model's performance directly against a standard baseline model.

---

## Quick Start

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at **http://localhost:3000** — requires the backend to be running.

---

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Runs at **http://localhost:8000**

---

### Curriculum Training (Google Colab / SageMaker)

To train the models from scratch, follow these data preparation steps first in order:

1. **Download CLEVR dataset** — Run `utils/clevr_downloader.ipynb`
2. **Split into tiers L1–L5** — Run `utils/tier_division_clevr.ipynb`
3. **Downsample tiers** — Run `utils/downsampling_clevr.ipynb`

Once the dataset is prepared, you can train your models. 
- **For Curriculum Training (DyCDC):** Please run the `trainer_sagemaker.ipynb` notebook.
- **For Baseline Training:** Please run the `baseline_trainer_sagemaker.ipynb` notebook.

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

---

*Note: AI was used to assist with the competence power law and Self-Paced Learning (SPL) calculations from a mathematical perspective.*
