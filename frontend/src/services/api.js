import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    headers: { 'Content-Type': 'application/json' },
});

// ── Models ───────────────────────────────────────────────────────────────────

export const getModels = async () => {
    const res = await api.get('/models');
    return res.data;
};

export const getModel = async (modelId) => {
    const res = await api.get(`/models/${modelId}`);
    return res.data;
};

export const getModelDownloadUrl = (modelId) =>
    `${api.defaults.baseURL}/models/${modelId}/download`;

// ── Inference ────────────────────────────────────────────────────────────────

export const predictVQA = async (question, imageFile, modelId = 'vilt_curriculum') => {
    let imagePath = null;

    if (imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);
        const uploadRes = await api.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        imagePath = uploadRes.data.image_path;
    }

    const inferenceRes = await api.post('/inference/predict', {
        question,
        image_path: imagePath,
        model_id: modelId,
    });

    return inferenceRes.data;
};

export default api;
