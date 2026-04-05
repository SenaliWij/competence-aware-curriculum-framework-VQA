import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    headers: { 'Content-Type': 'application/json' },
});

// Model APIs used by Models repository
export const getModelDownloadUrl = (modelId) =>
    `${api.defaults.baseURL}/models/${modelId}/download`;

// Inference APIs

// Upload the image & run single-model inference on proposed curriculum model
export const predictVQA = async (question, imageFile) => {
    const imagePath = await _uploadImage(imageFile);

    const res = await api.post('/inference/predict', {
        question,
        image_path: imagePath,
        model_id: 'vilt_curriculum',
    });

    return res.data;
};

// Upload the image & run side by side comparison on proposed vs baseline.
export const compareVQA = async (question, imageFile) => {
    const imagePath = await _uploadImage(imageFile);

    const res = await api.post('/inference/compare', {
        question,
        image_path: imagePath,
    });

    return res.data;
};

// Helper function to upload the file & returns the server-side path.
async function _uploadImage(imageFile) {
    if (!imageFile) return null;

    const formData = new FormData();
    formData.append('file', imageFile);
    const uploadRes = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return uploadRes.data.image_path;
}

export default api;
