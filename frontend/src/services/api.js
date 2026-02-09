import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api', // Adjust if you're not using the proxy for some reason, but relative path is better with proxy
    headers: {
        'Content-Type': 'application/json',
    },
});

export const startTraining = async (config) => {
    const response = await api.post('/training/start', config);
    return response.data;
};

export const stopTraining = async () => {
    const response = await api.post('/training/stop');
    return response.data;
};

export const getTrainingStatus = async () => {
    const response = await api.get('/training/status');
    return response.data;
};

export const predictVQA = async (question, imageFile) => {
    // Ideally we upload the image first or send as multipart
    // For this prototype, we'll send the question and a dummy image ID
    // If we had a real file upload, we'd use FormData

    // 1. Upload file (mock)
    if (imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);
        await api.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    }

    // 2. Predict
    const response = await api.post('/inference/predict', {
        question: question,
        image_id: "uploaded_image_id"
    });
    return response.data;
};

export default api;
