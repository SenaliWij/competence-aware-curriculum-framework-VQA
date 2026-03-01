import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api', 
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
    let imagePath = null;

    console.log(imageFile)
    // Upload image
    if (imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);
        console.log(formData)

        const uploadRes = await api.post('/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        console.log(uploadRes)

        imagePath = uploadRes.data.image_path;
    }

    // Call inference with image_path
    const inferenceRes = await api.post('/inference/predict', {
        question,
        image_path: imagePath
    });

    return inferenceRes.data;
};

export default api;
