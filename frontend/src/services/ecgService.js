import api from './api';

export const ecgService = {
  uploadEcg: (patientId, file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('patient_id', patientId);
    formData.append('file', file);
    return api.post('/ecg/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
  },

  getEcgRecord: (ecgId) => api.get(`/ecg/${ecgId}`),

  getPatientEcgHistory: (patientId) => api.get(`/ecg/patient/${patientId}`),

  getEcgImageUrl: (ecgId) => {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
    return `${baseUrl}/ecg/${ecgId}/image`;
  },

  getEcgImageBlob: (ecgId) => api.get(`/ecg/${ecgId}/image`, { responseType: 'blob' }),

  runExplainability: (ecgId, options = {}) => {
    const payload = {};
    if (options.target_class) {
      payload.target_class = options.target_class;
    }
    if (options.integration_steps) {
      payload.integration_steps = options.integration_steps;
    }
    return api.post(`/ecg/${ecgId}/explainability`, payload);
  },
};

export default ecgService;
