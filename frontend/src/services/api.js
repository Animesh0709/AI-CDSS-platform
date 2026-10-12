import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: attach JWT token if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: handle 401 unauth
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear token and user if unauthorized
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      if (window.location.pathname !== '/login' && window.location.pathname !== '/register') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const authService = {
  login: (email, password) => api.post('/auth/login', { email, password }),
  register: (userData) => api.post('/auth/register', userData),
  getMe: () => api.get('/auth/me'),
};

export const patientService = {
  getStats: () => api.get('/patients/stats/summary'),
  getPatients: (search = '', skip = 0, limit = 100) => 
    api.get(`/patients?search=${encodeURIComponent(search)}&skip=${skip}&limit=${limit}`),
  getPatientById: (id) => api.get(`/patients/${id}`),
  createPatient: (patientData) => api.post('/patients', patientData),
  updatePatient: (id, patientData) => api.put(`/patients/${id}`, patientData),
  deletePatient: (id) => api.delete(`/patients/${id}`),
  addMedicalHistory: (patientId, historyData) => api.post(`/patients/${patientId}/history`, historyData),
  // Reports
  uploadReport: (patientId, formData) => api.post(`/patients/${patientId}/reports`, formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  listReports: (patientId) => api.get(`/patients/${patientId}/reports`),
  getReport: (patientId, reportId) => api.get(`/patients/${patientId}/reports/${reportId}`),
  extractEntities: (patientId, reportId) => api.post(`/patients/${patientId}/reports/${reportId}/extract`),
  approveCondition: (patientId, reportId, data) => api.post(`/patients/${patientId}/reports/${reportId}/conditions/approve`, data),
  listLabResults: (patientId) => api.get(`/patients/${patientId}/lab-results`),
  addLabResult: (patientId, data) => api.post(`/patients/${patientId}/lab-results`, data),
};

export const progressionService = {
  getModelInfo: () => api.get('/progression/model-info'),
  getEvalReport: () => api.get('/progression/eval-report'),
  predict: (data) => api.post('/progression/predict', data),
  analyzeImagePair: (formData) => api.post('/progression/analyze-image-pair', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  }),
  getHistory: (patientId) => api.get(`/progression/patients/${patientId}/history`),
  confirmOutcome: (studyId, data) => api.post(`/progression/studies/${studyId}/confirm`, data),
};

export const segmentationService = {
  getCases: () => api.get('/segmentation/cases'),
  getEvalReport: () => api.get('/segmentation/eval-report'),
  getMetadata: () => api.get('/segmentation/metadata'),
  predict: (data) => api.post('/segmentation/predict', data, { timeout: 120000 }),
  uploadScan: (formData) => api.post('/segmentation/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  }),
  uploadAndPredict: (formData) => api.post('/segmentation/upload-and-predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  }),
};

export default api;

