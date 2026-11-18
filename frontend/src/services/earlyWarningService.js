/**
 * Early Warning System API service
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

const api = axios.create({ baseURL: API_BASE });
api.interceptors.request.use(cfg => {
  cfg.headers = { ...cfg.headers, ...getAuthHeaders() };
  return cfg;
});

/**
 * Record a new set of vital signs for a patient.
 * @param {Object} vitals - VitalSignCreate payload
 */
export const recordVitals = async (vitals) => {
  const { data } = await api.post('/early-warning/vitals', vitals);
  return data;
};

/**
 * Retrieve time-series vitals for a patient.
 * @param {number} patientId
 * @param {number} limit
 */
export const getPatientVitals = async (patientId, limit = 50) => {
  const { data } = await api.get(`/early-warning/patients/${patientId}/vitals`, {
    params: { limit },
  });
  return data;
};

/**
 * Run ML early-warning risk prediction for a patient.
 * @param {number} patientId
 */
export const predictEWS = async (patientId) => {
  const { data } = await api.post(`/early-warning/patients/${patientId}/predict`);
  return data;
};

/**
 * Get EWS assessment history for a patient.
 * @param {number} patientId
 * @param {number} limit
 */
export const getEWSHistory = async (patientId, limit = 20) => {
  const { data } = await api.get(`/early-warning/patients/${patientId}/history`, {
    params: { limit },
  });
  return data;
};

/**
 * Get unacknowledged alert queue for the logged-in doctor.
 * @param {boolean} includeAcknowledged
 */
export const getAlerts = async (includeAcknowledged = false, sortBy = 'risk', riskLevel = null) => {
  const { data } = await api.get('/early-warning/alerts', {
    params: {
      include_acknowledged: includeAcknowledged,
      sort_by: sortBy,
      ...(riskLevel ? { risk_level: riskLevel } : {}),
    },
  });
  return data;
};

/**
 * Bulk-import vital signs from a CSV file.
 */
export const importVitalsCsv = async (patientId, file) => {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/early-warning/vitals/csv', form, {
    params: { patient_id: patientId },
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

/**
 * Doctor acknowledges an EWS alert.
 * @param {number} assessmentId
 * @param {string|null} notes
 */
export const acknowledgeAlert = async (assessmentId, notes = null) => {
  const { data } = await api.post(`/early-warning/assessments/${assessmentId}/acknowledge`, {
    acknowledge_notes: notes,
  });
  return data;
};
