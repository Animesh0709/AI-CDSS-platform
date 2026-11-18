import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import {
  AlertTriangle, Activity, Heart, Wind, Thermometer, Droplets,
  Eye, CheckCircle2, RefreshCw, Plus,
  Clock, User, Shield, TrendingUp,
  Bell, BellOff, Info, Zap, Loader2, Upload, FileDown,
} from 'lucide-react';
import {
  recordVitals,
  getPatientVitals,
  predictEWS,
  getEWSHistory,
  getAlerts,
  acknowledgeAlert,
  importVitalsCsv,
} from '../services/earlyWarningService';
import axios from 'axios';

// ─── helpers ──────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const getAuthHeaders = () => {
  const t = localStorage.getItem('token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};

const fetchPatients = async () => {
  const { data } = await axios.get(`${API_BASE}/patients`, { headers: getAuthHeaders() });
  return data;
};

// ─── Risk colours ──────────────────────────────────────────────────────────────
const RISK_CONFIG = {
  Low:      { bg: 'bg-emerald-50',  border: 'border-emerald-400', text: 'text-emerald-700',  badge: 'bg-emerald-100 text-emerald-800',  dot: 'bg-emerald-500', barColor: '#10b981' },
  Medium:   { bg: 'bg-amber-50',    border: 'border-amber-400',   text: 'text-amber-700',    badge: 'bg-amber-100 text-amber-800',     dot: 'bg-amber-500',   barColor: '#f59e0b' },
  High:     { bg: 'bg-orange-50',   border: 'border-orange-400',  text: 'text-orange-700',   badge: 'bg-orange-100 text-orange-800',   dot: 'bg-orange-500',  barColor: '#f97316' },
  Critical: { bg: 'bg-red-50',      border: 'border-red-500',     text: 'text-red-700',      badge: 'bg-red-100 text-red-800',         dot: 'bg-red-600',     barColor: '#ef4444' },
  default:  { bg: 'bg-slate-50',    border: 'border-slate-300',   text: 'text-slate-600',    badge: 'bg-slate-100 text-slate-700',     dot: 'bg-slate-400',   barColor: '#94a3b8' },
};
const getRisk = (label) => RISK_CONFIG[label] || RISK_CONFIG.default;

function exportAssessmentPdf(patient, prediction, vitals) {
  const latest = vitals[vitals.length - 1] || {};
  const shapRows = (prediction.shap_features || []).map(f =>
    `<tr><td>${f.display_name}</td><td>${f.value}</td><td>${f.direction}</td><td>${f.shap_value}</td></tr>`
  ).join('');
  const html = `<!DOCTYPE html><html><head><title>EWS Report</title>
    <style>
      body { font-family: Georgia, serif; padding: 32px; color: #1e293b; }
      h1 { font-size: 20px; margin-bottom: 4px; }
      .muted { color: #64748b; font-size: 12px; }
      table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
      th, td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }
      th { background: #f8fafc; }
      .box { border: 1px solid #cbd5e1; padding: 12px; margin-top: 16px; }
    </style></head><body>
    <h1>Early Warning System — Research Report</h1>
    <p class="muted">Not a diagnosis. Clinician review required.</p>
    <div class="box">
      <p><strong>Patient:</strong> ${patient.first_name} ${patient.last_name}</p>
      <p><strong>Assessed:</strong> ${new Date(prediction.assessment_dt).toLocaleString()}</p>
      <p><strong>Target:</strong> ${prediction.prediction_target || 'Deterioration risk'}</p>
      <p><strong>Horizon:</strong> ${prediction.prediction_horizon || '—'}</p>
      <p><strong>ML risk:</strong> ${prediction.ml_risk_label} (${Math.round((prediction.ml_risk_score || 0) * 100)}%)</p>
      <p><strong>Model used:</strong> ${prediction.model_used || '—'}</p>
      <p><strong>NEWS2:</strong> ${prediction.news2_score ?? '—'} (${prediction.news2_risk_level || '—'})</p>
      <p><strong>Recommendations:</strong> ${prediction.news2_recommendations || '—'}</p>
      <p><strong>Latest vitals:</strong> HR ${latest.heart_rate ?? '—'}, RR ${latest.respiratory_rate ?? '—'},
         SpO₂ ${latest.spo2 ?? '—'}, BP ${latest.systolic_bp ?? '—'}/${latest.diastolic_bp ?? '—'},
         Temp ${latest.temperature_c ?? '—'}</p>
    </div>
    <h2>Feature contributions</h2>
    <table><thead><tr><th>Feature</th><th>Value</th><th>Direction</th><th>SHAP</th></tr></thead>
    <tbody>${shapRows || '<tr><td colspan="4">None</td></tr>'}</tbody></table>
    <script>window.onload = function () { window.print(); }</script>
    </body></html>`;
  const win = window.open('', '_blank', 'noopener,noreferrer');
  if (!win) return;
  win.document.write(html);
  win.document.close();
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function RiskBadge({ label, score }) {
  const c = getRisk(label);
  const pct = score != null ? Math.round(score * 100) : null;
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold ${c.badge}`}>
      <span className={`w-2 h-2 rounded-full ${c.dot} animate-pulse`} />
      {label}{pct != null ? ` (${pct}%)` : ''}
    </span>
  );
}

function NEWS2Badge({ score, level }) {
  const c = getRisk(level);
  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border-2 ${c.border} ${c.bg}`}>
      <div className={`text-3xl font-black ${c.text}`}>{score ?? '—'}</div>
      <div>
        <div className="text-xs text-slate-500 font-medium">NEWS2 Score</div>
        <div className={`text-sm font-bold ${c.text}`}>{level || '—'}</div>
      </div>
    </div>
  );
}

function VitalInput({ label, id, unit, value, onChange, min, max, step = 'any', icon: Icon }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-xs font-semibold text-slate-500 flex items-center gap-1">
        {Icon && <Icon className="w-3.5 h-3.5" />} {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type="number"
          step={step}
          min={min}
          max={max}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder="—"
          className="w-full px-3 py-2 pr-10 text-sm border border-slate-200 rounded-lg bg-white
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
                     placeholder:text-slate-300"
        />
        {unit && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 pointer-events-none">
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

function ShapWaterfall({ features }) {
  if (!features || features.length === 0) return null;
  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 0.001);
  return (
    <div className="space-y-2">
      {features.map((f, i) => {
        const isPositive = f.shap_value > 0;
        const width = `${Math.round((Math.abs(f.shap_value) / maxAbs) * 100)}%`;
        return (
          <div key={i} className="flex items-center gap-3">
            <div className="w-44 text-xs text-slate-600 text-right truncate shrink-0" title={f.display_name}>
              {f.display_name}
            </div>
            <div className="flex-1 h-6 bg-slate-100 rounded relative overflow-hidden">
              <div
                className={`h-full rounded transition-all duration-700 ${isPositive ? 'bg-red-400' : 'bg-emerald-400'}`}
                style={{ width }}
              />
            </div>
            <div className={`w-20 text-xs font-medium shrink-0 ${isPositive ? 'text-red-600' : 'text-emerald-600'}`}>
              {f.direction} <span className="opacity-70">({f.value})</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function RiskGauge({ score }) {
  const pct = Math.round((score || 0) * 100);
  const angle = -135 + (pct / 100) * 270;
  const color = pct < 25 ? '#10b981' : pct < 50 ? '#f59e0b' : pct < 75 ? '#f97316' : '#ef4444';
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-32 h-20 overflow-hidden">
        <svg viewBox="0 0 200 120" className="w-full h-full">
          <path d="M 20 110 A 90 90 0 0 1 180 110" fill="none" stroke="#e2e8f0" strokeWidth="18" strokeLinecap="round" />
          <path d="M 20 110 A 90 90 0 0 1 180 110" fill="none" stroke={color} strokeWidth="18"
            strokeLinecap="round"
            strokeDasharray={`${(pct / 100) * 283} 283`}
            style={{ transition: 'stroke-dasharray 0.8s ease, stroke 0.4s' }}
          />
          <line
            x1="100" y1="110"
            x2={100 + 60 * Math.cos((angle * Math.PI) / 180)}
            y2={110 + 60 * Math.sin((angle * Math.PI) / 180)}
            stroke={color} strokeWidth="4" strokeLinecap="round"
            style={{ transition: 'all 0.8s ease' }}
          />
          <circle cx="100" cy="110" r="6" fill={color} />
        </svg>
      </div>
      <div className="text-center">
        <div className="text-3xl font-black" style={{ color }}>{pct}%</div>
        <div className="text-xs text-slate-500">Risk Score</div>
      </div>
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────────
export default function EarlyWarning() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [vitals, setVitals] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [tab, setTab] = useState('record'); // record | trends | predict | alerts
  const [loading, setLoading] = useState({ patients: false, vitals: false, predict: false, record: false, alerts: false });
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [acknowledgeModal, setAcknowledgeModal] = useState(null); // {id, notes}
  const [showAcknowledgedAlerts, setShowAcknowledgedAlerts] = useState(false);
  const [alertSort, setAlertSort] = useState('risk');
  const [alertRiskFilter, setAlertRiskFilter] = useState('');
  const [csvReport, setCsvReport] = useState(null);

  // Vital form state
  const [form, setForm] = useState({
    heart_rate: '', respiratory_rate: '', systolic_bp: '', diastolic_bp: '',
    spo2: '', temperature_c: '', consciousness_level: 'A', supplemental_o2: false,
    urine_output_ml_hr: '', notes: '',
  });

  const setLoad = (key, val) => setLoading(prev => ({ ...prev, [key]: val }));

  // Load patients
  useEffect(() => {
    setLoad('patients', true);
    fetchPatients()
      .then(setPatients)
      .catch(() => setError('Failed to load patients.'))
      .finally(() => setLoad('patients', false));
  }, []);

  // Load alerts
  const loadAlerts = useCallback(() => {
    setLoad('alerts', true);
    getAlerts(showAcknowledgedAlerts, alertSort, alertRiskFilter || null)
      .then(setAlerts)
      .catch(() => {})
      .finally(() => setLoad('alerts', false));
  }, [showAcknowledgedAlerts, alertSort, alertRiskFilter]);

  useEffect(() => { loadAlerts(); }, [loadAlerts]);

  // Load vitals when patient changes
  useEffect(() => {
    if (!selectedPatient) return;
    setLoad('vitals', true);
    Promise.all([
      getPatientVitals(selectedPatient.id),
      getEWSHistory(selectedPatient.id),
    ])
      .then(([v, h]) => { setVitals(v); setHistory(h); })
      .catch(() => setError('Failed to load patient data.'))
      .finally(() => setLoad('vitals', false));
  }, [selectedPatient]);

  const handleFormChange = (field, value) => setForm(prev => ({ ...prev, [field]: value }));

  const handleRecordVitals = async () => {
    if (!selectedPatient) { setError('Please select a patient.'); return; }
    setLoad('record', true);
    setError(null);
    try {
      const payload = {
        patient_id: selectedPatient.id,
        heart_rate: form.heart_rate ? parseFloat(form.heart_rate) : null,
        respiratory_rate: form.respiratory_rate ? parseFloat(form.respiratory_rate) : null,
        systolic_bp: form.systolic_bp ? parseFloat(form.systolic_bp) : null,
        diastolic_bp: form.diastolic_bp ? parseFloat(form.diastolic_bp) : null,
        spo2: form.spo2 ? parseFloat(form.spo2) : null,
        temperature_c: form.temperature_c ? parseFloat(form.temperature_c) : null,
        consciousness_level: form.consciousness_level,
        supplemental_o2: form.supplemental_o2,
        urine_output_ml_hr: form.urine_output_ml_hr ? parseFloat(form.urine_output_ml_hr) : null,
        notes: form.notes || null,
      };
      const newVital = await recordVitals(payload);
      setVitals(prev => [...prev, newVital]);
      setSuccess(`Vitals recorded. NEWS2: ${newVital.news2_score ?? '—'} (${newVital.news2_risk_level ?? '—'})`);
      setForm(prev => ({ ...prev, heart_rate: '', respiratory_rate: '', systolic_bp: '', diastolic_bp: '', spo2: '', temperature_c: '', urine_output_ml_hr: '', notes: '' }));
      setTimeout(() => setSuccess(null), 5000);
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to record vitals.');
    } finally {
      setLoad('record', false);
    }
  };

  const handlePredict = async () => {
    if (!selectedPatient) { setError('Select a patient first.'); return; }
    setLoad('predict', true);
    setError(null);
    try {
      const result = await predictEWS(selectedPatient.id);
      setPrediction(result);
      setHistory(prev => [{ ...result, id: result.assessment_id }, ...prev.slice(0, 19)]);
      loadAlerts();
      setTab('predict');
    } catch (e) {
      setError(e.response?.data?.detail || 'Prediction failed. Record vitals first.');
    } finally {
      setLoad('predict', false);
    }
  };

  const handleCsvImport = async (file) => {
    if (!selectedPatient) { setError('Please select a patient.'); return; }
    if (!file) return;
    setLoad('record', true);
    setError(null);
    setCsvReport(null);
    try {
      const report = await importVitalsCsv(selectedPatient.id, file);
      setCsvReport(report);
      const refreshed = await getPatientVitals(selectedPatient.id);
      setVitals(refreshed);
      if (report.imported_count > 0) {
        setSuccess(`Imported ${report.imported_count} vital-sign row(s). Missing values were left blank.`);
        setTimeout(() => setSuccess(null), 5000);
      } else {
        setError(report.errors?.[0] || 'No rows were imported.');
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'CSV import failed.');
    } finally {
      setLoad('record', false);
    }
  };

  const handleAcknowledge = async () => {
    if (!acknowledgeModal) return;
    try {
      await acknowledgeAlert(acknowledgeModal.id, acknowledgeModal.notes);
      setAlerts(prev => prev.filter(a => a.assessment_id !== acknowledgeModal.id));
      setAcknowledgeModal(null);
      setSuccess('Alert acknowledged.');
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      setError('Failed to acknowledge alert.');
    }
  };

  // Trend chart data
  const trendData = vitals.map(v => ({
    time: new Date(v.recorded_at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
    'Heart Rate': v.heart_rate,
    'SpO₂': v.spo2,
    'Resp. Rate': v.respiratory_rate,
    'Sys. BP': v.systolic_bp,
    'Temp (°C)': v.temperature_c,
    'NEWS2': v.news2_score,
  }));

  const unackCount = alerts.filter(a => !a.doctor_acknowledged).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-4 sm:p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-red-200">
            <AlertTriangle className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black text-slate-800">Early Warning System</h1>
            <p className="text-sm text-slate-500">Explainable time-series deterioration risk · NEWS2 + ML</p>
          </div>
          {unackCount > 0 && (
            <div className="ml-auto flex items-center gap-2 bg-red-100 border border-red-300 text-red-700 px-3 py-1.5 rounded-full text-sm font-bold animate-pulse cursor-pointer" onClick={() => setTab('alerts')}>
              <Bell className="w-4 h-4" />
              {unackCount} Active Alert{unackCount > 1 ? 's' : ''}
            </div>
          )}
        </div>
        <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2">
          <Shield className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-700">
            <strong>Research prototype</strong> — for clinical decision support only. Not a substitute for clinical judgement. Always verify with full clinical assessment.
          </p>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div className="mb-4 flex items-center gap-2 bg-red-50 border border-red-300 text-red-700 px-4 py-3 rounded-xl text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
          <button className="ml-auto" onClick={() => setError(null)}>✕</button>
        </div>
      )}
      {success && (
        <div className="mb-4 flex items-center gap-2 bg-emerald-50 border border-emerald-300 text-emerald-700 px-4 py-3 rounded-xl text-sm">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          {success}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* LEFT — Patient selector + quick stats */}
        <div className="lg:col-span-1 space-y-4">
          {/* Patient Picker */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4">
            <h2 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
              <User className="w-4 h-4 text-blue-500" /> Select Patient
            </h2>
            {loading.patients ? (
              <div className="flex justify-center py-4"><Loader2 className="w-5 h-5 text-slate-400 animate-spin" /></div>
            ) : (
              <select
                className="w-full text-sm px-3 py-2 border border-slate-200 rounded-lg bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-400"
                value={selectedPatient?.id || ''}
                onChange={e => {
                  const p = patients.find(p => p.id === parseInt(e.target.value));
                  setSelectedPatient(p || null);
                  setPrediction(null);
                }}
              >
                <option value="">— Choose patient —</option>
                {patients.map(p => (
                  <option key={p.id} value={p.id}>
                    {p.first_name} {p.last_name}
                  </option>
                ))}
              </select>
            )}
            {selectedPatient && (
              <div className="mt-3 p-3 bg-blue-50 rounded-xl">
                <div className="text-sm font-semibold text-blue-800">{selectedPatient.first_name} {selectedPatient.last_name}</div>
                <div className="text-xs text-blue-600">{selectedPatient.gender} · DOB: {selectedPatient.date_of_birth || '—'}</div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-center">
                  <div className="bg-white rounded-lg p-2">
                    <div className="text-lg font-black text-slate-800">{vitals.length}</div>
                    <div className="text-xs text-slate-500">Readings</div>
                  </div>
                  <div className="bg-white rounded-lg p-2">
                    <div className="text-lg font-black text-slate-800">{history.length}</div>
                    <div className="text-xs text-slate-500">Assessments</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Latest NEWS2 */}
          {vitals.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4">
              <h2 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-orange-500" /> Latest NEWS2
              </h2>
              <NEWS2Badge score={vitals[vitals.length - 1]?.news2_score} level={vitals[vitals.length - 1]?.news2_risk_level} />
              {vitals[vitals.length - 1]?.news2_recommendations && (
                <p className="mt-3 text-xs text-slate-600 leading-relaxed">
                  {vitals[vitals.length - 1].news2_recommendations}
                </p>
              )}
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
                {[
                  ['HR', vitals[vitals.length-1]?.heart_rate, 'bpm'],
                  ['RR', vitals[vitals.length-1]?.respiratory_rate, '/min'],
                  ['SpO₂', vitals[vitals.length-1]?.spo2, '%'],
                  ['BP', vitals[vitals.length-1]?.systolic_bp ? `${vitals[vitals.length-1].systolic_bp}/${vitals[vitals.length-1].diastolic_bp}` : null, 'mmHg'],
                  ['Temp', vitals[vitals.length-1]?.temperature_c, '°C'],
                  ['AVPU', vitals[vitals.length-1]?.consciousness_level, ''],
                ].map(([label, val, unit]) => (
                  <div key={label} className="bg-slate-50 rounded-lg px-2 py-1.5">
                    <div className="text-slate-400 text-[10px]">{label}</div>
                    <div className="font-bold text-slate-700">{val ?? '—'} <span className="font-normal text-slate-400">{unit}</span></div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Run Prediction button */}
          {selectedPatient && (
            <button
              onClick={handlePredict}
              disabled={loading.predict || vitals.length === 0}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-orange-500 to-red-600 
                         text-white font-bold rounded-xl shadow-lg shadow-red-200 hover:shadow-red-300 
                         hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading.predict ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {loading.predict ? 'Analysing…' : 'Run EWS Prediction'}
            </button>
          )}
        </div>

        {/* RIGHT — Main content tabs */}
        <div className="lg:col-span-3 space-y-4">
          {/* Tabs */}
          <div className="flex gap-1 bg-white border border-slate-200 rounded-xl p-1 shadow-sm">
            {[
              { key: 'record', label: 'Record Vitals', icon: Plus },
              { key: 'trends', label: 'Trend Charts', icon: TrendingUp },
              { key: 'predict', label: 'Risk Assessment', icon: Zap },
              { key: 'alerts', label: `Alerts ${unackCount > 0 ? `(${unackCount})` : ''}`, icon: Bell },
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-sm font-medium transition-all
                  ${tab === key
                    ? 'bg-gradient-to-r from-orange-500 to-red-600 text-white shadow-md'
                    : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
                  }`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </div>

          {/* ── TAB: Record Vitals ── */}
          {tab === 'record' && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-lg font-bold text-slate-800 mb-1 flex items-center gap-2">
                <Heart className="w-5 h-5 text-red-500" /> Record Vital Signs
              </h2>
              <p className="text-sm text-slate-500 mb-5">Enter the patient's current physiological measurements. NEWS2 score is computed automatically.</p>

              {!selectedPatient ? (
                <div className="flex flex-col items-center py-12 gap-3 text-slate-400">
                  <User className="w-12 h-12" />
                  <p className="text-sm">Select a patient from the left panel</p>
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
                    <VitalInput label="Heart Rate" id="hr" unit="bpm" icon={Heart}
                      value={form.heart_rate} onChange={v => handleFormChange('heart_rate', v)}
                      min={20} max={300} />
                    <VitalInput label="Respiratory Rate" id="rr" unit="/min" icon={Wind}
                      value={form.respiratory_rate} onChange={v => handleFormChange('respiratory_rate', v)}
                      min={1} max={80} />
                    <VitalInput label="SpO₂" id="spo2" unit="%" icon={Droplets}
                      value={form.spo2} onChange={v => handleFormChange('spo2', v)}
                      min={50} max={100} step="0.1" />
                    <VitalInput label="Systolic BP" id="sbp" unit="mmHg" icon={Activity}
                      value={form.systolic_bp} onChange={v => handleFormChange('systolic_bp', v)}
                      min={50} max={300} />
                    <VitalInput label="Diastolic BP" id="dbp" unit="mmHg" icon={Activity}
                      value={form.diastolic_bp} onChange={v => handleFormChange('diastolic_bp', v)}
                      min={20} max={200} />
                    <VitalInput label="Temperature" id="temp" unit="°C" icon={Thermometer}
                      value={form.temperature_c} onChange={v => handleFormChange('temperature_c', v)}
                      min={30} max={44} step="0.1" />
                    <VitalInput label="Urine Output" id="uo" unit="mL/hr" icon={Droplets}
                      value={form.urine_output_ml_hr} onChange={v => handleFormChange('urine_output_ml_hr', v)}
                      min={0} max={1000} />
                  </div>

                  {/* Consciousness + O2 */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <label className="text-xs font-semibold text-slate-500 flex items-center gap-1 mb-1">
                        <Eye className="w-3.5 h-3.5" /> Consciousness (AVPU)
                      </label>
                      <select
                        value={form.consciousness_level}
                        onChange={e => handleFormChange('consciousness_level', e.target.value)}
                        className="w-full text-sm px-3 py-2 border border-slate-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                      >
                        <option value="A">A — Alert</option>
                        <option value="V">V — Voice (responds to)</option>
                        <option value="P">P — Pain (responds to)</option>
                        <option value="U">U — Unresponsive</option>
                      </select>
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-semibold text-slate-500">Supplemental O₂</label>
                      <button
                        onClick={() => handleFormChange('supplemental_o2', !form.supplemental_o2)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 text-sm font-medium transition-all
                          ${form.supplemental_o2
                            ? 'border-blue-500 bg-blue-50 text-blue-700'
                            : 'border-slate-200 bg-white text-slate-500 hover:border-slate-300'
                          }`}
                      >
                        <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${form.supplemental_o2 ? 'border-blue-500 bg-blue-500' : 'border-slate-300'}`}>
                          {form.supplemental_o2 && <CheckCircle2 className="w-3 h-3 text-white" />}
                        </div>
                        On supplemental O₂
                      </button>
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="text-xs font-semibold text-slate-500 mb-1 block">Clinical Notes</label>
                    <textarea
                      rows={2}
                      value={form.notes}
                      onChange={e => handleFormChange('notes', e.target.value)}
                      placeholder="Optional notes for this observation…"
                      className="w-full text-sm px-3 py-2 border border-slate-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
                    />
                  </div>

                  <button
                    onClick={handleRecordVitals}
                    disabled={loading.record}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600
                               text-white font-bold rounded-xl shadow-lg shadow-blue-200 hover:shadow-blue-300
                               hover:scale-[1.01] active:scale-[0.99] transition-all disabled:opacity-60"
                  >
                    {loading.record ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                    {loading.record ? 'Recording…' : 'Record Vitals & Compute NEWS2'}
                  </button>

                  <div className="mt-4 p-4 rounded-xl border border-dashed border-slate-300 bg-slate-50">
                    <label className="text-xs font-semibold text-slate-600 flex items-center gap-1 mb-2">
                      <Upload className="w-3.5 h-3.5" /> Optional CSV import
                    </label>
                    <p className="text-xs text-slate-500 mb-2">
                      Headers such as recorded_at, heart_rate, respiratory_rate, spo2, temperature_c, systolic_bp.
                      Missing values are reported, not filled in.
                    </p>
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={e => handleCsvImport(e.target.files?.[0])}
                      className="text-xs"
                    />
                    {csvReport && (
                      <div className="mt-3 space-y-1 text-xs">
                        <p className="font-medium text-slate-700">Imported {csvReport.imported_count} · skipped {csvReport.skipped_count}</p>
                        {csvReport.errors?.slice(0, 6).map((msg, i) => <p key={`e${i}`} className="text-red-600">{msg}</p>)}
                        {csvReport.warnings?.slice(0, 4).map((msg, i) => <p key={`w${i}`} className="text-amber-700">{msg}</p>)}
                        {csvReport.clinical_flags?.slice(0, 4).map((msg, i) => <p key={`c${i}`} className="text-orange-700">{msg}</p>)}
                      </div>
                    )}
                  </div>

                  {/* Recent readings mini-table */}
                  {vitals.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-sm font-bold text-slate-700 mb-2">Recent Observations</h3>
                      <div className="overflow-x-auto rounded-xl border border-slate-200">
                        <table className="w-full text-xs text-slate-600">
                          <thead className="bg-slate-50 border-b border-slate-200">
                            <tr>
                              {['Time', 'HR', 'RR', 'SpO₂', 'BP', 'Temp', 'NEWS2', 'Risk'].map(h => (
                                <th key={h} className="px-3 py-2 text-left font-semibold text-slate-500">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {[...vitals].reverse().slice(0, 8).map(v => {
                              const c = getRisk(v.news2_risk_level);
                              return (
                                <tr key={v.id} className="border-b border-slate-100 hover:bg-slate-50">
                                  <td className="px-3 py-2 whitespace-nowrap">{new Date(v.recorded_at).toLocaleString('en-GB', {dateStyle:'short',timeStyle:'short'})}</td>
                                  <td className="px-3 py-2">{v.heart_rate ?? '—'}</td>
                                  <td className="px-3 py-2">{v.respiratory_rate ?? '—'}</td>
                                  <td className="px-3 py-2">{v.spo2 ?? '—'}</td>
                                  <td className="px-3 py-2 whitespace-nowrap">{v.systolic_bp && v.diastolic_bp ? `${v.systolic_bp}/${v.diastolic_bp}` : '—'}</td>
                                  <td className="px-3 py-2">{v.temperature_c ?? '—'}</td>
                                  <td className="px-3 py-2 font-bold">{v.news2_score ?? '—'}</td>
                                  <td className="px-3 py-2">
                                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${c.badge}`}>{v.news2_risk_level || '—'}</span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ── TAB: Trends ── */}
          {tab === 'trends' && (
            <div className="space-y-5">
              {!selectedPatient ? (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center py-20 gap-3 text-slate-400">
                  <TrendingUp className="w-12 h-12" />
                  <p>Select a patient to view vital-sign trends</p>
                </div>
              ) : vitals.length < 2 ? (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center py-20 gap-3 text-slate-400">
                  <Activity className="w-12 h-12" />
                  <p>Record at least 2 vital-sign readings to view trends</p>
                </div>
              ) : (
                <>
                  {/* NEWS2 over time */}
                  <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                    <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-orange-500" /> NEWS2 Score Over Time
                    </h3>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 20]} tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <ReferenceLine y={5} stroke="#f97316" strokeDasharray="4 4" label={{ value: 'Medium', position: 'right', fontSize: 10, fill: '#f97316' }} />
                        <ReferenceLine y={7} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'High', position: 'right', fontSize: 10, fill: '#ef4444' }} />
                        <Line type="monotone" dataKey="NEWS2" stroke="#f97316" strokeWidth={2.5} dot={{ r: 4, fill: '#f97316' }} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Heart Rate + SpO2 */}
                  <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                    <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                      <Heart className="w-4 h-4 text-red-500" /> Heart Rate & SpO₂
                    </h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                        <YAxis yAxisId="left" tick={{ fontSize: 11 }} domain={[40, 200]} />
                        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[85, 100]} />
                        <Tooltip />
                        <Legend />
                        <ReferenceLine yAxisId="left" y={50} stroke="#94a3b8" strokeDasharray="3 3" />
                        <ReferenceLine yAxisId="left" y={130} stroke="#f97316" strokeDasharray="3 3" />
                        <Line yAxisId="left" type="monotone" dataKey="Heart Rate" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                        <Line yAxisId="right" type="monotone" dataKey="SpO₂" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* BP + Resp Rate + Temp */}
                  <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                    <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-indigo-500" /> Blood Pressure, Resp. Rate & Temperature
                    </h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                        <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[34, 42]} />
                        <Tooltip />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="Sys. BP" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                        <Line yAxisId="left" type="monotone" dataKey="Resp. Rate" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} />
                        <Line yAxisId="right" type="monotone" dataKey="Temp (°C)" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── TAB: Risk Assessment ── */}
          {tab === 'predict' && (
            <div className="space-y-5">
              {!prediction ? (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center py-20 gap-4 text-slate-400">
                  <Zap className="w-12 h-12" />
                  <p className="text-center text-sm">
                    {!selectedPatient ? 'Select a patient first' : vitals.length === 0 ? 'Record vitals first, then run prediction' : 'Click "Run EWS Prediction" to analyse risk'}
                  </p>
                  {selectedPatient && vitals.length > 0 && (
                    <button onClick={handlePredict} disabled={loading.predict}
                      className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-orange-500 to-red-600 text-white font-bold rounded-xl shadow-md text-sm">
                      {loading.predict ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                      Run Prediction
                    </button>
                  )}
                </div>
              ) : (
                <>
                  {/* Risk overview */}
                  <div className={`bg-white rounded-2xl border-2 shadow-sm p-6 ${getRisk(prediction.ml_risk_label).border}`}>
                    <div className="flex flex-col sm:flex-row items-center gap-6">
                      <RiskGauge score={prediction.ml_risk_score} />
                      <div className="flex-1 space-y-3">
                        <div>
                          <div className="text-xs text-slate-500 mb-1">ML Deterioration Risk</div>
                          <RiskBadge label={prediction.ml_risk_label} score={prediction.ml_risk_score} />
                        </div>
                        <div className="flex gap-3 flex-wrap">
                          <NEWS2Badge score={prediction.news2_score} level={prediction.news2_risk_level} />
                        </div>
                        <div className="text-xs text-slate-500">
                          Target: {prediction.prediction_target} · Horizon: {prediction.prediction_horizon} · Model: {prediction.model_used}
                        </div>
                        {prediction.news2_recommendations && (
                          <p className="text-xs text-slate-600">{prediction.news2_recommendations}</p>
                        )}
                        <div className="text-xs text-slate-500 flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5" />
                          Assessed {new Date(prediction.assessment_dt).toLocaleString('en-GB')} · Based on {prediction.vitals_count} readings
                        </div>
                        <button
                          onClick={() => exportAssessmentPdf(selectedPatient, prediction, vitals)}
                          className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 font-medium"
                        >
                          <FileDown className="w-3.5 h-3.5" /> Export PDF report
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Key triggers */}
                  {prediction.trigger_features?.length > 0 && (
                    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                      <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-orange-500" /> Key Risk Triggers
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {prediction.trigger_features.map((f, i) => (
                          <span key={i} className="px-3 py-1 bg-orange-50 border border-orange-200 text-orange-700 rounded-full text-xs font-medium">
                            {f}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* SHAP waterfall */}
                  {prediction.shap_features?.length > 0 && (
                    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                      <h3 className="text-sm font-bold text-slate-700 mb-1 flex items-center gap-2">
                        <Info className="w-4 h-4 text-blue-500" /> Explainability — Feature Contributions
                      </h3>
                      <p className="text-xs text-slate-500 mb-4">Longer bars indicate greater contribution to risk. Red = increases risk, Green = decreases risk.</p>
                      <ShapWaterfall features={prediction.shap_features} />
                      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-400 rounded" /> Increases risk</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-emerald-400 rounded" /> Decreases risk</span>
                      </div>
                    </div>
                  )}

                  {/* Assessment history */}
                  {history.length > 0 && (
                    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                      <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-slate-500" /> Assessment History
                      </h3>
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {history.map(h => {
                          const c = getRisk(h.ml_risk_label);
                          return (
                            <div key={h.id || h.assessment_id} className={`flex items-center gap-3 p-3 rounded-xl border ${c.bg} ${c.border}`}>
                              <span className={`w-2.5 h-2.5 rounded-full ${c.dot}`} />
                              <div className="flex-1 text-xs">
                                <span className="font-semibold">{h.ml_risk_label}</span>
                                <span className="text-slate-500 ml-2">score {h.ml_risk_score ? Math.round(h.ml_risk_score*100) + '%' : '—'}</span>
                                {h.news2_score != null && <span className="text-slate-500 ml-2">NEWS2 {h.news2_score}</span>}
                              </div>
                              <div className="text-xs text-slate-400 whitespace-nowrap">
                                {new Date(h.assessment_dt).toLocaleString('en-GB', {dateStyle:'short',timeStyle:'short'})}
                              </div>
                              {h.doctor_acknowledged && <CheckCircle2 className="w-4 h-4 text-emerald-500" title="Acknowledged" />}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Disclaimer */}
                  <div className="text-xs text-slate-400 text-center flex items-center justify-center gap-1">
                    <Shield className="w-3.5 h-3.5" /> {prediction.disclaimer}
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── TAB: Alerts ── */}
          {tab === 'alerts' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <Bell className="w-5 h-5 text-red-500" /> Alert Queue
                </h2>
                <div className="flex gap-2 flex-wrap justify-end">
                  <select
                    value={alertSort}
                    onChange={e => setAlertSort(e.target.value)}
                    className="text-xs px-2 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-600"
                  >
                    <option value="risk">Sort by risk</option>
                    <option value="time">Sort by last seen</option>
                  </select>
                  <select
                    value={alertRiskFilter}
                    onChange={e => setAlertRiskFilter(e.target.value)}
                    className="text-xs px-2 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-600"
                  >
                    <option value="">All active levels</option>
                    <option value="Critical">Critical</option>
                    <option value="High">High</option>
                    <option value="Medium">Medium</option>
                  </select>
                  <button
                    onClick={() => setShowAcknowledgedAlerts(!showAcknowledgedAlerts)}
                    className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-all
                      ${showAcknowledgedAlerts ? 'border-slate-400 bg-slate-100 text-slate-700' : 'border-slate-200 text-slate-500 hover:border-slate-300'}`}
                  >
                    {showAcknowledgedAlerts ? <BellOff className="w-3.5 h-3.5 inline mr-1" /> : <Bell className="w-3.5 h-3.5 inline mr-1" />}
                    {showAcknowledgedAlerts ? 'Hide acknowledged' : 'Show all'}
                  </button>
                  <button onClick={loadAlerts} disabled={loading.alerts}
                    className="p-1.5 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50">
                    <RefreshCw className={`w-4 h-4 ${loading.alerts ? 'animate-spin' : ''}`} />
                  </button>
                </div>
              </div>

              {loading.alerts ? (
                <div className="flex justify-center py-12"><Loader2 className="w-6 h-6 animate-spin text-slate-400" /></div>
              ) : alerts.length === 0 ? (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center py-20 gap-3 text-slate-400">
                  <CheckCircle2 className="w-12 h-12 text-emerald-400" />
                  <p className="text-sm font-medium text-emerald-600">No active alerts — all patients monitored</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {alerts.map(a => {
                    const c = getRisk(a.ml_risk_label);
                    return (
                      <div key={a.assessment_id} className={`bg-white rounded-2xl border-2 shadow-sm p-4 ${c.border}`}>
                        <div className="flex items-start gap-4">
                          <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${c.bg}`}>
                            <AlertTriangle className={`w-5 h-5 ${c.text}`} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-bold text-slate-800 text-sm">{a.patient_name}</span>
                              <RiskBadge label={a.ml_risk_label} score={a.ml_risk_score} />
                              {a.doctor_acknowledged && (
                                <span className="text-xs text-emerald-600 flex items-center gap-0.5"><CheckCircle2 className="w-3.5 h-3.5" /> Ack.</span>
                              )}
                            </div>
                            <div className="flex gap-4 text-xs text-slate-500">
                              <span className="flex items-center gap-1"><Clock className="w-3.5 h-3.5" />{new Date(a.assessment_dt).toLocaleString('en-GB', {dateStyle:'short',timeStyle:'short'})}</span>
                              {a.news2_score != null && <span>NEWS2: <strong className={c.text}>{a.news2_score}</strong></span>}
                              <span>Risk: <strong className={c.text}>{Math.round(a.ml_risk_score * 100)}%</strong></span>
                            </div>
                          </div>
                          <div className="flex gap-2 shrink-0">
                            <button
                              onClick={() => { setSelectedPatient(patients.find(p => p.id === a.patient_id)); setTab('predict'); }}
                              className="text-xs px-3 py-1.5 border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 font-medium"
                            >
                              View
                            </button>
                            {!a.doctor_acknowledged && (
                              <button
                                onClick={() => setAcknowledgeModal({ id: a.assessment_id, notes: '' })}
                                className="text-xs px-3 py-1.5 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 flex items-center gap-1"
                              >
                                <CheckCircle2 className="w-3.5 h-3.5" /> Acknowledge
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Acknowledge Modal ── */}
      {acknowledgeModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md">
            <h3 className="text-lg font-bold text-slate-800 mb-1 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Acknowledge Alert
            </h3>
            <p className="text-sm text-slate-500 mb-4">Confirm you have reviewed this patient's condition and are taking appropriate action.</p>
            <textarea
              rows={3}
              value={acknowledgeModal.notes}
              onChange={e => setAcknowledgeModal(prev => ({ ...prev, notes: e.target.value }))}
              placeholder="Optional clinical notes (e.g., 'Reviewed patient, stable, continuing current management')"
              className="w-full text-sm px-3 py-2 border border-slate-200 rounded-xl bg-slate-50 focus:outline-none focus:ring-2 focus:ring-emerald-400 resize-none mb-4"
            />
            <div className="flex gap-3">
              <button onClick={() => setAcknowledgeModal(null)}
                className="flex-1 px-4 py-2 border border-slate-200 rounded-xl text-sm font-medium text-slate-600 hover:bg-slate-50">
                Cancel
              </button>
              <button onClick={handleAcknowledge}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-xl text-sm font-bold hover:bg-emerald-700">
                Confirm Acknowledge
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
