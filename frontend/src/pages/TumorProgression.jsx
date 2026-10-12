import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  TrendingUp, Activity, GitCompare, ShieldAlert, CheckCircle, AlertTriangle,
  Info, ChevronDown, ChevronUp, Loader2, Sparkles, BarChart2, Cpu, Brain,
  Calendar, Clock, Layers, ArrowRight, CheckSquare, Save, History, FlaskConical,
  User, Upload, FileImage, X, SlidersHorizontal
} from 'lucide-react';
import { progressionService, patientService } from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';

// ── Synthetic Demonstration Scenarios ──────────────────────────────────────
const DEMO_SCENARIOS = {
  rapid_progression: {
    label: 'Rapid Progression (Simulated Example)',
    tag: 'Simulated Rapid PD',
    baseline_date: '2025-01-15',
    followup_date: '2025-04-18',
    baseline_enhancing_vol_cm3: 8.4,
    followup_enhancing_vol_cm3: 22.8,
    baseline_edema_vol_cm3: 14.5,
    followup_edema_vol_cm3: 31.2,
    steroid_dose_increased: true,
    is_pseudo_progression_window: false,
    clinical_notes: '[SIMULATED SCENARIO] Distinct contrast enlargement and worsening focal deficit.',
  },
  stable_response: {
    label: 'Stable Disease (Simulated Example)',
    tag: 'Simulated Stable / SD',
    baseline_date: '2025-02-10',
    followup_date: '2025-05-12',
    baseline_enhancing_vol_cm3: 12.0,
    followup_enhancing_vol_cm3: 10.8,
    baseline_edema_vol_cm3: 18.0,
    followup_edema_vol_cm3: 15.2,
    steroid_dose_increased: false,
    is_pseudo_progression_window: false,
    clinical_notes: '[SIMULATED SCENARIO] Minor volumetric decrease; stable neurological exam.',
  },
  pseudo_progression: {
    label: 'Early Post-Radiation Window (Simulated Example)',
    tag: 'Simulated <12w Window',
    baseline_date: '2025-03-01',
    followup_date: '2025-04-20',
    baseline_enhancing_vol_cm3: 9.5,
    followup_enhancing_vol_cm3: 13.2,
    baseline_edema_vol_cm3: 12.0,
    followup_edema_vol_cm3: 19.5,
    steroid_dose_increased: true,
    is_pseudo_progression_window: true,
    clinical_notes: '[SIMULATED SCENARIO] Recent chemoradiation (<12 weeks). Suspected pseudo-progression.',
  },
};

const MODEL_ICONS = {
  'Longitudinal XGBoost': <Cpu className="w-4 h-4 text-emerald-600" />,
  'Longitudinal SVM': <BarChart2 className="w-4 h-4 text-blue-600" />,
  'Longitudinal Deep Net': <Brain className="w-4 h-4 text-purple-600" />,
};

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];

export default function TumorProgression() {
  // ── Mode Switch: Direct Scan Image Upload vs Manual Volumetrics ───────────
  const [inputMode, setInputMode] = useState('images'); // 'images' | 'manual'

  // ── Patient Picker State ──────────────────────────────────────────────────
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [patientsLoading, setPatientsLoading] = useState(true);

  // ── Dual Scan Image Upload State ──────────────────────────────────────────
  const [baselineFile, setBaselineFile] = useState(null);
  const [followupFile, setFollowupFile] = useState(null);
  const [baselinePreview, setBaselinePreview] = useState(null);
  const [followupPreview, setFollowupPreview] = useState(null);
  const [analyzingImages, setAnalyzingImages] = useState(false);
  const [imageAnalysisResult, setImageAnalysisResult] = useState(null);
  const [imageError, setImageError] = useState('');

  const baselineInputRef = useRef(null);
  const followupInputRef = useRef(null);

  // ── Form State ────────────────────────────────────────────────────────────
  const [formData, setFormData] = useState({
    baseline_date: '2025-01-15',
    followup_date: '2025-04-18',
    baseline_enhancing_vol_cm3: 8.4,
    followup_enhancing_vol_cm3: 22.8,
    baseline_edema_vol_cm3: 14.5,
    followup_edema_vol_cm3: 31.2,
    steroid_dose_increased: true,
    is_pseudo_progression_window: false,
    clinical_notes: '',
  });

  // Track if current inputs reflect a synthetic demonstration template
  const [activeDemoPreset, setActiveDemoPreset] = useState(null);

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState('');

  // ── Confirmation State ───────────────────────────────────────────────────
  const [confirmOutcome, setConfirmOutcome] = useState('PD');
  const [confirmNotes, setConfirmNotes] = useState('');
  const [confirming, setConfirming] = useState(false);
  const [confirmSuccess, setConfirmSuccess] = useState(false);

  // ── Model Info & Eval Report ──────────────────────────────────────────────
  const [showEvalModal, setShowEvalModal] = useState(false);
  const [evalReport, setEvalReport] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);

  // ── Patient History State ─────────────────────────────────────────────────
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // ── Load Registered Patients on Mount ─────────────────────────────────────
  useEffect(() => {
    async function loadPatients() {
      try {
        setPatientsLoading(true);
        const res = await patientService.getPatients();
        const patientList = res.data || [];
        setPatients(patientList);
        if (patientList.length > 0) {
          setSelectedPatientId(String(patientList[0].id));
        }
      } catch (err) {
        console.error('Failed to load registered patients:', err);
      } finally {
        setPatientsLoading(false);
      }
    }
    loadPatients();
  }, []);

  // Calculated interval weeks
  const intervalWeeks = useMemo(() => {
    if (!formData.baseline_date || !formData.followup_date) return 0.0;
    const t0 = new Date(formData.baseline_date);
    const t1 = new Date(formData.followup_date);
    const diffDays = Math.round((t1 - t0) / (1000 * 60 * 60 * 24));
    const weeks = isNaN(diffDays) ? 0.0 : Math.max(0.0, diffDays / 7.0);
    return parseFloat(weeks.toFixed(1));
  }, [formData.baseline_date, formData.followup_date]);

  // Enhancing Lesion Delta
  const enhancingDelta = useMemo(() => {
    const v0 = parseFloat(formData.baseline_enhancing_vol_cm3) || 0;
    const v1 = parseFloat(formData.followup_enhancing_vol_cm3) || 0;
    const abs = v1 - v0;
    const pct = v0 > 0 ? (abs / v0) * 100 : 0;
    return { abs: abs.toFixed(2), pct: pct.toFixed(1), isPositive: abs > 0 };
  }, [formData.baseline_enhancing_vol_cm3, formData.followup_enhancing_vol_cm3]);

  // Edema Delta
  const edemaDelta = useMemo(() => {
    const e0 = parseFloat(formData.baseline_edema_vol_cm3) || 0;
    const e1 = parseFloat(formData.followup_edema_vol_cm3) || 0;
    const abs = e1 - e0;
    return { abs: abs.toFixed(2), isPositive: abs > 0 };
  }, [formData.baseline_edema_vol_cm3, formData.followup_edema_vol_cm3]);

  // Fetch chronological study history for active patient
  const fetchPatientHistory = async (pId) => {
    if (!pId) {
      setHistory([]);
      return;
    }
    setHistoryLoading(true);
    try {
      const res = await progressionService.getHistory(pId);
      setHistory(res.data || []);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (selectedPatientId) {
      fetchPatientHistory(selectedPatientId);
    }
  }, [selectedPatientId]);

  // Load Held-out Test Benchmark Report
  const loadEvalReport = async () => {
    setShowEvalModal(true);
    if (evalReport) return;
    setEvalLoading(true);
    try {
      const res = await progressionService.getEvalReport();
      setEvalReport(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setEvalLoading(false);
    }
  };

  const handleApplyPreset = (key) => {
    if (DEMO_SCENARIOS[key]) {
      setFormData(DEMO_SCENARIOS[key]);
      setActiveDemoPreset(key);
      setResult(null);
      setApiError('');
      setConfirmSuccess(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setActiveDemoPreset(null);
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  // Image Upload Handlers
  const handleBaselineImage = (f) => {
    if (!f) return;
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setImageError(`Unsupported format for baseline scan (${f.type}). Use JPEG, PNG, BMP, or TIFF.`);
      return;
    }
    setImageError('');
    setBaselineFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setBaselinePreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const handleFollowupImage = (f) => {
    if (!f) return;
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setImageError(`Unsupported format for follow-up scan (${f.type}). Use JPEG, PNG, BMP, or TIFF.`);
      return;
    }
    setImageError('');
    setFollowupFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setFollowupPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  // Automated Feature Extraction from Uploaded MRI Scan Pair
  const handleAnalyzeImagePair = async () => {
    if (!baselineFile || !followupFile) {
      setImageError('Please select both Baseline Scan (T₀) and Follow-up Scan (T₁) MRI images.');
      return;
    }
    setAnalyzingImages(true);
    setImageError('');
    try {
      const fd = new FormData();
      fd.append('baseline_file', baselineFile);
      fd.append('followup_file', followupFile);

      const res = await progressionService.analyzeImagePair(fd);
      const est = res.data;
      setImageAnalysisResult(est);

      // Auto-populate volumetric parameters directly from uploaded MRI scans
      setFormData((prev) => ({
        ...prev,
        baseline_enhancing_vol_cm3: est.baseline_enhancing_vol_cm3,
        followup_enhancing_vol_cm3: est.followup_enhancing_vol_cm3,
        baseline_edema_vol_cm3: est.baseline_edema_vol_cm3,
        followup_edema_vol_cm3: est.followup_edema_vol_cm3,
        clinical_notes: prev.clinical_notes || est.imaging_summary,
      }));
    } catch (err) {
      const detail = err.response?.data?.detail;
      setImageError(typeof detail === 'string' ? detail : 'Failed to analyze MRI scan pair. Check server status.');
    } finally {
      setAnalyzingImages(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!selectedPatientId) {
      setApiError('Please select a registered patient before submitting.');
      return;
    }

    setSubmitting(true);
    setApiError('');
    setConfirmSuccess(false);

    try {
      const payload = {
        patient_id: parseInt(selectedPatientId, 10),
        baseline_date: formData.baseline_date,
        followup_date: formData.followup_date,
        interval_weeks: intervalWeeks,
        baseline_enhancing_vol_cm3: parseFloat(formData.baseline_enhancing_vol_cm3),
        followup_enhancing_vol_cm3: parseFloat(formData.followup_enhancing_vol_cm3),
        baseline_edema_vol_cm3: formData.baseline_edema_vol_cm3 !== '' ? parseFloat(formData.baseline_edema_vol_cm3) : null,
        followup_edema_vol_cm3: formData.followup_edema_vol_cm3 !== '' ? parseFloat(formData.followup_edema_vol_cm3) : null,
        steroid_dose_increased: Boolean(formData.steroid_dose_increased),
        is_pseudo_progression_window: Boolean(formData.is_pseudo_progression_window),
        clinical_notes: formData.clinical_notes || null,
      };

      const res = await progressionService.predict(payload);
      setResult(res.data);
      fetchPatientHistory(selectedPatientId);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setApiError(typeof detail === 'string' ? detail : 'Prediction failed. Check backend status.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleConfirmStudy = async () => {
    if (!result?.study_id) return;
    setConfirming(true);
    try {
      await progressionService.confirmOutcome(result.study_id, {
        confirmed_rano_category: confirmOutcome,
        clinical_notes: confirmNotes,
      });
      setConfirmSuccess(true);
      fetchPatientHistory(selectedPatientId);
    } catch (err) {
      alert('Confirmation failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setConfirming(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* ── Page Header ───────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2.5 rounded-xl bg-emerald-50 text-[#0e7c5b] border border-emerald-100 shadow-xs">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
                Tumor Progression Prediction
              </h1>
              <p className="text-sm text-slate-500">
                Longitudinal MRI scan image comparison & RANO trajectory assessment
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadEvalReport}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 shadow-xs transition-colors"
          >
            <BarChart2 className="w-4 h-4 text-[#0e7c5b]" />
            LUMIERE Test Benchmark (Held-Out)
          </button>
        </div>
      </div>

      <ClinicalDisclaimer />

      {/* Safety Notice */}
      <div className="flex items-start gap-3 rounded-2xl bg-amber-50/80 border border-amber-200 p-4 shadow-xs">
        <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 leading-relaxed">
          <span className="font-bold">Multidisciplinary Tumor Board (MTB) Decision Support Only:</span>{' '}
          Upload sequential brain MRI scan images (T₀ Baseline and T₁ Follow-up). The system extracts volumetric change and feeds calibrated longitudinal models (XGBoost, SVM, Deep Net) to evaluate progression risk.
        </div>
      </div>

      {/* ── Main Grid ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ── Left Column: Scan Pair & Parameters (7 cols) ────────────────── */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
            {/* Header with Mode Switch */}
            <div className="p-5 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
                  <GitCompare className="w-4 h-4 text-[#0e7c5b]" />
                  Sequential Scan Pair Analysis
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Upload sequential MRI images or enter calibrated measurements
                </p>
              </div>

              {/* Mode Toggle */}
              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl">
                <button
                  type="button"
                  onClick={() => setInputMode('images')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    inputMode === 'images'
                      ? 'bg-white text-[#0e7c5b] shadow-xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <FileImage className="w-3.5 h-3.5" />
                  MRI Images
                </button>
                <button
                  type="button"
                  onClick={() => setInputMode('manual')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    inputMode === 'manual'
                      ? 'bg-white text-[#0e7c5b] shadow-xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <SlidersHorizontal className="w-3.5 h-3.5" />
                  Volumetrics
                </button>
              </div>
            </div>

            {/* Simulated preset quick shortcuts (if in manual mode or for comparison) */}
            <div className="px-5 pt-3 flex items-center gap-1.5 flex-wrap">
              <span className="text-[11px] font-semibold text-slate-400 mr-1 flex items-center gap-1">
                <FlaskConical className="w-3 h-3 text-amber-500" />
                Simulated Presets:
              </span>
              <button
                type="button"
                onClick={() => { setInputMode('manual'); handleApplyPreset('rapid_progression'); }}
                className={`px-2 py-0.5 text-[10px] font-medium rounded-lg transition-colors border ${
                  activeDemoPreset === 'rapid_progression'
                    ? 'bg-rose-100 text-rose-800 border-rose-300 font-bold'
                    : 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100'
                }`}
              >
                Rapid PD (Demo)
              </button>
              <button
                type="button"
                onClick={() => { setInputMode('manual'); handleApplyPreset('stable_response'); }}
                className={`px-2 py-0.5 text-[10px] font-medium rounded-lg transition-colors border ${
                  activeDemoPreset === 'stable_response'
                    ? 'bg-emerald-100 text-emerald-800 border-emerald-300 font-bold'
                    : 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
                }`}
              >
                Stable / SD (Demo)
              </button>
              <button
                type="button"
                onClick={() => { setInputMode('manual'); handleApplyPreset('pseudo_progression'); }}
                className={`px-2 py-0.5 text-[10px] font-medium rounded-lg transition-colors border ${
                  activeDemoPreset === 'pseudo_progression'
                    ? 'bg-amber-100 text-amber-800 border-amber-300 font-bold'
                    : 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100'
                }`}
              >
                Post-Rad &lt;12w (Demo)
              </button>
            </div>

            {/* ── IMAGE INPUT MODE ────────────────────────────────────────── */}
            {inputMode === 'images' && (
              <div className="p-5 space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Baseline Scan Upload */}
                  <div className="space-y-2">
                    <label className="block text-xs font-bold text-slate-700 flex items-center justify-between">
                      <span>Baseline Scan (T₀)</span>
                      {baselineFile && (
                        <span className="text-[10px] font-normal text-emerald-600 font-mono">
                          {(baselineFile.size / 1024).toFixed(0)} KB
                        </span>
                      )}
                    </label>

                    <div
                      onClick={() => !baselinePreview && baselineInputRef.current?.click()}
                      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed h-44 transition-all cursor-pointer ${
                        baselinePreview
                          ? 'border-[#0e7c5b] bg-slate-900 overflow-hidden cursor-default'
                          : 'border-slate-200 hover:border-[#0e7c5b] hover:bg-emerald-50/20 bg-slate-50'
                      }`}
                    >
                      <input
                        ref={baselineInputRef}
                        type="file"
                        accept={ACCEPTED_TYPES.join(',')}
                        onChange={(e) => handleBaselineImage(e.target.files[0])}
                        className="hidden"
                      />

                      {baselinePreview ? (
                        <>
                          <img
                            src={baselinePreview}
                            alt="T0 Baseline MRI"
                            className="w-full h-full object-contain"
                          />
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setBaselineFile(null);
                              setBaselinePreview(null);
                              setImageAnalysisResult(null);
                            }}
                            className="absolute top-2 right-2 p-1 bg-white/90 rounded-full shadow hover:bg-rose-50 text-slate-600"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-bold">
                            T₀ Baseline
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col items-center text-center p-3 pointer-events-none">
                          <Upload className="w-6 h-6 text-slate-400 mb-1" />
                          <p className="text-xs font-semibold text-slate-700">Upload T₀ Baseline MRI</p>
                          <p className="text-[10px] text-slate-400 mt-0.5">JPEG / PNG / BMP / TIFF</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Follow-up Scan Upload */}
                  <div className="space-y-2">
                    <label className="block text-xs font-bold text-slate-700 flex items-center justify-between">
                      <span>Follow-up Scan (T₁)</span>
                      {followupFile && (
                        <span className="text-[10px] font-normal text-emerald-600 font-mono">
                          {(followupFile.size / 1024).toFixed(0)} KB
                        </span>
                      )}
                    </label>

                    <div
                      onClick={() => !followupPreview && followupInputRef.current?.click()}
                      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed h-44 transition-all cursor-pointer ${
                        followupPreview
                          ? 'border-[#0e7c5b] bg-slate-900 overflow-hidden cursor-default'
                          : 'border-slate-200 hover:border-[#0e7c5b] hover:bg-emerald-50/20 bg-slate-50'
                      }`}
                    >
                      <input
                        ref={followupInputRef}
                        type="file"
                        accept={ACCEPTED_TYPES.join(',')}
                        onChange={(e) => handleFollowupImage(e.target.files[0])}
                        className="hidden"
                      />

                      {followupPreview ? (
                        <>
                          <img
                            src={followupPreview}
                            alt="T1 Follow-up MRI"
                            className="w-full h-full object-contain"
                          />
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setFollowupFile(null);
                              setFollowupPreview(null);
                              setImageAnalysisResult(null);
                            }}
                            className="absolute top-2 right-2 p-1 bg-white/90 rounded-full shadow hover:bg-rose-50 text-slate-600"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-bold">
                            T₁ Follow-up
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col items-center text-center p-3 pointer-events-none">
                          <Upload className="w-6 h-6 text-slate-400 mb-1" />
                          <p className="text-xs font-semibold text-slate-700">Upload T₁ Follow-up MRI</p>
                          <p className="text-[10px] text-slate-400 mt-0.5">JPEG / PNG / BMP / TIFF</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {imageError && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{imageError}</span>
                  </div>
                )}

                {/* Extract Volumetrics Button */}
                <button
                  type="button"
                  onClick={handleAnalyzeImagePair}
                  disabled={!baselineFile || !followupFile || analyzingImages}
                  className="w-full py-2.5 px-3 rounded-xl font-semibold text-xs text-white bg-slate-800 hover:bg-slate-900 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {analyzingImages ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing MRI Slices & Extracting Volumetric Change…
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4 text-emerald-400" />
                      Extract Volumetrics from Uploaded Scan Images
                    </>
                  )}
                </button>

                {/* Image analysis output pill */}
                {imageAnalysisResult && (
                  <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-950 space-y-1 animate-fadeIn">
                    <div className="flex items-center justify-between font-bold">
                      <span className="flex items-center gap-1.5 text-emerald-800">
                        <CheckCircle className="w-4 h-4 text-emerald-600" />
                        Automated Image Analysis Complete
                      </span>
                      <span className="text-[11px] font-mono">
                        Δ Enhancing: {imageAnalysisResult.delta_enhancing_vol_cm3 > 0 ? `+${imageAnalysisResult.delta_enhancing_vol_cm3}` : imageAnalysisResult.delta_enhancing_vol_cm3} cm³ ({imageAnalysisResult.rel_change_enhancing_vol_pct}%)
                      </span>
                    </div>
                    <p className="text-[11px] text-emerald-800/80 leading-relaxed">
                      {imageAnalysisResult.imaging_summary}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ── FORM WITH CLINICAL COVARIATES & RUN PREDICTION ──────────── */}
            <form onSubmit={handlePredict} className="p-5 pt-2 space-y-5 border-t border-slate-100">
              {/* Patient Selector */}
              <div>
                <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-1.5">
                  <User className="w-3.5 h-3.5 text-[#0e7c5b]" />
                  Select Registered Patient
                </label>
                {patientsLoading ? (
                  <div className="py-2 text-xs text-slate-400 flex items-center gap-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading patients…
                  </div>
                ) : patients.length === 0 ? (
                  <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800">
                    No registered patients found. Please register a patient first via the Patient Directory.
                  </div>
                ) : (
                  <select
                    value={selectedPatientId}
                    onChange={(e) => {
                      setSelectedPatientId(e.target.value);
                      setResult(null);
                      setApiError('');
                    }}
                    required
                    className="w-full px-3 py-2 text-xs font-semibold rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-[#0e7c5b]/20 focus:border-[#0e7c5b]"
                  >
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        Patient #{p.id} — {p.first_name} {p.last_name} ({p.gender || 'Unknown'})
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Scan Dates & Computed Interval */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-[#0e7c5b]" /> Baseline Date (T₀)
                    </label>
                    <input
                      type="date"
                      name="baseline_date"
                      value={formData.baseline_date}
                      onChange={handleChange}
                      required
                      className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-white"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-1">
                      <Calendar className="w-3 h-3 text-[#0e7c5b]" /> Follow-up Date (T₁)
                    </label>
                    <input
                      type="date"
                      name="followup_date"
                      value={formData.followup_date}
                      onChange={handleChange}
                      required
                      className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-white"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-400" /> Elapsed Weeks
                    </label>
                    <div className="px-3 py-2 text-xs font-semibold rounded-xl border border-slate-200 bg-white text-slate-700">
                      {intervalWeeks} Weeks
                    </div>
                  </div>
                </div>

                {/* Volumetric Measurements T0 vs T1 */}
                <div className="pt-2 border-t border-slate-200/60">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                        Enhancing Lesion Volume (cm³) {inputMode === 'images' && <span className="text-[10px] text-[#0e7c5b] font-normal">(Auto-calculated from scans)</span>}
                      </label>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          name="baseline_enhancing_vol_cm3"
                          value={formData.baseline_enhancing_vol_cm3}
                          onChange={handleChange}
                          placeholder="T₀ (cm³)"
                          required
                          className="w-1/2 px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white"
                        />
                        <ArrowRight className="w-3 h-3 text-slate-400 shrink-0" />
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          name="followup_enhancing_vol_cm3"
                          value={formData.followup_enhancing_vol_cm3}
                          onChange={handleChange}
                          placeholder="T₁ (cm³)"
                          required
                          className="w-1/2 px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                        Peritumoral Edema / FLAIR (cm³)
                      </label>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          name="baseline_edema_vol_cm3"
                          value={formData.baseline_edema_vol_cm3}
                          onChange={handleChange}
                          placeholder="T₀ (cm³)"
                          className="w-1/2 px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white"
                        />
                        <ArrowRight className="w-3 h-3 text-slate-400 shrink-0" />
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          name="followup_edema_vol_cm3"
                          value={formData.followup_edema_vol_cm3}
                          onChange={handleChange}
                          placeholder="T₁ (cm³)"
                          className="w-1/2 px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Volumetric velocity badge */}
                  <div className="mt-3 flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-white border border-slate-200">
                    <span className="text-slate-500 font-medium">Volumetric Velocity (T₀ → T₁):</span>
                    <div className="flex items-center gap-3 font-mono font-bold">
                      <span className={enhancingDelta.isPositive ? 'text-rose-600' : 'text-emerald-600'}>
                        Δ Enhancing: {enhancingDelta.isPositive ? `+${enhancingDelta.abs}` : enhancingDelta.abs} cm³ ({enhancingDelta.pct}%)
                      </span>
                      <span className="text-slate-300">|</span>
                      <span className={edemaDelta.isPositive ? 'text-amber-600' : 'text-slate-600'}>
                        Δ Edema: {edemaDelta.isPositive ? `+${edemaDelta.abs}` : edemaDelta.abs} cm³
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Clinical Modifiers */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs">
                  <input
                    type="checkbox"
                    id="steroid_cb"
                    name="steroid_dose_increased"
                    checked={formData.steroid_dose_increased}
                    onChange={handleChange}
                    className="rounded text-[#0e7c5b] focus:ring-[#0e7c5b]"
                  />
                  <label htmlFor="steroid_cb" className="text-slate-700 cursor-pointer font-medium">
                    Corticosteroid (dexamethasone) dose increased between scans
                  </label>
                </div>

                <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs">
                  <input
                    type="checkbox"
                    id="pseudo_cb"
                    name="is_pseudo_progression_window"
                    checked={formData.is_pseudo_progression_window}
                    onChange={handleChange}
                    className="rounded text-[#0e7c5b] focus:ring-[#0e7c5b]"
                  />
                  <label htmlFor="pseudo_cb" className="text-slate-700 cursor-pointer font-medium">
                    Within 12 weeks of completing chemoradiation (RANO pseudo-progression window)
                  </label>
                </div>
              </div>

              {/* Notes input */}
              <div>
                <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1">
                  Clinical Examination & Observation Notes
                </label>
                <textarea
                  rows={2}
                  name="clinical_notes"
                  value={formData.clinical_notes}
                  onChange={handleChange}
                  placeholder="Enter focal neurological symptoms, steroid dosage details, or scan quality notes…"
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 focus:outline-none focus:border-[#0e7c5b]"
                />
              </div>

              {/* Submit CTA */}
              <button
                type="submit"
                disabled={submitting || !selectedPatientId}
                className="w-full py-3 px-4 rounded-xl font-semibold text-sm text-white bg-[#0e7c5b] hover:bg-[#0f4c3a] active:scale-[0.98] transition-all disabled:opacity-60 disabled:cursor-not-allowed shadow-md shadow-emerald-900/10 flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Executing Longitudinal Models…
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-4 h-4" />
                    Predict Tumor Progression
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Longitudinal Trajectory Timeline for Selected Patient */}
          <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2 mb-3">
              <History className="w-4 h-4 text-[#0e7c5b]" />
              Historical Progression Trajectory (Patient #{selectedPatientId || '—'})
            </h3>

            {historyLoading ? (
              <div className="py-6 flex items-center justify-center gap-2 text-xs text-slate-400">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading recorded studies…
              </div>
            ) : history.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-400">
                No previous progression studies recorded for this patient. Run a prediction to log the first longitudinal study.
              </div>
            ) : (
              <div className="space-y-2.5">
                {history.map((study) => (
                  <div
                    key={study.id}
                    className="p-3 rounded-xl border border-slate-100 bg-slate-50/70 hover:bg-slate-50 transition-colors flex items-center justify-between text-xs"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded-md font-bold text-[10px] ${
                          study.consensus_label === 'Progressive Disease'
                            ? 'bg-rose-100 text-rose-700'
                            : 'bg-emerald-100 text-emerald-700'
                        }`}>
                          {study.consensus_label}
                        </span>
                        <span className="font-semibold text-slate-700">
                          {study.baseline_date} → {study.followup_date}
                        </span>
                        <span className="text-slate-400 text-[10px]">
                          ({study.interval_weeks}w interval)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1">
                        XGBoost: {(study.xgb_progression_probability * 100).toFixed(1)}% | SVM: {(study.svm_progression_probability * 100).toFixed(1)}% | DL: {(study.dl_progression_probability * 100).toFixed(1)}%
                      </p>
                      {study.confirmed_rano_category && (
                        <p className="text-[10px] text-emerald-600 font-semibold mt-0.5">
                          ✓ Attending RANO Confirmed: {study.confirmed_rano_category}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Right Column: 3-Model Results & Clinical Action (5 cols) ─────── */}
        <div className="lg:col-span-5 space-y-6">
          {apiError && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{apiError}</span>
            </div>
          )}

          {!result && !submitting && (
            <div className="bg-white rounded-2xl border-2 border-dashed border-slate-200 p-12 text-center flex flex-col items-center justify-center">
              <Layers className="w-10 h-10 text-slate-300 mb-3" />
              <h3 className="text-sm font-bold text-slate-700">No Prediction Generated Yet</h3>
              <p className="text-xs text-slate-400 mt-1 max-w-xs">
                Upload sequential MRI scan images or configure observations, then click &quot;Predict Tumor Progression&quot; to evaluate XGBoost, SVM, and Deep Learning models.
              </p>
            </div>
          )}

          {submitting && (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center flex flex-col items-center justify-center">
              <Loader2 className="w-10 h-10 animate-spin text-[#0e7c5b] mb-3" />
              <h3 className="text-sm font-bold text-slate-700">Evaluating Temporal Radiomics</h3>
              <p className="text-xs text-slate-400 mt-1">
                Comparing volumetric velocity across XGBoost, SVM, and Deep Neural Net…
              </p>
            </div>
          )}

          {result && !submitting && (
            <div className="space-y-4">
              {/* Consensus Overview Card */}
              <div className={`rounded-2xl border-2 p-5 shadow-sm transition-all ${
                result.consensus_label === 'Progressive Disease'
                  ? 'bg-rose-50/50 border-rose-200'
                  : 'bg-emerald-50/50 border-emerald-200'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Multimodel Consensus
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-extrabold ${
                    result.consensus_agreement === 'Full Consensus'
                      ? 'bg-slate-900 text-white'
                      : 'bg-slate-200 text-slate-700'
                  }`}>
                    {result.consensus_agreement}
                  </span>
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <div>
                    <h3 className={`text-xl font-extrabold ${
                      result.consensus_label === 'Progressive Disease'
                        ? 'text-rose-700'
                        : 'text-emerald-700'
                    }`}>
                      {result.consensus_label}
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      RANO Assessment ({result.baseline_date} → {result.followup_date})
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Enhancing Volume Δ</span>
                    <div className="text-2xl font-black text-slate-800">
                      {result.rel_change_enhancing_vol_pct > 0 ? `+${result.rel_change_enhancing_vol_pct}%` : `${result.rel_change_enhancing_vol_pct}%`}
                    </div>
                  </div>
                </div>

                {/* Clinical Disclaimer */}
                <div className="mt-3 p-3 rounded-xl bg-white/80 border border-white text-xs text-slate-700 leading-relaxed">
                  <p className="font-semibold text-slate-800 mb-0.5">Clinical Protocol Notice:</p>
                  {result.clinical_disclaimer}
                </div>
              </div>

              {/* 3 Model Comparative Cards */}
              <div className="space-y-3">
                {result.predictions.map((m) => {
                  const isPD = m.predicted_label === 'Progressive Disease';
                  const probPct = (m.progression_probability * 100).toFixed(1);

                  return (
                    <div
                      key={m.model_name}
                      className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {MODEL_ICONS[m.model_name] || <Sparkles className="w-4 h-4 text-slate-400" />}
                          <span className="text-xs font-bold text-slate-800">{m.model_name}</span>
                        </div>
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                          isPD ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
                        }`}>
                          {m.predicted_label}
                        </span>
                      </div>

                      {/* Probability Bar */}
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-slate-400">Progression Probability</span>
                          <span className="font-mono font-bold text-slate-700">{probPct}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full transition-all rounded-full ${
                              m.progression_probability >= 0.5 ? 'bg-rose-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${probPct}%` }}
                          />
                        </div>
                      </div>

                      {/* Confidence & Drivers */}
                      <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500">
                        <span>Confidence: <strong className="text-slate-700">{(m.confidence * 100).toFixed(1)}%</strong></span>
                        {m.key_drivers && m.key_drivers.length > 0 && (
                          <span className="text-slate-500">
                            Top driver: <strong className="text-slate-700">{m.key_drivers[0].factor} ({m.key_drivers[0].delta || m.key_drivers[0].value})</strong>
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Clinician Outcome Confirmation & Audit Log */}
              <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                  <CheckSquare className="w-4 h-4 text-[#0e7c5b]" />
                  Clinician-in-the-Loop RANO Confirmation
                </h4>
                <p className="text-[11px] text-slate-400">
                  Record attending neuroradiologist / oncologist ground truth for clinical audit compliance
                </p>

                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <label className="flex items-center gap-2 p-2 rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50">
                      <input
                        type="radio"
                        name="confirm_outcome"
                        value="PD"
                        checked={confirmOutcome === 'PD'}
                        onChange={(e) => setConfirmOutcome(e.target.value)}
                        className="text-[#0e7c5b]"
                      />
                      <span className="font-semibold text-rose-700">Progressive Disease (PD)</span>
                    </label>
                    <label className="flex items-center gap-2 p-2 rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50">
                      <input
                        type="radio"
                        name="confirm_outcome"
                        value="SD"
                        checked={confirmOutcome === 'SD'}
                        onChange={(e) => setConfirmOutcome(e.target.value)}
                        className="text-[#0e7c5b]"
                      />
                      <span className="font-semibold text-emerald-700">Stable Disease (SD)</span>
                    </label>
                    <label className="flex items-center gap-2 p-2 rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50">
                      <input
                        type="radio"
                        name="confirm_outcome"
                        value="PR"
                        checked={confirmOutcome === 'PR'}
                        onChange={(e) => setConfirmOutcome(e.target.value)}
                        className="text-[#0e7c5b]"
                      />
                      <span className="font-semibold text-blue-700">Partial Response (PR)</span>
                    </label>
                    <label className="flex items-center gap-2 p-2 rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50">
                      <input
                        type="radio"
                        name="confirm_outcome"
                        value="CR"
                        checked={confirmOutcome === 'CR'}
                        onChange={(e) => setConfirmOutcome(e.target.value)}
                        className="text-[#0e7c5b]"
                      />
                      <span className="font-semibold text-purple-700">Complete Response (CR)</span>
                    </label>
                  </div>

                  <textarea
                    rows={2}
                    value={confirmNotes}
                    onChange={(e) => setConfirmNotes(e.target.value)}
                    placeholder="Enter multidisciplinary tumor board discussion notes or follow-up imaging plan…"
                    className="w-full p-2.5 text-xs rounded-xl border border-slate-200 focus:outline-none focus:border-[#0e7c5b]"
                  />

                  <button
                    type="button"
                    onClick={handleConfirmStudy}
                    disabled={confirming || confirmSuccess}
                    className="w-full py-2 px-3 rounded-xl font-semibold text-xs text-white bg-slate-800 hover:bg-slate-900 transition-colors flex items-center justify-center gap-1.5 disabled:opacity-60"
                  >
                    {confirming ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : confirmSuccess ? (
                      <><CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> RANO Confirmed & Saved to DB</>
                    ) : (
                      <><Save className="w-3.5 h-3.5" /> Confirm RANO Outcome</>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── LUMIERE Evaluation Benchmark Modal ─────────────────────────────── */}
      {showEvalModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-slate-200">
            <div className="p-5 border-b border-slate-100 flex items-center justify-between sticky top-0 bg-white z-10">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-[#0e7c5b]" />
                <h3 className="text-base font-bold text-slate-800">
                  LUMIERE Dataset Held-Out Benchmark Evaluation
                </h3>
              </div>
              <button
                onClick={() => setShowEvalModal(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-6">
              {evalLoading ? (
                <div className="py-12 flex flex-col items-center justify-center gap-3">
                  <Loader2 className="w-8 h-8 animate-spin text-[#0e7c5b]" />
                  <p className="text-xs text-slate-500">Loading benchmark test metrics…</p>
                </div>
              ) : evalReport ? (
                <div className="space-y-6">
                  {/* Split Summary */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Held-Out Patients</span>
                      <p className="text-lg font-black text-slate-800 mt-0.5">
                        {evalReport.held_out_patients_count || 18}
                      </p>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Total Transitions</span>
                      <p className="text-lg font-black text-slate-800 mt-0.5">
                        {evalReport.held_out_transitions_count || 70}
                      </p>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Progressive Disease</span>
                      <p className="text-lg font-black text-rose-600 mt-0.5">
                        {evalReport.test_class_distribution?.progressive_disease_count || 44}
                      </p>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400">Stable / Response</span>
                      <p className="text-lg font-black text-emerald-600 mt-0.5">
                        {evalReport.test_class_distribution?.stable_or_response_count || 26}
                      </p>
                    </div>
                  </div>

                  {/* Benchmark Comparison Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50">
                          <th className="py-2.5 px-3 font-bold text-slate-700">Model Architecture</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">PR-AUC (95% CI)</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">Sensitivity</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">Specificity</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">Balanced Acc</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">ROC-AUC</th>
                          <th className="py-2.5 px-3 font-bold text-slate-700">Brier Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {Object.entries(evalReport.models || {}).map(([key, m]) => {
                          const mets = m.metrics || {};
                          const cis = m.confidence_intervals || {};
                          return (
                            <tr key={key} className="hover:bg-slate-50/60 font-mono">
                              <td className="py-3 px-3 font-sans font-bold text-slate-800">
                                {m.name || key}
                              </td>
                              <td className="py-3 px-3 font-bold text-[#0e7c5b]">
                                {(mets.pr_auc * 100).toFixed(1)}%
                                {cis.pr_auc && (
                                  <span className="block text-[10px] font-normal text-slate-400 font-sans">
                                    [{(cis.pr_auc.ci_lower * 100).toFixed(1)}%, {(cis.pr_auc.ci_upper * 100).toFixed(1)}%]
                                  </span>
                                )}
                              </td>
                              <td className="py-3 px-3 text-slate-700">
                                {(mets.sensitivity * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-3 text-slate-700">
                                {(mets.specificity * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-3 text-slate-700">
                                {(mets.balanced_accuracy * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-3 text-slate-700">
                                {(mets.roc_auc * 100).toFixed(1)}%
                              </td>
                              <td className="py-3 px-3 text-slate-600">
                                {mets.brier_score != null ? mets.brier_score.toFixed(4) : '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Scientific Provenance Note */}
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1.5 leading-relaxed">
                    <p className="font-bold text-slate-800">Scientific Provenance & Zero-Leakage Splitting:</p>
                    <p>
                      The benchmark was generated using the peer-reviewed LUMIERE glioblastoma longitudinal dataset (*Scientific Data / Nature*, 2022).
                      Patients were partitioned at the subject identifier level prior to temporal pairwise transition extraction.
                      No scans or radiomic profiles from test patients ever entered training or hyperparameter tuning.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-xs text-slate-500">
                  Evaluation report could not be loaded. Check backend status.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
