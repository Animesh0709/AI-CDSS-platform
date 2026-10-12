import React, { useState, useRef, useCallback } from 'react';
import {
  Microscope, Upload, X, AlertTriangle, CheckCircle, Loader2,
  Info, ChevronDown, ChevronUp, BarChart2, ShieldAlert, FileImage,
  Brain, Cpu, Sparkles, ClipboardList
} from 'lucide-react';
import api from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';

// ── Constants ─────────────────────────────────────────────────────────────────
const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
const MAX_MB = 10;
const CLASS_COLORS = {
  glioma:      { bg: 'bg-rose-50',    border: 'border-rose-200',    text: 'text-rose-700',    bar: 'bg-rose-400' },
  meningioma:  { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   bar: 'bg-amber-400' },
  notumor:     { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', bar: 'bg-emerald-500' },
  pituitary:   { bg: 'bg-violet-50',  border: 'border-violet-200',  text: 'text-violet-700',  bar: 'bg-violet-400' },
};
const CLASS_LABELS = {
  glioma:     'Glioma',
  meningioma: 'Meningioma',
  notumor:    'No Tumor',
  pituitary:  'Pituitary Tumor',
};
const MODEL_ICONS = {
  'XGBoost':        <Cpu className="w-4 h-4" />,
  'SVM':            <BarChart2 className="w-4 h-4" />,
  'ViT (fine-tuned)': <Brain className="w-4 h-4" />,
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function classLabel(cls) {
  return cls ? (CLASS_LABELS[cls] || cls) : '—';
}
function confPct(v) {
  return v != null ? `${(v * 100).toFixed(1)}%` : '—';
}
function StatusBadge({ status }) {
  if (status === 'ok')          return <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-700 border border-emerald-200">Ready</span>;
  if (status === 'unavailable') return <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-100 text-slate-500 border border-slate-200">Not trained</span>;
  return                               <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-600 border border-rose-200">Error</span>;
}

// ── Probability bar chart ─────────────────────────────────────────────────────
function ProbabilityBars({ probabilities, predicted }) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="mt-3 space-y-1.5">
      {entries.map(([cls, prob]) => {
        const colors = CLASS_COLORS[cls] || { bar: 'bg-slate-300', text: 'text-slate-600' };
        const isTop = cls === predicted;
        return (
          <div key={cls} className="flex items-center gap-2">
            <span className={`text-[10px] font-semibold w-24 truncate ${isTop ? 'text-slate-800' : 'text-slate-500'}`}>
              {classLabel(cls)}
            </span>
            <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${colors.bar}`}
                style={{ width: `${(prob * 100).toFixed(1)}%` }}
              />
            </div>
            <span className={`text-[10px] font-mono font-bold w-10 text-right ${isTop ? 'text-slate-800' : 'text-slate-400'}`}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Single model prediction card ──────────────────────────────────────────────
function ModelCard({ prediction }) {
  const [expanded, setExpanded] = useState(false);
  const { model_name, status, predicted_class, confidence, probabilities, message } = prediction;
  const colors = predicted_class ? CLASS_COLORS[predicted_class] || {} : {};

  return (
    <div className={`rounded-2xl border-2 shadow-sm overflow-hidden transition-all ${
      status === 'ok'
        ? `${colors.bg || 'bg-white'} ${colors.border || 'border-slate-200'}`
        : 'bg-slate-50 border-slate-200'
    }`}>
      {/* Header */}
      <div className="px-4 py-3 flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`p-1.5 rounded-lg ${status === 'ok' ? 'bg-white/70' : 'bg-slate-100'} ${colors.text || 'text-slate-500'}`}>
            {MODEL_ICONS[model_name] || <Sparkles className="w-4 h-4" />}
          </span>
          <div>
            <p className="text-xs font-bold text-slate-800">{model_name}</p>
            <StatusBadge status={status} />
          </div>
        </div>
        {status === 'ok' && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="p-1 rounded-lg hover:bg-white/50 transition-colors text-slate-400"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        )}
      </div>

      {/* Result body */}
      <div className="px-4 pb-4">
        {status === 'ok' ? (
          <>
            <div className="flex items-end justify-between">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wide font-semibold">Predicted Class</p>
                <p className={`text-lg font-extrabold mt-0.5 ${colors.text || 'text-slate-800'}`}>
                  {classLabel(predicted_class)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-slate-500 uppercase tracking-wide font-semibold">Confidence</p>
                <p className="text-2xl font-extrabold text-slate-800 mt-0.5">{confPct(confidence)}</p>
              </div>
            </div>

            {/* AI safety label */}
            <div className="mt-2 px-2 py-1.5 rounded-lg bg-white/60 border border-white text-[10px] text-slate-600 font-medium flex items-center gap-1.5">
              <ShieldAlert className="w-3 h-3 text-amber-500 shrink-0" />
              AI-assisted classification — requires clinician review
            </div>

            {/* Expandable prob bars */}
            {expanded && <ProbabilityBars probabilities={probabilities} predicted={predicted_class} />}
          </>
        ) : (
          <div className="flex items-start gap-2 mt-1">
            <AlertTriangle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-500 leading-relaxed">
              {status === 'unavailable'
                ? (message || 'Model artifact not found. Run ml/scripts/train_mri.py to generate it.')
                : (message || 'An unexpected error occurred during inference.')
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Confusion matrix display ──────────────────────────────────────────────────
function ConfusionMatrix({ matrix, classNames }) {
  if (!matrix) return null;
  const maxVal = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] border-collapse w-full">
        <thead>
          <tr>
            <th className="text-slate-400 font-normal p-1 text-right text-[9px]">True\Pred</th>
            {classNames.map(c => (
              <th key={c} className="p-1 font-semibold text-slate-600 text-center">{CLASS_LABELS[c] || c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td className="p-1 font-semibold text-slate-600 text-right pr-2">{CLASS_LABELS[classNames[ri]] || classNames[ri]}</td>
              {row.map((val, ci) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isOnDiag = ri === ci;
                return (
                  <td key={ci} className="p-1 text-center font-mono" style={{
                    backgroundColor: isOnDiag
                      ? `rgba(16,124,91,${0.1 + intensity * 0.7})`
                      : intensity > 0 ? `rgba(239,68,68,${intensity * 0.3})` : 'transparent',
                    color: intensity > 0.5 && isOnDiag ? 'white' : '#334155',
                    borderRadius: '4px',
                    minWidth: '36px',
                  }}>
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Per-class metrics table ───────────────────────────────────────────────────
function PerClassTable({ perClass }) {
  if (!perClass) return null;
  return (
    <table className="w-full text-[11px] border-collapse">
      <thead>
        <tr className="border-b border-slate-100">
          {['Class', 'Precision', 'Recall', 'F1', 'Support'].map(h => (
            <th key={h} className="text-left py-1.5 px-2 text-slate-500 font-semibold text-[10px] uppercase tracking-wide">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Object.entries(perClass).map(([cls, m]) => (
          <tr key={cls} className="border-b border-slate-50 hover:bg-slate-50">
            <td className="py-1.5 px-2 font-semibold text-slate-700">{CLASS_LABELS[cls] || cls}</td>
            <td className="py-1.5 px-2 font-mono text-slate-600">{(m.precision * 100).toFixed(1)}%</td>
            <td className="py-1.5 px-2 font-mono text-slate-600">{(m.recall    * 100).toFixed(1)}%</td>
            <td className="py-1.5 px-2 font-mono text-slate-600">{(m.f1        * 100).toFixed(1)}%</td>
            <td className="py-1.5 px-2 font-mono text-slate-400">{m.support}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Eval panel for one model ──────────────────────────────────────────────────
function ModelEvalPanel({ modelKey, label, data }) {
  const [open, setOpen] = useState(false);
  if (!data || data.status === 'not_trained') {
    return (
      <div className="p-4 rounded-xl border border-slate-100 bg-slate-50 text-xs text-slate-400">
        <p className="font-semibold text-slate-600 mb-1">{label}</p>
        Not trained — run <code className="font-mono bg-slate-100 px-1 rounded">ml/scripts/train_mri.py</code> to generate metrics.
      </div>
    );
  }

  const summary = [
    { label: 'Accuracy',     val: data.accuracy  != null ? `${(data.accuracy  * 100).toFixed(2)}%` : '—' },
    { label: 'Macro Prec.',  val: data.macro_precision != null ? `${(data.macro_precision * 100).toFixed(2)}%` : '—' },
    { label: 'Macro Recall', val: data.macro_recall    != null ? `${(data.macro_recall    * 100).toFixed(2)}%` : '—' },
    { label: 'Macro F1',     val: data.macro_f1  != null ? `${(data.macro_f1  * 100).toFixed(2)}%` : '—' },
  ];

  return (
    <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors text-left"
      >
        <span className="text-sm font-bold text-slate-800">{label}</span>
        {open ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
      </button>

      {/* Summary row always visible */}
      <div className="grid grid-cols-4 gap-0 border-t border-slate-100">
        {summary.map(s => (
          <div key={s.label} className="p-3 text-center border-r last:border-r-0 border-slate-100">
            <p className="text-[9px] font-semibold uppercase tracking-wider text-slate-400">{s.label}</p>
            <p className="text-base font-extrabold text-[#0f4c3a] mt-0.5">{s.val}</p>
          </div>
        ))}
      </div>

      {open && (
        <div className="px-4 pb-4 pt-3 space-y-4 border-t border-slate-100">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">Per-class Metrics</p>
            <PerClassTable perClass={data.per_class} />
          </div>
          {data.confusion_matrix && (
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">Confusion Matrix</p>
              <ConfusionMatrix matrix={data.confusion_matrix} classNames={data.class_names || ['glioma','meningioma','notumor','pituitary']} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DiseaseDiagnosis() {
  const [dragging,    setDragging]    = useState(false);
  const [preview,     setPreview]     = useState(null);      // data URL
  const [file,        setFile]        = useState(null);      // File object
  const [fileError,   setFileError]   = useState('');
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [apiError,    setApiError]    = useState('');
  const [showEval,    setShowEval]    = useState(false);
  const [evalData,    setEvalData]    = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError,   setEvalError]   = useState('');
  const [modelInfo,   setModelInfo]   = useState(null);
  const [showInfo,    setShowInfo]    = useState(false);
  const inputRef = useRef(null);

  // ── File handling ─────────────────────────────────────────────────────────
  const handleFile = useCallback((f) => {
    setFileError('');
    setResult(null);
    setApiError('');
    if (!f) return;
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setFileError(`Unsupported file type "${f.type}". Please upload JPEG, PNG, BMP, or TIFF.`);
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFileError(`File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e) => {
    handleFile(e.target.files[0]);
    e.target.value = '';
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setApiError('');
    setFileError('');
  };

  // ── Inference ─────────────────────────────────────────────────────────────
  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setApiError('');
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await api.post('/mri/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 70000,
      });
      setResult(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setApiError(detail || 'Inference failed. Please try again or check that the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // ── Eval report ───────────────────────────────────────────────────────────
  const loadEvalReport = async () => {
    if (evalData) { setShowEval(e => !e); return; }
    setEvalLoading(true);
    setEvalError('');
    try {
      const res = await api.get('/mri/eval-report');
      setEvalData(res.data);
      setShowEval(true);
    } catch (err) {
      setEvalError('Could not load evaluation report.');
    } finally {
      setEvalLoading(false);
    }
  };

  // ── Model info ────────────────────────────────────────────────────────────
  const loadModelInfo = async () => {
    if (modelInfo) { setShowInfo(e => !e); return; }
    try {
      const res = await api.get('/mri/model-info');
      setModelInfo(res.data.metadata);
      setShowInfo(true);
    } catch (err) {
      /* fail silently */
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-6xl space-y-6">
      {/* Page header */}
      <div>
        <h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800">
          <Microscope className="w-6 h-6 text-[#0e7c5b]" />
          Disease Diagnosis — MRI Classification
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          AI-assisted brain tumour classification using XGBoost, SVM, and Vision Transformer (ViT) models.
        </p>
      </div>

      <ClinicalDisclaimer />

      {/* Safety banner */}
      <div className="flex items-start gap-3 rounded-xl bg-amber-50 border border-amber-200 px-4 py-3">
        <ShieldAlert className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed text-amber-800">
          <p className="font-bold mb-0.5">AI-assisted classification only — not a clinical diagnosis</p>
          <p>
            These model outputs describe predicted image classes and are intended for decision support only.
            They must be reviewed and confirmed by a qualified clinician before any clinical action is taken.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Upload panel ── */}
        <div className="space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !preview && inputRef.current?.click()}
            className={`relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all cursor-pointer
              ${preview
                ? 'border-[#0e7c5b] bg-[#f7faf8] cursor-default p-0'
                : dragging
                  ? 'border-[#0e7c5b] bg-emerald-50 p-12'
                  : 'border-slate-200 hover:border-[#0e7c5b] hover:bg-[#f7faf8] p-12 bg-white'
              }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES.join(',')}
              onChange={onInputChange}
              className="hidden"
              id="mri-upload-input"
            />

            {preview ? (
              <>
                <img
                  src={preview}
                  alt="MRI preview"
                  className="w-full rounded-2xl object-contain max-h-80"
                />
                <button
                  onClick={(e) => { e.stopPropagation(); clearFile(); }}
                  className="absolute top-2 right-2 p-1.5 bg-white rounded-full border border-slate-200 shadow hover:bg-rose-50 transition-colors"
                >
                  <X className="w-4 h-4 text-slate-500" />
                </button>
              </>
            ) : (
              <div className="flex flex-col items-center gap-3 text-center pointer-events-none">
                <div className={`p-4 rounded-2xl ${dragging ? 'bg-[#0e7c5b] text-white' : 'bg-emerald-50 text-[#0e7c5b]'}`}>
                  <FileImage className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-700">
                    {dragging ? 'Drop image here' : 'Drag & drop MRI image'}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">or click to browse — JPEG/PNG/BMP/TIFF · max {MAX_MB} MB</p>
                </div>
              </div>
            )}
          </div>

          {fileError && (
            <div className="flex items-start gap-2 text-xs text-rose-700 bg-rose-50 border border-rose-100 rounded-lg px-3 py-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {fileError}
            </div>
          )}

          {preview && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-slate-600 bg-white rounded-xl border border-slate-100 px-3 py-2">
                <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />
                <span className="truncate font-medium">{file?.name}</span>
                <span className="ml-auto text-slate-400 shrink-0">{(file?.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
              <button
                onClick={runInference}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-semibold text-sm text-white
                  bg-[#0e7c5b] hover:bg-[#0f4c3a] active:scale-[0.98] transition-all
                  disabled:opacity-60 disabled:cursor-not-allowed shadow-lg shadow-emerald-900/10"
                id="mri-run-inference-btn"
              >
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Classifying…</>
                  : <><Microscope className="w-4 h-4" /> Run Classification</>
                }
              </button>
            </div>
          )}

          {apiError && (
            <div className="flex items-start gap-2 text-xs text-rose-700 bg-rose-50 border border-rose-100 rounded-xl px-3 py-3">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{apiError}</span>
            </div>
          )}

          {/* Model info button */}
          <button
            onClick={loadModelInfo}
            className="w-full flex items-center gap-2 text-xs text-slate-500 hover:text-[#0e7c5b] transition-colors py-2 px-3 rounded-xl border border-slate-100 bg-white hover:bg-[#f7faf8]"
            id="mri-model-info-btn"
          >
            <Info className="w-3.5 h-3.5" />
            {showInfo ? 'Hide model information' : 'View model information & limitations'}
            {showInfo ? <ChevronUp className="w-3.5 h-3.5 ml-auto" /> : <ChevronDown className="w-3.5 h-3.5 ml-auto" />}
          </button>

          {showInfo && modelInfo && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3 text-xs">
              <p className="font-bold text-slate-800">Model Information</p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-slate-400 font-semibold">Task</p>
                  <p className="text-slate-700">{modelInfo.task || '—'}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-slate-400 font-semibold">Status</p>
                  <p className={`font-semibold ${modelInfo.status === 'trained' ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {modelInfo.status === 'trained' ? 'Trained' : 'Not trained'}
                  </p>
                </div>
              </div>
              {modelInfo.dataset_provenance && (
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-slate-400 font-semibold mb-1">Dataset Provenance</p>
                  <p className="text-slate-600 leading-relaxed">{modelInfo.dataset_provenance}</p>
                </div>
              )}
              {modelInfo.limitations && (
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-slate-400 font-semibold mb-1">Known Limitations</p>
                  <ul className="space-y-1">
                    {modelInfo.limitations.map((l, i) => (
                      <li key={i} className="flex gap-1.5 text-slate-600">
                        <span className="text-slate-300 shrink-0">•</span>{l}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="text-[10px] text-amber-700 bg-amber-50 border border-amber-100 rounded-lg px-2 py-1.5 leading-relaxed">
                {modelInfo.clinical_safety_notice}
              </p>
            </div>
          )}
        </div>

        {/* ── Results panel ── */}
        <div className="space-y-4">
          {!result && !loading && (
            <div className="rounded-2xl border-2 border-dashed border-slate-100 bg-slate-50 flex flex-col items-center justify-center text-center p-12 h-full">
              <Microscope className="w-10 h-10 text-slate-200 mb-3" />
              <p className="text-sm font-medium text-slate-400">Upload an MRI image and run classification</p>
              <p className="text-xs text-slate-300 mt-1">Three-model comparison will appear here</p>
            </div>
          )}

          {loading && (
            <div className="rounded-2xl border border-slate-100 bg-white flex flex-col items-center justify-center text-center p-12 h-full gap-4">
              <div className="relative">
                <div className="w-14 h-14 rounded-full border-4 border-emerald-100 border-t-[#0e7c5b] animate-spin" />
                <Microscope className="w-5 h-5 text-[#0e7c5b] absolute inset-0 m-auto" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">Classifying MRI image…</p>
                <p className="text-xs text-slate-400 mt-1">Running XGBoost, SVM, and ViT models</p>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="space-y-3 animate-[fadeIn_0.3s_ease]">
              <div className="flex items-center justify-between">
                <p className="text-sm font-bold text-slate-800 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-emerald-500" />
                  Classification Results
                </p>
                <span className="text-[10px] text-slate-400">{result.filename}</span>
              </div>

              {/* Three model cards */}
              {result.predictions.map((pred) => (
                <ModelCard key={pred.model_name} prediction={pred} />
              ))}

              {/* Disclaimer */}
              <div className="text-[11px] text-slate-500 bg-slate-50 rounded-xl border border-slate-100 px-3 py-2.5 leading-relaxed">
                {result.disclaimer}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Evaluation Report Section ── */}
      <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
        <button
          onClick={loadEvalReport}
          id="mri-eval-report-btn"
          className="w-full flex items-center gap-3 px-5 py-4 hover:bg-slate-50 transition-colors text-left"
        >
          <ClipboardList className="w-5 h-5 text-[#0e7c5b]" />
          <div className="flex-1">
            <p className="text-sm font-bold text-slate-800">Model Evaluation Report</p>
            <p className="text-xs text-slate-500">Accuracy, Precision, Recall, F1-score and Confusion Matrices per model</p>
          </div>
          {evalLoading
            ? <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
            : showEval
              ? <ChevronUp className="w-4 h-4 text-slate-400" />
              : <ChevronDown className="w-4 h-4 text-slate-400" />
          }
        </button>

        {evalError && (
          <p className="px-5 pb-4 text-xs text-rose-600">{evalError}</p>
        )}

        {showEval && evalData && (
          <div className="px-5 pb-5 pt-2 space-y-4 border-t border-slate-100">
            {evalData.status === 'not_trained' ? (
              <div className="text-xs text-slate-500 bg-slate-50 rounded-xl border border-slate-100 p-4">
                <p className="font-semibold text-slate-700 mb-1">Models not yet trained</p>
                <p>{evalData.message}</p>
                <code className="mt-2 block bg-slate-100 rounded-lg px-3 py-2 font-mono text-slate-600">
                  python ml/scripts/train_mri.py
                </code>
              </div>
            ) : (
              <div className="space-y-4">
                <p className="text-xs text-slate-400">
                  Evaluated on held-out test set · generated {evalData.report?.generated_at
                    ? new Date(evalData.report.generated_at).toLocaleString() : 'unknown'}
                </p>
                <ModelEvalPanel modelKey="xgboost" label="XGBoost" data={evalData.report?.xgboost} />
                <ModelEvalPanel modelKey="svm"     label="SVM"     data={evalData.report?.svm} />
                <ModelEvalPanel modelKey="vit"     label="ViT (fine-tuned)" data={evalData.report?.vit} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
