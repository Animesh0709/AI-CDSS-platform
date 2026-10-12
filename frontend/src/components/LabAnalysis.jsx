import React, { useEffect, useState } from 'react';
import { FlaskConical, ShieldCheck, Sparkles, RotateCcw } from 'lucide-react';
import { patientService } from '../services/api';

const initialForm = { test_name: 'Glucose', value: '', unit: 'mg/dL', test_date: new Date().toISOString().slice(0, 10) };
const statusStyle = {
  Normal: 'bg-emerald-50 text-emerald-800 border-emerald-200',
  'Potentially High': 'bg-amber-50 text-amber-800 border-amber-200',
  'Potentially Low': 'bg-sky-50 text-sky-800 border-sky-200',
};

const extractOcrLabs = (reports) => {
  const ocrLabs = [];
  const seen = new Set();

  for (const report of reports || []) {
    // 1. Parse from pre-extracted entities if available
    const entities = report.clinical_entities?.lab_values || [];
    for (const ent of entities) {
      const name = ent.name || ent.text;
      const key = `${name.toLowerCase()}:${ent.value}`;
      if (!seen.has(key)) {
        seen.add(key);
        ocrLabs.push({
          test_name: name,
          value: String(ent.value),
          unit: ent.unit || (name.toLowerCase() === 'glucose' ? 'mg/dL' : ''),
          source: report.filename,
          date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
        });
      }
    }

    // 2. Direct regex scan on extracted_text as client-side fallback
    const text = report.extracted_text || '';
    if (text) {
      // Glucose and clinical aliases (FBS, Blood Sugar, etc.)
      const glucoseRegex = /\b(?:fasting\s+(?:blood\s+)?glucose|fasting\s+blood\s+sugar|blood\s+glucose|blood\s+sugar|serum\s+glucose|random\s+blood\s+sugar|glucose|fbs|rbs|fpg)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg\/dL|mmol\/L)?/gi;
      let m;
      while ((m = glucoseRegex.exec(text)) !== null) {
        const val = m[1];
        const key = `glucose:${val}`;
        if (!seen.has(key)) {
          seen.add(key);
          ocrLabs.push({
            test_name: 'Glucose',
            value: val,
            unit: m[2] || 'mg/dL',
            source: report.filename,
            date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
          });
        }
      }

      // HbA1c
      const a1cRegex = /\b(?:glycated\s+hemoglobin|glycosylated\s+hemoglobin|hba1c|ha1c|a1c)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(%)?/gi;
      while ((m = a1cRegex.exec(text)) !== null) {
        const val = m[1];
        const key = `hba1c:${val}`;
        if (!seen.has(key)) {
          seen.add(key);
          ocrLabs.push({
            test_name: 'HbA1c',
            value: val,
            unit: '%',
            source: report.filename,
            date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
          });
        }
      }

      // Creatinine
      const creatRegex = /\b(?:serum\s+creatinine|creatinine|creat)\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(mg\/dL)?/gi;
      while ((m = creatRegex.exec(text)) !== null) {
        const val = m[1];
        const key = `creatinine:${val}`;
        if (!seen.has(key)) {
          seen.add(key);
          ocrLabs.push({
            test_name: 'Creatinine',
            value: val,
            unit: 'mg/dL',
            source: report.filename,
            date: report.uploaded_at?.slice(0, 10) || new Date().toISOString().slice(0, 10),
          });
        }
      }
    }
  }
  return ocrLabs;
};

export default function LabAnalysis({ patientId, refreshKey = 0 }) {
  const [results, setResults] = useState([]);
  const [form, setForm] = useState(initialForm);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [ocrLabs, setOcrLabs] = useState([]);
  const [autoFilled, setAutoFilled] = useState(null);

  const load = async () => {
    try {
      const [resLabs, resReports] = await Promise.all([
        patientService.listLabResults(patientId),
        patientService.listReports(patientId).catch(() => ({ data: [] })),
      ]);
      setResults(resLabs.data || []);

      const detected = extractOcrLabs(resReports.data || []);
      setOcrLabs(detected);

      // Auto-fill glucose value if form value is currently empty
      const glucoseItem = detected.find((item) => item.test_name.toLowerCase() === 'glucose');
      if (glucoseItem) {
        setForm((prev) => {
          if (!prev.value || prev.value === '') {
            return {
              ...prev,
              test_name: 'Glucose',
              value: glucoseItem.value,
              unit: glucoseItem.unit || 'mg/dL',
              test_date: glucoseItem.date,
            };
          }
          return prev;
        });
        setAutoFilled(glucoseItem);
      }
    } catch {
      setError('Unable to load laboratory results.');
    }
  };

  useEffect(() => {
    load();
  }, [patientId, refreshKey]);

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    setSaving(true);
    try {
      await patientService.addLabResult(patientId, { ...form, value: Number(form.value), unit: form.unit || undefined });
      setForm(initialForm);
      setAutoFilled(null);
      await load();
    } catch (err) {
      setError(err.response?.data?.detail || 'Unable to analyse and save this result.');
    } finally {
      setSaving(false);
    }
  };

  const change = (event) => {
    setForm({ ...form, [event.target.name]: event.target.value });
    if (event.target.name === 'value' && autoFilled && event.target.value !== autoFilled.value) {
      setAutoFilled(null);
    }
  };

  const fillFromOcr = (item) => {
    setForm({
      test_name: item.test_name,
      value: item.value,
      unit: item.unit || (item.test_name.toLowerCase() === 'glucose' ? 'mg/dL' : ''),
      test_date: item.date,
    });
    setAutoFilled(item);
  };

  const clearAutoFill = () => {
    setForm(initialForm);
    setAutoFilled(null);
  };

  return (
    <section className="med-card p-5">
      <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-[#0e7c5b]" />
          <h2 className="text-sm font-bold text-slate-800">Laboratory Analysis</h2>
        </div>
        <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded font-medium">
          Live Rule Engine
        </span>
      </div>
      
      <p className="mt-3 text-xs text-slate-600">
        Enter a clinician-reviewed result. The configured rule only flags a value as Normal, Potentially High, or Potentially Low; it is not a diagnosis.
      </p>

      {/* Auto-filled OCR Notification Banner */}
      {autoFilled && form.value === autoFilled.value && (
        <div className="mt-3 flex items-center justify-between gap-2 p-2.5 rounded-lg border border-emerald-200 bg-emerald-50/70 text-xs text-emerald-900 animate-in fade-in">
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>
              Auto-filled <strong>{autoFilled.test_name}</strong> from OCR: <strong className="font-mono">{autoFilled.value} {autoFilled.unit}</strong>
              {autoFilled.source && <span className="text-emerald-700 ml-1">({autoFilled.source})</span>}
            </span>
          </div>
          <button
            type="button"
            onClick={clearAutoFill}
            className="flex items-center gap-1 text-[11px] text-emerald-700 hover:text-emerald-950 font-medium underline px-1 py-0.5"
            title="Reset form"
          >
            <RotateCcw className="w-3 h-3" />
            Clear
          </button>
        </div>
      )}

      {/* OCR Quick Chips if any values were detected */}
      {ocrLabs.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-1 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-[#0e7c5b]" /> OCR Detected:
          </span>
          {ocrLabs.map((lab, idx) => (
            <button
              key={`${lab.test_name}-${lab.value}-${idx}`}
              type="button"
              onClick={() => fillFromOcr(lab)}
              className={`px-2 py-0.5 rounded-md text-[11px] font-medium border transition-colors flex items-center gap-1 ${
                form.test_name.toLowerCase() === lab.test_name.toLowerCase() && String(form.value) === String(lab.value)
                  ? 'bg-[#0e7c5b] text-white border-[#0e7c5b]'
                  : 'bg-[#eaf5f0] text-[#0e7c5b] border-[#c2e4d4] hover:bg-[#d5ece1]'
              }`}
            >
              <span>{lab.test_name}: {lab.value} {lab.unit}</span>
            </button>
          ))}
        </div>
      )}

      <form onSubmit={submit} className="mt-3 grid grid-cols-1 sm:grid-cols-4 gap-2">
        <input
          name="test_name"
          value={form.test_name}
          onChange={change}
          className="input-med text-xs"
          placeholder="Test (e.g. Glucose)"
          required
        />
        <input
          name="value"
          type="number"
          step="any"
          value={form.value}
          onChange={change}
          className="input-med text-xs font-mono font-medium"
          placeholder="Result (e.g. 140)"
          required
        />
        <input
          name="unit"
          value={form.unit}
          onChange={change}
          className="input-med text-xs"
          placeholder="Unit (mg/dL)"
        />
        <div className="flex gap-2">
          <input
            name="test_date"
            type="date"
            value={form.test_date}
            onChange={change}
            className="input-med text-xs min-w-0"
          />
          <button
            disabled={saving || !form.value}
            className="btn-med-primary text-xs px-3 whitespace-nowrap"
          >
            {saving ? 'Checking…' : 'Analyse'}
          </button>
        </div>
      </form>

      <p className="mt-2 text-[10px] text-slate-400">
        Supported: Glucose, HbA1c, Hemoglobin, WBC, Platelets, Creatinine, Urea, Sodium, Potassium, Cholesterol, LDL, HDL, Triglycerides, TSH.
      </p>

      {error && <p className="mt-2 text-xs text-rose-700">{error}</p>}

      <div className="mt-4 space-y-2">
        {!results.length ? (
          <p className="text-xs text-slate-400">No clinician-reviewed laboratory results saved.</p>
        ) : (
          results.map((result) => (
            <div key={result.id} className="rounded-lg border border-slate-100 p-3 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-slate-800">
                  {result.test_name}: {result.value} {result.unit}
                </span>
                <span className={`border rounded-full px-2 py-0.5 font-semibold ${statusStyle[result.status] || 'bg-slate-50 text-slate-700 border-slate-200'}`}>
                  {result.status}
                </span>
              </div>
              <p className="mt-1 text-slate-500">
                Reference: {result.reference_range}. {result.clinical_basis}
              </p>
              <p className="mt-1 flex items-center gap-1 text-slate-400">
                <ShieldCheck className="w-3 h-3" /> Doctor review required
                {result.test_date ? ` · Test date: ${result.test_date}` : ''}
              </p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
