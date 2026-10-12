import React, { useState, useMemo, useEffect } from 'react';
import {
  BrainCircuit, AlertTriangle, Info,
  Search, ChevronDown, ChevronUp, Eye, EyeOff,
  BarChart2, AlignLeft, SlidersHorizontal, AlertCircle, RefreshCw
} from 'lucide-react';
import api from '../services/api';

// ─── Form inputs ──────────────────────────────────────────────────────────────
const DEFAULTS = {
  time_in_hospital: 3, num_lab_procedures: 40, num_procedures: 0,
  num_medications: 12, number_outpatient: 0, number_emergency: 0,
  number_inpatient: 0, number_diagnoses: 5,
};
const INPUT_LABELS = {
  time_in_hospital: 'Days in hospital', num_lab_procedures: 'Lab procedures',
  num_procedures: 'Procedures', num_medications: 'Medications',
  number_outpatient: 'Outpatient visits', number_emergency: 'Emergency visits',
  number_inpatient: 'Inpatient visits', number_diagnoses: 'Diagnoses',
};

// ─── Method descriptions ──────────────────────────────────────────────────────
const METHOD_INFO = {
  shap: {
    label: 'SHAP',
    description: 'SHAP (SHapley Additive exPlanations) measures each factor\'s exact mathematical contribution to this patient\'s risk score relative to an average patient. Values are in log-odds — they sum precisely to the total shift from the baseline. Positive = raises risk; negative = lowers risk.',
  },
  lime: {
    label: 'LIME',
    description: 'LIME (Local Interpretable Model-agnostic Explanations) creates thousands of "what-if" scenarios around this specific patient, then fits a simple surrogate model to explain the local decision. It highlights which clinical conditions had the most influence on this particular prediction.',
  },
  whatif: {
    label: 'What-If Analysis',
    description: 'What-if shows how the model\'s score changes when you modify key clinical features. It is not a treatment recommendation; clinician review is required.',
  },
  global: {
    label: 'Model Overview',
    description: 'Global importance shows what the model relies on overall across the entire patient population, not causation. This helps contextualize an individual patient\'s risk factors.',
  }
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function strengthLabel(abs, max) {
  const r = max > 0 ? abs / max : 0;
  if (r > 0.66) return { label: 'Strong',   bg: 'bg-[#0f4c3a]',     text: 'text-white' };
  if (r > 0.33) return { label: 'Moderate', bg: 'bg-emerald-700',   text: 'text-white' };
  return             { label: 'Minor',    bg: 'bg-slate-200',     text: 'text-slate-600' };
}

function impactPct(abs, total) {
  if (!total) return '0 %';
  return `${((abs / total) * 100).toFixed(1)} %`;
}

function humanize(raw) {
  return raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Summary strip ────────────────────────────────────────────────────────────
function SummaryStrip({ items }) {
  const pos    = items.filter(x => x.contribution > 0);
  const neg    = items.filter(x => x.contribution < 0);
  const sumPos = pos.reduce((s, x) => s + x.contribution, 0);
  const sumNeg = neg.reduce((s, x) => s + x.contribution, 0);
  const net    = sumPos + sumNeg;

  const cells = [
    {
      label: 'Raising risk',
      value: `+${sumPos.toFixed(3)}`,
      sub: `${pos.length} factor${pos.length !== 1 ? 's' : ''}`,
      valCls: 'text-rose-600',
      bg: 'bg-rose-50 border-rose-100',
    },
    {
      label: 'Lowering risk',
      value: sumNeg.toFixed(3),
      sub: `${neg.length} factor${neg.length !== 1 ? 's' : ''}`,
      valCls: 'text-emerald-700',
      bg: 'bg-emerald-50 border-emerald-100',
    },
    {
      label: 'Net impact',
      value: (net >= 0 ? '+' : '') + net.toFixed(3),
      sub: 'log-odds units',
      valCls: net >= 0 ? 'text-rose-600' : 'text-emerald-700',
      bg: 'bg-slate-50 border-slate-200',
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-2">
      {cells.map(c => (
        <div key={c.label} className={`rounded-xl border px-3 py-2.5 text-center ${c.bg}`}>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{c.label}</p>
          <p className={`text-sm font-bold font-mono mt-0.5 ${c.valCls}`}>{c.value}</p>
          <p className="text-[10px] text-slate-400 mt-0.5">{c.sub}</p>
        </div>
      ))}
    </div>
  );
}

// ─── Factor row (expandable card) ─────────────────────────────────────────────
function FactorRow({ item, sharedMax, totalAbs, showRaw, diverging }) {
  const [open, setOpen] = useState(false);
  const isPos = item.contribution > 0;
  const abs   = Math.abs(item.contribution);
  const pct   = sharedMax > 0 ? Math.min((abs / sharedMax) * 100, 100) : 0;
  const str   = strengthLabel(abs, sharedMax);

  return (
    <div
      onClick={() => setOpen(o => !o)}
      className={`rounded-xl border cursor-pointer select-none transition-all duration-150
        ${isPos
          ? 'border-rose-100 hover:border-rose-200'
          : 'border-[#e3ebe6] hover:border-emerald-200'}
        bg-white shadow-sm hover:shadow`}
    >
      {/* ── Main row ──────────────────────────────────────────────────── */}
      <div className="px-3.5 py-2.5">
        <div className="flex items-start justify-between gap-3">
          {/* Label */}
          <div className="flex-1 min-w-0">
            <p className="text-[12px] font-semibold text-slate-800 leading-snug">
              {item.human_label || item.feature}
            </p>
            {showRaw && (
              <p className="text-[10px] font-mono text-slate-400 mt-0.5 truncate">
                {item.feature}
              </p>
            )}
          </div>
          {/* Badges + chevron */}
          <div className="flex items-center gap-1.5 shrink-0">
            {item.is_outlier && (
              <span title="Unusually large value — near-constant feature. Interpret with care.">
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
              </span>
            )}
            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${str.bg} ${str.text}`}>
              {str.label}
            </span>
            <span className={`text-[12px] font-mono font-bold tabular-nums ${isPos ? 'text-rose-600' : 'text-emerald-700'}`}>
              {item.contribution > 0 ? '+' : ''}{item.contribution}
            </span>
            {open
              ? <ChevronUp  className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              : <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />}
          </div>
        </div>

        {/* ── Bar ───────────────────────────────────────────────────────── */}
        <div className="mt-2">
          {diverging ? (
            /* Diverging view: center line, green left / red right */
            <div className="relative h-3 flex items-center">
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300 z-10" />
              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden flex">
                <div className="w-1/2 flex justify-end overflow-hidden rounded-l-full">
                  {!isPos && (
                    <div className="h-full bg-emerald-500 rounded-l-full"
                         style={{ width: `${pct}%` }} />
                  )}
                </div>
                <div className="w-1/2 flex justify-start overflow-hidden rounded-r-full">
                  {isPos && (
                    <div className="h-full bg-rose-400 rounded-r-full"
                         style={{ width: `${pct}%` }} />
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Magnitude view: plain left-to-right */
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${isPos ? 'bg-rose-400' : 'bg-emerald-500'}`}
                style={{ width: `${pct}%` }}
              />
            </div>
          )}
        </div>
      </div>

      {/* ── Expanded detail ────────────────────────────────────────────── */}
      {open && (
        <div className="border-t border-slate-100 px-3.5 py-2.5 bg-[#f7faf8] rounded-b-xl">
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
            <div>
              <span className="text-slate-500">Direction: </span>
              <span className={`font-semibold ${isPos ? 'text-rose-600' : 'text-emerald-700'}`}>
                {isPos ? '↑ Raises readmission risk' : '↓ Lowers readmission risk'}
              </span>
            </div>
            <div>
              <span className="text-slate-500">Share of total impact: </span>
              <span className="font-semibold text-slate-700">{impactPct(abs, totalAbs)}</span>
            </div>
            <div className="col-span-2">
              <span className="text-slate-500">Raw feature name: </span>
              <span className="font-mono text-slate-600 break-all">{item.feature}</span>
            </div>
            {item.is_outlier && (
              <div className="col-span-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800 leading-relaxed">
                ⚠ This factor's contribution is unusually large. It likely reflects a very rare clinical feature (e.g., Chlorpropamide is used by &lt;0.1 % of patients in the training dataset), making the model highly sensitive to its presence. Treat with additional caution.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Explanation panel (controls + factor list) ───────────────────────────────
function ExplanationPanel({ allFactors, showRaw }) {
  const [filter,   setFilter]   = useState('all');
  const [search,   setSearch]   = useState('');
  const [sortBy,   setSortBy]   = useState('impact');
  const [diverging, setDiverging] = useState(true);

  const sharedMax = useMemo(
    () => Math.max(...allFactors.map(x => Math.abs(x.contribution)), 0.001),
    [allFactors]
  );
  const totalAbs = useMemo(
    () => allFactors.reduce((s, x) => s + Math.abs(x.contribution), 0),
    [allFactors]
  );

  const visible = useMemo(() => {
    let items = allFactors;
    if (filter === 'raising')  items = items.filter(x => x.contribution > 0);
    if (filter === 'lowering') items = items.filter(x => x.contribution < 0);
    if (search.trim()) {
      const q = search.toLowerCase();
      items = items.filter(x =>
        (x.human_label || '').toLowerCase().includes(q) ||
        (x.feature     || '').toLowerCase().includes(q)
      );
    }
    const sorted = [...items];
    if (sortBy === 'impact') sorted.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    if (sortBy === 'name')   sorted.sort((a, b) => (a.human_label || a.feature).localeCompare(b.human_label || b.feature));
    return sorted;
  }, [allFactors, filter, search, sortBy]);

  const chip = (id, label, activeCls) => (
    <button
      key={id}
      onClick={() => setFilter(id)}
      className={`px-2.5 py-1 rounded-full text-[11px] font-semibold transition-colors border
        ${filter === id
          ? activeCls
          : 'bg-white border-slate-200 text-slate-600 hover:bg-[#f1f7f4] hover:border-emerald-200 hover:text-[#0f4c3a]'}`}
    >
      {label}
    </button>
  );

  return (
    <div className="space-y-3">
      <SummaryStrip items={allFactors} />

      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5 items-center">
          {chip('all',      'All',        'bg-[#0f4c3a] border-[#0f4c3a] text-white')}
          {chip('raising',  '↑ Raising',  'bg-rose-500 border-rose-500 text-white')}
          {chip('lowering', '↓ Lowering', 'bg-emerald-600 border-emerald-600 text-white')}

          <div className="ml-auto flex gap-1.5">
            <button
              onClick={() => setSortBy(s => s === 'impact' ? 'name' : 'impact')}
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-slate-200 text-[11px] text-slate-600 bg-white hover:bg-[#f1f7f4] hover:border-emerald-200 hover:text-[#0f4c3a] transition-colors"
            >
              {sortBy === 'impact' ? <BarChart2 className="w-3 h-3" /> : <AlignLeft className="w-3 h-3" />}
              <span className="hidden sm:inline">{sortBy === 'impact' ? 'By impact' : 'By name'}</span>
            </button>
            <button
              onClick={() => setDiverging(d => !d)}
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg border border-slate-200 text-[11px] text-slate-600 bg-white hover:bg-[#f1f7f4] hover:border-emerald-200 hover:text-[#0f4c3a] transition-colors"
            >
              <SlidersHorizontal className="w-3 h-3" />
              <span className="hidden sm:inline">{diverging ? 'Magnitude' : 'Diverging'}</span>
            </button>
          </div>
        </div>

        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
          <input
            type="text"
            placeholder="Search factors by label or feature name…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="input-med text-xs pl-9 py-2"
          />
        </div>
      </div>

      <div className="space-y-2">
        {visible.length === 0
          ? <p className="text-center text-[12px] text-slate-400 py-8">No factors match your filter.</p>
          : visible.map((item, i) => (
            <FactorRow
              key={`${item.feature}-${i}`}
              item={item}
              sharedMax={sharedMax}
              totalAbs={totalAbs}
              showRaw={showRaw}
              diverging={diverging}
            />
          ))
        }
      </div>
    </div>
  );
}

// ─── What-if Panel ────────────────────────────────────────────────────────────
function WhatIfPanel({ patientId, originalResult, baseForm }) {
  const [overrides, setOverrides] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Find top 8 SHAP features to show inputs for
  const topFeatures = useMemo(() => {
    const all = [...(originalResult.increasing_factors || []), ...(originalResult.decreasing_factors || [])];
    all.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    // Filter out things that are unlikely to change easily (like gender/age), keep medications/labs/admissions
    return all.filter(f => !['age_num', 'gender'].includes(f.feature)).slice(0, 8);
  }, [originalResult]);

  // Debounced prediction
  useEffect(() => {
    if (Object.keys(overrides).length === 0) {
      setResult(null);
      return;
    }
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const payload = { ...baseForm, ...overrides };
        const res = await api.post(`/xai/patients/${patientId}/predict`, payload);
        setResult(res.data);
      } catch (err) {
        setError('Failed to compute what-if score.');
      } finally {
        setLoading(false);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [overrides, patientId, baseForm]);

  const handleChange = (feature, val) => {
    setOverrides(prev => ({ ...prev, [feature]: val }));
  };

  const origProb = originalResult.probability * 100;
  const newProb = result ? result.probability * 100 : origProb;
  const delta = newProb - origProb;

  return (
    <div className="space-y-4">
      <div className="flex gap-4 p-4 rounded-xl border border-slate-200 bg-white shadow-sm items-center">
        <div className="flex-1 text-center">
          <p className="text-[11px] text-slate-500 uppercase font-semibold tracking-wide">Original Risk</p>
          <p className="text-2xl font-bold text-slate-700 mt-1">{origProb.toFixed(1)}%</p>
        </div>
        <div className="w-px h-10 bg-slate-200"></div>
        <div className="flex-1 text-center">
          <p className="text-[11px] text-slate-500 uppercase font-semibold tracking-wide">What-if Risk</p>
          <p className="text-2xl font-bold text-[#0f4c3a] mt-1">
            {loading ? <RefreshCw className="w-6 h-6 animate-spin mx-auto text-emerald-600" /> : `${newProb.toFixed(1)}%`}
          </p>
        </div>
        <div className="w-px h-10 bg-slate-200"></div>
        <div className="flex-1 text-center">
          <p className="text-[11px] text-slate-500 uppercase font-semibold tracking-wide">Change</p>
          <p className={`text-xl font-bold mt-1 ${delta > 0.05 ? 'text-rose-600' : delta < -0.05 ? 'text-emerald-600' : 'text-slate-400'}`}>
            {delta > 0 ? '+' : ''}{delta.toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="flex justify-between items-end border-b border-slate-100 pb-2">
        <h3 className="text-xs font-bold text-slate-700">Adjust clinical features</h3>
        <button
          onClick={() => setOverrides({})}
          disabled={Object.keys(overrides).length === 0}
          className="text-[11px] text-[#0e7c5b] hover:underline disabled:opacity-50"
        >
          Reset to original
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {topFeatures.map(f => {
          const isBinary = f.feature.includes('_') && !f.feature.startsWith('num') && !f.feature.startsWith('time');
          const origVal = originalResult.patient_features[f.feature];
          const currVal = overrides[f.feature] !== undefined ? overrides[f.feature] : origVal;
          
          return (
            <div key={f.feature} className="p-3 rounded-lg border border-slate-100 bg-[#f7faf8] flex flex-col justify-between">
              <label className="text-[11px] font-semibold text-slate-700 mb-2 truncate" title={f.human_label}>
                {f.human_label || f.feature}
              </label>
              {isBinary ? (
                <select
                  value={currVal}
                  onChange={e => handleChange(f.feature, Number(e.target.value))}
                  className="input-med text-xs py-1.5"
                >
                  <option value={0}>0 (No / False)</option>
                  <option value={1}>1 (Yes / True)</option>
                </select>
              ) : (
                <input
                  type="number"
                  step="any"
                  value={currVal}
                  onChange={e => handleChange(f.feature, Number(e.target.value))}
                  className="input-med text-xs py-1.5"
                />
              )}
            </div>
          );
        })}
      </div>

      {error && <p className="text-xs text-rose-600">{error}</p>}

      {result && !loading && (
        <div className="mt-4 p-4 rounded-xl border border-emerald-100 bg-emerald-50/30">
          <h4 className="text-xs font-bold text-slate-700 mb-2">New top driving factors</h4>
          <div className="space-y-2">
            {[...result.increasing_factors, ...result.decreasing_factors]
              .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
              .slice(0, 3)
              .map(item => {
                const isPos = item.contribution > 0;
                return (
                  <div key={item.feature} className="flex justify-between items-center text-[11px] bg-white p-2 rounded border border-slate-100">
                    <span className="truncate flex-1 font-medium text-slate-700">{item.human_label || item.feature}</span>
                    <span className={`font-mono font-bold ${isPos ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {isPos ? '+' : ''}{item.contribution.toFixed(4)}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Global XAI Panel ──────────────────────────────────────────────────────────
function GlobalXaiPanel({ globalData, patientResult }) {
  if (!globalData) return null;

  // Identify patient's top driver
  const patientTop = [...(patientResult.increasing_factors || []), ...(patientResult.decreasing_factors || [])]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))[0];

  let patientContextNote = null;
  if (patientTop) {
    const globalRank = globalData.global_shap.findIndex(x => x.feature === patientTop.feature);
    if (globalRank !== -1) {
      patientContextNote = `This patient's top driver, ${patientTop.human_label || patientTop.feature}, is also #${globalRank + 1} globally.`;
    } else {
      patientContextNote = `This patient's top driver, ${patientTop.human_label || patientTop.feature}, is unusual for the model and not in the global top 15.`;
    }
  }

  return (
    <div className="space-y-6">
      <div className="p-4 rounded-xl border border-slate-200 bg-white">
        <h3 className="text-sm font-bold text-slate-800 mb-1">Global SHAP Importance</h3>
        <p className="text-[11px] text-slate-500 mb-4">Top features driving the model across all patients.</p>
        
        {patientContextNote && (
          <div className="mb-4 px-3 py-2 bg-emerald-50 border border-emerald-100 rounded-lg text-xs text-emerald-800 font-medium">
            <span className="font-bold">Patient context:</span> {patientContextNote}
          </div>
        )}

        <div className="space-y-3">
          {globalData.global_shap.map((x, index) => {
            const pct = Math.min((x.mean_abs_shap / globalData.global_shap[0].mean_abs_shap) * 100, 100);
            const isPatientTop = patientTop && patientTop.feature === x.feature;
            
            return (
              <div key={x.feature} className={`flex items-center gap-3 p-1 rounded ${isPatientTop ? 'bg-emerald-50/50' : ''}`}>
                <div className="w-1/3 truncate text-[11px] font-semibold text-slate-700" title={humanize(x.feature)}>
                  <span className="text-slate-400 font-normal mr-1">#{index + 1}</span> {humanize(x.feature)}
                </div>
                <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${pct}%` }} />
                </div>
                <div className="w-12 text-right font-mono text-[11px] text-slate-500 font-bold">
                  {x.mean_abs_shap.toFixed(3)}
                </div>
                <div className="hidden sm:block w-36 text-right text-[10px] text-slate-400">
                  <span title={`Min: ${x.val_min}, Mean: ${x.val_mean}, Max: ${x.val_max}`}>
                    Avg: {x.val_mean} (Max: {x.val_max})
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      <p className="text-[11px] text-slate-400">{globalData.calibration_note}</p>
    </div>
  );
}


// ─── Main component ───────────────────────────────────────────────────────────
export default function XaiAnalysis({ patientId }) {
  const [form,     setForm]     = useState(DEFAULTS);
  const [result,   setResult]   = useState(null);
  const [global,   setGlobal]   = useState(null);
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [xaiMode,  setXaiMode]  = useState('shap'); // 'shap' | 'lime' | 'whatif' | 'global'
  const [showRaw,  setShowRaw]  = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);

  // Load global data once on mount if we want, or we can just fetch it when they click the tab
  useEffect(() => {
    if (xaiMode === 'global' && !global) {
      loadGlobal();
    }
  }, [xaiMode]);

  const predict = async e => {
    if (e) e.preventDefault();
    setLoading(true); setError('');
    try { setResult((await api.post(`/xai/patients/${patientId}/predict`, form)).data); }
    catch (err) { setError(err.response?.data?.detail || 'Prediction is unavailable.'); }
    finally { setLoading(false); }
  };

  const loadGlobal = async () => {
    setLoading(true); setError('');
    try { setGlobal((await api.get('/xai/global')).data); }
    catch (err) { setError(err.response?.data?.detail || 'Global analysis is unavailable.'); }
    finally { setLoading(false); }
  };

  const activeFactors = useMemo(() => {
    if (!result) return [];
    if (xaiMode === 'lime') {
      return [...(result.lime_increasing_factors || []), ...(result.lime_decreasing_factors || [])];
    }
    return [...(result.increasing_factors || []), ...(result.decreasing_factors || [])];
  }, [result, xaiMode]);

  return (
    <section className="med-card p-5 md:col-span-2">
      <div className="flex justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex gap-2">
          <BrainCircuit className="w-5 h-5 text-[#0e7c5b]" />
          <div>
            <h2 className="text-sm font-bold text-slate-800">Readmission prediction &amp; explainability</h2>
            <p className="text-[11px] text-slate-500">XGBoost · SHAP &amp; LIME local explanations</p>
          </div>
        </div>
        <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded h-fit shrink-0">
          Doctor review
        </span>
      </div>

      <form onSubmit={predict} className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        {Object.entries(form).map(([key, value]) => (
          <label key={key} className="text-[11px] text-slate-600">
            {INPUT_LABELS[key]}
            <input type="number" min="0" value={value}
              onChange={e => setForm({ ...form, [key]: Number(e.target.value) })}
              className="input-med text-xs mt-1" />
          </label>
        ))}
        <div className="col-span-2 md:col-span-4 flex flex-wrap gap-2">
          <button disabled={loading} className="btn-med-primary text-xs">
            {loading ? 'Analysing…' : 'Run prediction'}
          </button>
        </div>
      </form>

      {error && <p className="mt-3 text-xs text-rose-700">{error}</p>}

      {result && (
        <div className="mt-5 space-y-4">
          {/* Tabs */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-100">
            <div className="flex gap-1.5 flex-wrap">
              {['shap', 'lime', 'whatif', 'global'].map(m => (
                <button key={m} type="button" onClick={() => setXaiMode(m)}
                  className={`px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors border
                    ${xaiMode === m
                      ? 'bg-[#0e7c5b] border-[#0e7c5b] text-white shadow-sm'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-[#f1f7f4] hover:border-emerald-200 hover:text-[#0f4c3a]'}`}>
                  {METHOD_INFO[m].label}
                </button>
              ))}
            </div>
            {(xaiMode === 'shap' || xaiMode === 'lime') && (
              <button type="button" onClick={() => setShowRaw(r => !r)}
                className="ml-auto flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-[#0f4c3a] transition-colors">
                {showRaw ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                {showRaw ? 'Hide' : 'Show'} model names
              </button>
            )}
          </div>

          <div className="rounded-xl border border-[#e3ebe6] overflow-hidden">
            <button type="button" onClick={() => setInfoOpen(o => !o)}
              className="w-full flex items-center gap-2 px-3.5 py-2.5 bg-[#f7faf8] hover:bg-[#f1f7f4] transition-colors text-left">
              <Info className="w-3.5 h-3.5 text-[#0e7c5b] shrink-0" />
              <span className="text-[11px] font-semibold text-slate-700 flex-1">
                {METHOD_INFO[xaiMode].label} Information
              </span>
              {infoOpen
                ? <ChevronUp className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                : <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />}
            </button>
            {infoOpen && (
              <div className="px-3.5 py-2.5 text-[11px] text-slate-600 leading-relaxed border-t border-[#e3ebe6] bg-white">
                {METHOD_INFO[xaiMode].description}
              </div>
            )}
          </div>

          {xaiMode === 'whatif' ? (
            <WhatIfPanel patientId={patientId} originalResult={result} baseForm={form} />
          ) : xaiMode === 'global' ? (
            loading && !global ? (
              <p className="text-xs text-slate-500 py-4 text-center">Loading global model insights...</p>
            ) : (
              <GlobalXaiPanel globalData={global} patientResult={result} />
            )
          ) : (
            <>
              {/* Score card shown for SHAP/LIME views */}
              <div className="rounded-xl bg-emerald-50 border border-emerald-100 p-4">
                <p className="text-xs font-semibold text-emerald-900">Estimated model score</p>
                <div className="flex items-end gap-3 mt-0.5">
                  <p className="text-3xl font-bold text-[#0f4c3a]">
                    {(result.probability * 100).toFixed(1)}%
                  </p>
                  <span className={`mb-0.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold border
                    ${result.high_risk
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : 'bg-emerald-100 text-emerald-800 border-emerald-300'}`}>
                    {result.high_risk ? '⬆ HIGH RISK' : '⬇ LOW RISK'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1">
                  {result.model_label} · decision threshold {(result.decision_threshold * 100).toFixed(0)} %
                </p>
              </div>


              {result.ood_warning && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800 flex gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 text-amber-500" />{result.ood_warning}
                </div>
              )}

              <ExplanationPanel
                key={xaiMode}
                allFactors={activeFactors}
                showRaw={showRaw}
              />
            </>
          )}

          <div className="text-[11px] text-slate-500 space-y-0.5 pt-3 border-t border-slate-100">
            {result.data_quality.map(x => <p key={x}>• {x}</p>)}
            <p className="mt-2 text-amber-700">{result.disclaimer}</p>
          </div>
        </div>
      )}
    </section>
  );
}
