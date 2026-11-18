import React from 'react';
import { 
  CheckCircle, AlertCircle, HelpCircle, BarChart3, 
  Layers, ChevronRight, ShieldCheck, Info
} from 'lucide-react';

const CLASS_DESCRIPTIONS = {
  NORM: 'Normal Electrocardiogram (Within standard physiological limits)',
  MI: 'Myocardial Infarction (Pathological Q-waves / acute injury patterns)',
  STTC: 'ST-T Wave Changes (Ischemia, repolarization abnormalities)',
  CD: 'Conduction Disturbance (Bundle branch blocks, fascicular delays)',
  HYP: 'Hypertrophy (Ventricular or atrial voltage/strain patterns)',
};

export const EcgClassificationPanel = ({ prediction, selectedClass, onSelectClass }) => {
  if (!prediction) {
    return (
      <div className="p-6 bg-slate-50 border border-slate-200 rounded-xl text-center text-slate-400 text-xs">
        No model classification results available.
      </div>
    );
  }

  const {
    model_name,
    model_version,
    target_classes = [],
    raw_model_logits = {},
    uncalibrated_model_probabilities = {},
    model_decisions = {},
    thresholds_used = {},
    queried_class,
  } = prediction;

  return (
    <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#0e7c5b]" />
            Multi-Label ResNet Predictions
          </h2>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Architecture: {model_name} • Checkpoint: {model_version}
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 self-start sm:self-auto">
          <ShieldCheck className="w-3 h-3 text-emerald-600" />
          Decision Support Only
        </div>
      </div>

      {/* Class Rows */}
      <div className="space-y-2.5">
        {target_classes.map((cls) => {
          const prob = uncalibrated_model_probabilities[cls] || 0.0;
          const logit = raw_model_logits[cls] !== undefined ? raw_model_logits[cls] : null;
          const decision = model_decisions[cls] || 'NEGATIVE';
          const threshold = thresholds_used[cls] || 0.5;
          const isPositive = decision.startsWith('POSITIVE');
          const isTarget = (selectedClass || queried_class) === cls;

          return (
            <div
              key={cls}
              onClick={() => onSelectClass && onSelectClass(cls)}
              className={`p-3 rounded-lg border transition-all cursor-pointer ${
                isTarget
                  ? 'border-[#0e7c5b] bg-[#ecfdf5]/40 ring-1 ring-[#0e7c5b]'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-slate-900">{cls}</span>
                  <span className="text-xs text-slate-500 hidden sm:inline">
                    — {CLASS_DESCRIPTIONS[cls] || cls}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                      isPositive
                        ? 'bg-red-50 text-red-800 border-red-200'
                        : 'bg-slate-100 text-slate-600 border-slate-200'
                    }`}
                  >
                    {isPositive ? 'POSITIVE' : 'NEGATIVE'}
                  </span>
                  {isTarget && (
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-[#0e7c5b] text-white">
                      Active
                    </span>
                  )}
                </div>
              </div>

              {/* Progress bar representing uncalibrated model probability */}
              <div className="space-y-1">
                <div className="w-full bg-slate-100 rounded-full h-2 relative overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      isPositive ? 'bg-amber-500' : 'bg-slate-400'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
                  />
                  {/* Threshold marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-slate-900 z-10"
                    style={{ left: `${threshold * 100}%` }}
                    title={`Decision Threshold: ${threshold}`}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
                  <span>
                    Uncalibrated Probability:{' '}
                    <strong className="text-slate-800 font-mono">{(prob * 100).toFixed(1)}%</strong>
                  </span>
                  <span>
                    Threshold: <strong className="text-slate-700 font-mono">{threshold.toFixed(2)}</strong>
                    {logit !== null && (
                      <span className="text-slate-400 ml-2">
                        (logit: {logit > 0 ? `+${logit.toFixed(2)}` : logit.toFixed(2)})
                      </span>
                    )}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Mandatory Disclaimer */}
      <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-lg flex items-start gap-2.5 text-xs text-amber-900">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <p className="leading-relaxed text-[11px]">
          <strong>Notice on Model Probabilities:</strong> Values shown are raw mathematical sigmoid outputs from a research 1D ResNet trained on PTB-XL. They represent uncalibrated class associations and do <em>not</em> reflect clinical probability, confirmed diagnosis, or diagnostic certainty. Physician validation is mandatory.
        </p>
      </div>
    </div>
  );
};

export const IntegratedGradientsAttributionPanel = ({
  attribution,
  leadAlignments,
  selectedLead,
  onSelectLead,
}) => {
  if (!attribution) {
    return (
      <div className="p-6 bg-slate-50 border border-slate-200 rounded-xl text-center text-slate-400 text-xs">
        No Integrated Gradients attribution available.
      </div>
    );
  }

  const {
    method,
    baseline,
    integration_steps,
    completeness_error,
    lead_importance = {},
    attribution_status,
  } = attribution;

  // Sort leads by importance
  const leadEntries = Object.entries(lead_importance).sort((a, b) => b[1] - a[1]);
  const maxScore = leadEntries.length > 0 ? Math.max(...leadEntries.map((e) => e[1])) : 1.0;

  return (
    <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[#0e7c5b]" />
            Integrated Gradients Lead Attribution
          </h2>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Method: {method} • Baseline: {baseline} • Steps: {integration_steps}
          </p>
        </div>
        <div className="text-[11px] text-slate-600 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg">
          Completeness Error:{' '}
          <strong className="font-mono text-slate-800">
            {completeness_error !== undefined ? completeness_error.toFixed(4) : 'N/A'}
          </strong>
        </div>
      </div>

      {/* Lead Ranking Bar Chart */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 pb-1 border-b border-slate-100">
          <span>Lead (Ranked by Importance)</span>
          <span>Attribution Score (Absolute IG)</span>
        </div>

        {leadEntries.map(([leadName, score], idx) => {
          const isSelected = selectedLead === leadName;
          const aln = leadAlignments ? leadAlignments[leadName] : null;
          const barWidth = maxScore > 0 ? (score / maxScore) * 100 : 0;

          return (
            <div
              key={leadName}
              onClick={() => onSelectLead && onSelectLead(leadName)}
              className={`p-2 rounded-lg border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                isSelected
                  ? 'border-[#0e7c5b] bg-[#ecfdf5]/50 ring-1 ring-[#0e7c5b]'
                  : 'border-slate-100 hover:border-slate-200 bg-white'
              }`}
            >
              <div className="flex items-center gap-2.5 w-24 shrink-0">
                <span className="w-5 text-center text-xs font-bold text-slate-500">
                  #{idx + 1}
                </span>
                <span className="font-bold text-xs text-slate-900">Lead {leadName}</span>
              </div>

              {/* Relative bar */}
              <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-2 rounded-full transition-all ${
                    aln && aln.alignment_status === 'UNCERTAIN'
                      ? 'bg-amber-500'
                      : 'bg-[#0e7c5b]'
                  }`}
                  style={{ width: `${Math.max(4, barWidth)}%` }}
                />
              </div>

              <div className="w-20 text-right shrink-0">
                <span className="font-mono text-xs font-semibold text-slate-800">
                  {score.toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-[10px] text-slate-400 italic">
        * Higher attribution indicates greater mathematical contribution to the ResNet logit score along the linear path integral. It does not prove anatomical lesion site or underlying physiological causality.
      </p>
    </div>
  );
};
