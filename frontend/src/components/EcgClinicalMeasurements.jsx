import React from 'react';
import { 
  Heart, Clock, Zap, Activity, AlertCircle, 
  CheckCircle2, AlertTriangle, ShieldAlert
} from 'lucide-react';

export const ClinicalMeasurementsPanel = ({ measurements, isCalibrated }) => {
  if (!measurements) {
    return (
      <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-2">
        <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
          <Heart className="w-4 h-4 text-[#0e7c5b]" />
          Clinical Measurement Summary
        </h3>
        <p className="text-xs text-slate-400">
          Clinical interval measurements not yet computed or unavailable for this recording.
        </p>
      </div>
    );
  }

  const {
    measurement_status,
    global_rhythm = {},
    rhythm_strip_summary = {},
    leads_summary = {},
  } = measurements;

  // Extract primary metrics (preferably from continuous rhythm strip Lead II)
  const meanHr = global_rhythm?.mean_hr_bpm;
  const isRegular = global_rhythm?.is_regular_rhythm;
  const meanRr = global_rhythm?.mean_rr_ms;
  const meanPr = rhythm_strip_summary?.mean_pr_ms;
  const meanQrs = rhythm_strip_summary?.mean_qrs_duration_ms;
  const meanQt = rhythm_strip_summary?.mean_qt_ms;
  const meanQtc = rhythm_strip_summary?.mean_qtc_bazett_ms;
  const stDev = rhythm_strip_summary?.mean_st_deviation_mv;

  const getMetricStatus = (val) => {
    if (val === null || val === undefined) return 'UNAVAILABLE';
    if (!isCalibrated) return 'UNRELIABLE';
    return 'VALID';
  };

  const statusBadge = (stat) => {
    if (stat === 'VALID') {
      return (
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
          VALID
        </span>
      );
    }
    if (stat === 'UNRELIABLE') {
      return (
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">
          UNRELIABLE (UNCALIBRATED)
        </span>
      );
    }
    return (
      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 border border-slate-200">
        UNAVAILABLE
      </span>
    );
  };

  return (
    <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Heart className="w-4 h-4 text-[#0e7c5b]" />
            Clinical Waveform Measurements
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Delineated from 12-lead image traces and Lead II rhythm strip.
          </p>
        </div>
        <div>
          {isCalibrated ? (
            <span className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Verified 25mm/s & 10mm/mV
            </span>
          ) : (
            <span className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-amber-50 text-amber-800 border border-amber-200 flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              Unverified Calibration (Pixel Units)
            </span>
          )}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Heart Rate */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            Heart Rate
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {meanHr ? `${Math.round(meanHr)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">bpm</span>
          </div>
          <div>{statusBadge(getMetricStatus(meanHr))}</div>
        </div>

        {/* RR Interval */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            RR Interval
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {meanRr ? `${Math.round(meanRr)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">ms</span>
          </div>
          <div>{statusBadge(getMetricStatus(meanRr))}</div>
        </div>

        {/* PR Interval */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            PR Interval
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {meanPr ? `${Math.round(meanPr)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">ms</span>
          </div>
          <div>{statusBadge(getMetricStatus(meanPr))}</div>
        </div>

        {/* QRS Duration */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            QRS Duration
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {meanQrs ? `${Math.round(meanQrs)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">ms</span>
          </div>
          <div>{statusBadge(getMetricStatus(meanQrs))}</div>
        </div>

        {/* QT / QTc */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            QT / QTc (Bazett)
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {meanQt ? `${Math.round(meanQt)}` : '—'}
            <span className="text-xs font-normal text-slate-400">/</span>
            {meanQtc ? `${Math.round(meanQtc)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">ms</span>
          </div>
          <div>{statusBadge(getMetricStatus(meanQt))}</div>
        </div>

        {/* ST Deviation */}
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/70 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">
            ST Deviation
          </span>
          <div className="text-lg font-bold text-slate-900 font-mono">
            {stDev !== null && stDev !== undefined ? `${stDev > 0 ? `+${stDev.toFixed(2)}` : stDev.toFixed(2)}` : '—'}
            <span className="text-xs font-normal text-slate-500 ml-1">
              {isCalibrated ? 'mV' : 'px'}
            </span>
          </div>
          <div>{statusBadge(getMetricStatus(stDev))}</div>
        </div>
      </div>

      {!isCalibrated && (
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-2.5 text-xs text-amber-900">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <p className="leading-relaxed text-[11px]">
            <strong>Calibration Gating Enforced:</strong> Physical paper calibration (10 mm/mV, 25 mm/s) could not be verified from scan calibration pulses. Metric durations and amplitudes are labeled unverified and must not be used for physical quantitative decisions without manual caliper verification.
          </p>
        </div>
      )}
    </div>
  );
};
