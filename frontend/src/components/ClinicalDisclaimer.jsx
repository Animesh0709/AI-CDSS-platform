import React from 'react';
import { ShieldAlert } from 'lucide-react';

const ClinicalDisclaimer = ({ compact = false }) => {
  if (compact) {
    return (
      <div className="flex items-center space-x-2 text-xs text-amber-800 bg-amber-50/80 px-3 py-1.5 rounded-lg border border-amber-200/70">
        <ShieldAlert className="w-3.5 h-3.5 flex-shrink-0 text-amber-600" />
        <span>Clinical Decision Support System (CDSS) • Physician review mandatory • Not autonomous diagnosis</span>
      </div>
    );
  }

  return (
    <div className="bg-emerald-50/60 border border-emerald-200/80 rounded-xl p-3.5 mb-6 flex items-start space-x-3 text-sm text-emerald-950">
      <div className="p-1.5 bg-emerald-100/90 text-emerald-800 rounded-lg mt-0.5">
        <ShieldAlert className="w-4 h-4" />
      </div>
      <div className="flex-1">
        <div className="font-semibold text-emerald-900 flex items-center justify-between">
          <span>Doctor-in-the-Loop Clinical Decision Support Prototype</span>
          <span className="text-[11px] font-medium uppercase tracking-wider bg-emerald-200/60 text-emerald-900 px-2 py-0.5 rounded-full">
            Physician Supervised
          </span>
        </div>
        <p className="text-emerald-800/90 text-xs mt-0.5 leading-relaxed">
          This system organizes patient metrics and provides statistical risk estimates to assist certified doctors. 
          It does not autonomously diagnose medical conditions or issue prescriptions. All insights require clinical validation.
        </p>
      </div>
    </div>
  );
};

export default ClinicalDisclaimer;
