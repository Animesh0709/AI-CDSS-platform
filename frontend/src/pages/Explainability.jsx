import React, { useEffect, useState } from 'react';
import { BrainCircuit } from 'lucide-react';
import { patientService } from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import XaiAnalysis from '../components/XaiAnalysis';

export default function Explainability() {
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState('');
  const [error, setError] = useState('');
  useEffect(() => { patientService.getPatients().then(({ data }) => { setPatients(data); if (data[0]) setPatientId(String(data[0].id)); }).catch(() => setError('Unable to load your patients.')); }, []);
  return <div className="max-w-6xl space-y-6"><div><h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800"><BrainCircuit className="w-6 h-6 text-[#0e7c5b]" />Risk Explainability</h1><p className="mt-1 text-sm text-slate-500">Interpret readmission risk predictions using SHAP, LIME, What-If analysis, and Model Overview.</p></div><ClinicalDisclaimer />
    <section className="med-card p-5"><label className="block text-xs font-semibold text-slate-700 mb-1">Patient for local explanation</label><select value={patientId} onChange={(e) => setPatientId(e.target.value)} className="input-med max-w-md text-sm"><option value="">Select a patient</option>{patients.map((p) => <option key={p.id} value={p.id}>{p.first_name} {p.last_name}</option>)}</select>{error && <p className="mt-2 text-xs text-rose-700">{error}</p>}{!patients.length && !error && <p className="mt-2 text-xs text-slate-500">Add a patient before running an individual prediction.</p>}</section>
    {patientId && <XaiAnalysis patientId={patientId} />}
  </div>;
}
