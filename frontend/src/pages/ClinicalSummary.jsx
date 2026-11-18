import React, { useState } from 'react';
import { ClipboardPlus } from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import PatientClinicalDraft from '../components/PatientClinicalDraft';
import PatientPicker from '../components/PatientPicker';

export default function ClinicalSummary() {
  const [patientId, setPatientId] = useState(null);
  return <div className="max-w-5xl space-y-6"><div><h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800"><ClipboardPlus className="w-6 h-6 text-[#0e7c5b]" />AI Clinical Summary</h1><p className="mt-1 text-sm text-slate-500">Create a concise draft from a selected patient’s stored history and extracted reports.</p></div><ClinicalDisclaimer /><PatientPicker onSelect={setPatientId} />{patientId && <PatientClinicalDraft patientId={patientId} />}</div>;
}
