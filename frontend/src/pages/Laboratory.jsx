import React, { useState } from 'react';
import { FlaskConical } from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import LabAnalysis from '../components/LabAnalysis';
import PatientPicker from '../components/PatientPicker';

export default function Laboratory() {
  const [patientId, setPatientId] = useState(null);
  return <div className="max-w-5xl space-y-6"><div><h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800"><FlaskConical className="w-6 h-6 text-[#0e7c5b]" />Laboratory Analysis</h1><p className="mt-1 text-sm text-slate-500">Configured reference rules flag values for clinician review.</p></div><ClinicalDisclaimer /><PatientPicker onSelect={setPatientId} />{patientId && <LabAnalysis patientId={patientId} />}</div>;
}
