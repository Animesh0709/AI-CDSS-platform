import React, { useState } from 'react';
import { BookOpen } from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import EvidenceSearch from '../components/EvidenceSearch';
import PatientPicker from '../components/PatientPicker';

export default function MedicalEvidence() {
  const [patientId, setPatientId] = useState(null);
  return <div className="max-w-5xl space-y-6"><div><h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800"><BookOpen className="w-6 h-6 text-[#0e7c5b]" />Medical Evidence</h1><p className="mt-1 text-sm text-slate-500">Search curated guideline evidence, optionally with patient record navigation.</p></div><ClinicalDisclaimer /><PatientPicker onSelect={setPatientId} /><EvidenceSearch patientId={patientId} /></div>;
}
