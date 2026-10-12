import React, { useState } from 'react';
import { FileText } from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import PatientPicker from '../components/PatientPicker';
import { PatientReports, ReportUploader } from './PatientDetails';
import PatientHistory from '../components/PatientHistory';

export default function Reports() {
  const [patientId, setPatientId] = useState(null), [refresh, setRefresh] = useState(0);
  const updated = () => setRefresh((x) => x + 1);
  return <div className="max-w-6xl space-y-6"><div><h1 className="flex items-center gap-2 text-2xl font-bold text-slate-800"><FileText className="w-6 h-6 text-[#0e7c5b]" />Medical Reports & OCR</h1><p className="mt-1 text-sm text-slate-500">Upload a report, inspect extracted text, review possible NLP findings, and confirm patient history.</p></div><ClinicalDisclaimer /><PatientPicker onSelect={setPatientId} />{patientId && <div className="grid grid-cols-1 xl:grid-cols-2 gap-6"><section className="med-card p-5"><h2 className="text-sm font-bold text-slate-800">Upload and review reports</h2><div className="mt-4"><ReportUploader patientId={patientId} onUploaded={updated} /></div><div className="mt-5 border-t border-slate-100 pt-5"><PatientReports patientId={patientId} refreshKey={refresh} onConditionApproved={updated} /></div></section><PatientHistory patientId={patientId} refreshKey={refresh} /></div>}</div>;
}
