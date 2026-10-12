import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { patientService } from '../services/api';
import {
  User, Calendar, HeartPulse, PlusCircle, Activity, TrendingUp,
  Bot, FileText, FlaskConical, ArrowLeft, Pencil
} from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import { patientCode } from '../utils/format';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';
import LabAnalysis from '../components/LabAnalysis';
import EvidenceSearch from '../components/EvidenceSearch';
import XaiAnalysis from '../components/XaiAnalysis';
import PatientClinicalDraft from '../components/PatientClinicalDraft';

const PatientDetails = () => {
  const { id } = useParams();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Medical history modal state
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [historyForm, setHistoryForm] = useState({
    condition: '',
    description: '',
    diagnosis_date: new Date().toISOString().split('T')[0],
    status: 'Active'
  });
  const [savingHistory, setSavingHistory] = useState(false);

  // Bumped after a successful report upload so <PatientReports> re-fetches its list
  const [reportsRefreshKey, setReportsRefreshKey] = useState(0);

  useEffect(() => {
    loadPatient();
  }, [id]);

  const loadPatient = async () => {
    try {
      setLoading(true);
      const res = await patientService.getPatientById(id);
      setPatient(res.data);
    } catch (err) {
      console.error(err);
      setError('Unable to load patient profile.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddHistory = async (e) => {
    e.preventDefault();
    if (!historyForm.condition.trim()) return;
    setSavingHistory(true);
    try {
      await patientService.addMedicalHistory(id, historyForm);
      setHistoryForm({
        condition: '',
        description: '',
        diagnosis_date: new Date().toISOString().split('T')[0],
        status: 'Active'
      });
      setShowHistoryModal(false);
      loadPatient();
    } catch (err) {
      console.error(err);
      alert('Failed to save medical history.');
    } finally {
      setSavingHistory(false);
    }
  };

  if (loading) {
    return <div className="py-20 text-center text-sm text-slate-400">Loading electronic patient profile...</div>;
  }

  if (!patient) {
    return (
      <div className="text-center py-20">
        <p className="text-base font-semibold text-slate-700">Patient not found</p>
        <Link to="/patients" className="mt-4 btn-med-primary text-xs inline-flex items-center space-x-1">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Directory</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Profile Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[#e2ece7]">
        <div className="flex items-center space-x-4">
          <Link to="/patients" className="p-2 text-slate-500 hover:text-slate-800 hover:bg-white rounded-lg border border-[#e2ece7] transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div className="w-12 h-12 rounded-2xl bg-[#0f4c3a] text-white font-bold text-lg flex items-center justify-center shadow-sm">
            {patient.first_name.charAt(0)}{patient.last_name.charAt(0)}
          </div>
          <div>
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
                {patient.first_name} {patient.last_name}
              </h1>
              <span className="text-xs font-mono font-semibold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded border border-emerald-200">
                {patientCode(patient.id)}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              {patient.gender} • DOB: {patient.date_of_birth || 'N/A'} • Blood Group: {patient.blood_group || 'Unrecorded'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 self-start sm:self-auto">
          <Link to={`/patients/${patient.id}/edit`} className="btn-med-secondary text-xs flex items-center space-x-1.5">
            <Pencil className="w-4 h-4" />
            <span>Edit Patient</span>
          </Link>
          <button
            onClick={() => setShowHistoryModal(true)}
            className="btn-med-primary text-xs flex items-center space-x-1.5 shadow-sm"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Add Medical Condition</span>
          </button>
        </div>
      </div>

      <ClinicalDisclaimer />

      {/* Grid Layout: Demographics & Medical History */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Demographics & Contact Card */}
        <div className="med-card p-5 space-y-4">
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center space-x-2 border-b border-slate-100 pb-2.5">
            <User className="w-4 h-4 text-[#0e7c5b]" />
            <span>Demographics & Contact</span>
          </h2>

          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Full Name</span>
              <span className="font-semibold text-slate-800">{patient.first_name} {patient.last_name}</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Gender</span>
              <span className="font-semibold text-slate-800">{patient.gender}</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Date of Birth</span>
              <span className="font-semibold text-slate-800">{patient.date_of_birth || 'Not recorded'}</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Blood Group</span>
              <span className="font-semibold text-rose-600 bg-rose-50 px-2 py-0.5 rounded border border-rose-100">
                {patient.blood_group || 'N/A'}
              </span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Phone</span>
              <span className="font-semibold text-slate-800">{patient.phone || 'N/A'}</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-50">
              <span className="text-slate-400">Email</span>
              <span className="font-semibold text-slate-800 truncate max-w-[150px]">{patient.email || 'N/A'}</span>
            </div>

            <div className="pt-1">
              <span className="text-slate-400 block mb-1">Residential Address</span>
              <span className="text-slate-700 font-medium leading-relaxed block bg-slate-50 p-2 rounded-lg border border-slate-100">
                {patient.address || 'No residential address on file.'}
              </span>
            </div>
          </div>
        </div>

        {/* Right 2 Columns: Medical History Timeline */}
        <div className="lg:col-span-2 med-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center space-x-2">
                <HeartPulse className="w-4 h-4 text-[#0e7c5b]" />
                <span>Recorded Medical History</span>
              </h2>
              <span className="text-xs bg-emerald-50 text-[#0f4c3a] border border-emerald-200 px-2.5 py-0.5 rounded-full font-semibold">
                {patient.medical_history?.length || 0} entries
              </span>
            </div>

            {patient.medical_history?.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-400">
                <Activity className="w-8 h-8 mx-auto mb-2 text-slate-300" />
                No medical conditions recorded for this patient.
                <div className="mt-3">
                  <button
                    onClick={() => setShowHistoryModal(true)}
                    className="btn-med-secondary text-xs inline-flex items-center space-x-1"
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    <span>Add Initial Diagnosis</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {patient.medical_history.map((hist) => (
                  <div key={hist.id} className="p-3.5 rounded-xl bg-[#fbfdfc] border border-[#e2ece7] hover:border-emerald-300 transition-colors">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-bold text-slate-800 text-sm">{hist.condition}</span>
                          <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full border ${
                            hist.status === 'Active' ? 'bg-amber-50 text-amber-800 border-amber-200' : 'bg-emerald-50 text-emerald-800 border-emerald-200'
                          }`}>
                            {hist.status}
                          </span>
                        </div>
                        {hist.description && (
                          <p className="text-xs text-slate-600 mt-1.5 leading-relaxed bg-white p-2 rounded-lg border border-slate-100">
                            {hist.description}
                          </p>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 flex items-center space-x-1 flex-shrink-0 ml-2">
                        <Calendar className="w-3 h-3" />
                        <span>{hist.diagnosis_date || new Date(hist.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Synchronized with PostgreSQL `medical_history` table</span>
            <span className="text-[#0e7c5b] font-medium">Doctor Verified</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 pt-2"><TrendingUp className="w-5 h-5 text-[#0e7c5b]" /><div><h2 className="text-base font-bold text-slate-800">Unified Clinical Decision Support</h2><p className="text-xs text-slate-500">Separate, clinician-reviewable record, laboratory, model, evidence, and summary sections.</p></div></div>

      {/* Multi-Section Modular Grid for Upcoming AI Roadmap (Weeks 2–12) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Module 1: Uploaded Reports & OCR (Week 2) */}
        <div className="med-card p-5 opacity-90">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-slate-500" />
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Medical Reports & OCR</h3>
            </div>
            <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-1.5 py-0.5 rounded font-medium">
              Live
            </span>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            PDF/Image report upload with Tesseract OCR text extraction and clinical entity parsing.
          </p>
          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-400">
            <div className="flex items-center justify-between">
              <span>Status: Active Pipeline</span>
            </div>
            <div className="mt-3">
              <ReportUploader
                patientId={patient.id}
                onUploaded={() => {
                  loadPatient();
                  setReportsRefreshKey((k) => k + 1);
                }}
              />
            </div>
            <div className="mt-4">
              <PatientReports patientId={patient.id} refreshKey={reportsRefreshKey} onConditionApproved={loadPatient} />
            </div>
          </div>
        </div>

        <LabAnalysis patientId={patient.id} refreshKey={reportsRefreshKey} />

        <EvidenceSearch patientId={patient.id} />

        <XaiAnalysis patientId={patient.id} />

        <PatientClinicalDraft patientId={patient.id} />
      </div>

      {/* Add Medical History Modal */}
      {showHistoryModal && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl border border-[#e2ece7] max-w-lg w-full p-6 shadow-xl animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h3 className="text-base font-bold text-slate-800 flex items-center space-x-2">
                <HeartPulse className="w-5 h-5 text-[#0e7c5b]" />
                <span>Add Medical History Record</span>
              </h3>
              <button
                onClick={() => setShowHistoryModal(false)}
                className="text-slate-400 hover:text-slate-600 p-1"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleAddHistory} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Medical Condition / Diagnosis *</label>
                <input
                  type="text"
                  required
                  value={historyForm.condition}
                  onChange={(e) => setHistoryForm({ ...historyForm, condition: e.target.value })}
                  placeholder="e.g. Hypertension, Diabetic Neuropathy, Asthma"
                  className="input-med text-xs"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Diagnosis Date</label>
                  <input
                    type="date"
                    value={historyForm.diagnosis_date}
                    onChange={(e) => setHistoryForm({ ...historyForm, diagnosis_date: e.target.value })}
                    className="input-med text-xs"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Status</label>
                  <select
                    value={historyForm.status}
                    onChange={(e) => setHistoryForm({ ...historyForm, status: e.target.value })}
                    className="input-med text-xs"
                  >
                    <option value="Active">Active</option>
                    <option value="Chronic">Chronic</option>
                    <option value="In Remission">In Remission</option>
                    <option value="Resolved">Resolved</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Notes & Observations</label>
                <textarea
                  rows="3"
                  value={historyForm.description}
                  onChange={(e) => setHistoryForm({ ...historyForm, description: e.target.value })}
                  placeholder="Include treatment regimen, medication response, or relevant clinical observations..."
                  className="input-med text-xs"
                ></textarea>
              </div>

              <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setShowHistoryModal(false)}
                  className="btn-med-secondary text-xs py-2 px-3"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={savingHistory}
                  className="btn-med-primary text-xs py-2 px-4"
                >
                  {savingHistory ? 'Saving to Database...' : 'Save Diagnosis'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientDetails;

/* Small helper components for uploading and listing reports */
export const ReportUploader = ({ patientId, onUploaded }) => {
  const [file, setFile] = React.useState(null);
  const [uploading, setUploading] = React.useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert('Select a file to upload');
    const fd = new FormData();
    fd.append('report_file', file);
    try {
      setUploading(true);
      await patientService.uploadReport(patientId, fd);
      setFile(null);
      if (onUploaded) onUploaded();
      alert('Report uploaded');
    } catch (err) {
      console.error(err);
      alert('Failed to upload report');
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <label className="block text-xs font-semibold text-slate-700">Upload PDF / Image</label>
      <input type="file" accept=".pdf,image/*" onChange={(e) => setFile(e.target.files[0])} />
      <div className="pt-2">
        <button type="submit" disabled={uploading} className="btn-med-primary text-xs py-2">
          {uploading ? 'Uploading...' : 'Upload Report & OCR'}
        </button>
      </div>
    </form>
  );
};

export const PatientReports = ({ patientId, refreshKey, onConditionApproved }) => {
  const [reports, setReports] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  // Report viewer state (moved here from PatientDetails: this is the component
  // that actually reads/writes it, so it must live in this scope)
  const [viewerText, setViewerText] = React.useState(null);
  const [viewerTitle, setViewerTitle] = React.useState('');
  const [reviewReport, setReviewReport] = React.useState(null);

  const load = async () => {
    try {
      setLoading(true);
      const res = await patientService.listReports(patientId);
      setReports(res.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => { load(); }, [patientId, refreshKey]);

  if (loading) return <div className="text-xs text-slate-400">Loading reports...</div>;
  if (!reports.length) return <div className="text-xs text-slate-400">No reports uploaded yet.</div>;

  return (
    <>
    <div className="space-y-3">
      {reports.map(r => (
        <div key={r.id} className="p-3 rounded-lg border border-slate-100 bg-white">
          <div className="flex items-center justify-between">
            <div className="font-semibold text-sm">{r.filename}</div>
            <div className="text-[11px] text-slate-400">{new Date(r.uploaded_at).toLocaleString()}</div>
          </div>
          {r.extracted_text ? (
            <div className="mt-2 text-xs text-slate-700 prose max-w-none">
              <ReactMarkdown>{r.extracted_text.slice(0, 400)}</ReactMarkdown>
            </div>
          ) : (
            <div className="mt-2 text-xs text-slate-500">No extracted text available (OCR not run or failed).</div>
          )}
          <div className="mt-3 flex items-center space-x-2">
            {!r.extracted_text && (
              <button
                onClick={async () => {
                  try {
                    await api.post(`/patients/${patientId}/reports/${r.id}/ocr`);
                    await load();
                  } catch (err) {
                    console.error(err);
                    alert(err.response?.data?.detail || 'OCR could not extract text from this report.');
                  }
                }}
                className="btn-med-primary text-xs py-1 px-2"
              >
                Retry OCR
              </button>
            )}
            {r.extracted_text && (
              <button
                onClick={() => setReviewReport(r)}
                className="btn-med-primary text-xs py-1 px-2"
              >
                Review extracted findings
              </button>
            )}
            <button
              onClick={async () => {
                try {
                  const res = await patientService.getReport(patientId, r.id);
                  setViewerTitle(r.filename);
                  setViewerText(res.data.extracted_text || 'No extracted text available.');
                } catch (err) {
                  console.error(err);
                  alert('Failed to load report text');
                }
              }}
              className="btn-med-secondary text-xs py-1 px-2"
            >
              View
            </button>
            <button
              onClick={async () => {
                try {
                  const resp = await api.get(`/patients/${patientId}/reports/${r.id}/download`, { responseType: 'blob' });
                  const url = window.URL.createObjectURL(new Blob([resp.data]));
                  const link = document.createElement('a');
                  link.href = url;
                  link.setAttribute('download', r.filename || 'report');
                  document.body.appendChild(link);
                  link.click();
                  link.remove();
                } catch (err) {
                  console.error(err);
                  alert('Download failed (check auth or server file)');
                }
              }}
              className="btn-med-secondary text-xs py-1 px-2"
            >
              Download
            </button>
          </div>
        </div>
      ))}
    </div>
    {viewerText && (
      <ReportViewerModal
        title={viewerTitle}
        text={viewerText}
        onClose={() => { setViewerText(null); setViewerTitle(''); }}
      />
    )}
    {reviewReport && (
      <ClinicalExtractionReview
        report={reviewReport}
        patientId={patientId}
        onClose={() => setReviewReport(null)}
        onApproved={() => { onConditionApproved?.(); setReviewReport(null); }}
      />
    )}
    </>
  );
};

const ClinicalExtractionReview = ({ report, patientId, onClose, onApproved }) => {
  const [entities, setEntities] = React.useState(report.clinical_entities || null);
  const [loading, setLoading] = React.useState(!report.clinical_entities);
  const [saving, setSaving] = React.useState('');
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    if (entities) return;
    patientService.extractEntities(patientId, report.id)
      .then(({ data }) => setEntities(data.clinical_entities || {}))
      .catch(() => setError('Unable to process the extracted text.'))
      .finally(() => setLoading(false));
  }, [patientId, report.id, entities]);

  const approve = async (condition) => {
    setSaving(condition.text);
    setError('');
    try {
      await patientService.approveCondition(patientId, report.id, {
        condition: condition.text,
        description: `Added after clinician review of ${report.filename}.`,
      });
      onApproved();
    } catch (err) {
      setError(err.response?.data?.detail || 'Unable to add the reviewed condition.');
    } finally {
      setSaving('');
    }
  };

  const groups = [
    ['Possible conditions', 'conditions'], ['Possible medications', 'medications'],
    ['Possible laboratory values', 'lab_values'], ['Clinical events', 'clinical_events'], ['Dates', 'dates'],
  ];
  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl border border-[#e2ece7] shadow-xl max-w-2xl w-full max-h-[85vh] overflow-auto p-6">
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <h3 className="font-bold text-slate-800">Review extracted clinical information</h3>
            <p className="text-xs text-slate-500 mt-1">{report.filename}</p>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-800">×</button>
        </div>
        <div className="my-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          These are possible text matches, not diagnoses or treatment advice. Verify against the source report before adding anything to the patient record.
        </div>
        {loading && <p className="text-sm text-slate-500">Extracting possible findings…</p>}
        {error && <p className="text-xs text-rose-700 mb-3">{error}</p>}
        {!loading && groups.map(([label, key]) => (
          <section key={key} className="mb-4">
            <h4 className="text-xs font-bold uppercase tracking-wide text-slate-500 mb-2">{label}</h4>
            {entities?.[key]?.length ? (
              <div className="space-y-2">
                {entities[key].map((item, index) => (
                  <div key={`${key}-${index}`} className="flex items-center justify-between gap-3 rounded-lg border border-slate-100 p-2.5 text-xs">
                    <span className="text-slate-700">{item.text}</span>
                    {key === 'conditions' && (
                      <button disabled={saving === item.text} onClick={() => approve(item)} className="btn-med-secondary text-[11px] py-1 px-2 whitespace-nowrap">
                        {saving === item.text ? 'Adding…' : 'Confirm & add'}
                      </button>
                    )}
                    {key === 'lab_values' && (
                      <button
                        disabled={saving === item.text}
                        onClick={async () => {
                          setSaving(item.text);
                          setError('');
                          try {
                            await patientService.addLabResult(patientId, {
                              test_name: item.name || 'Glucose',
                              value: Number(item.value),
                              unit: item.unit || (item.name?.toLowerCase() === 'glucose' ? 'mg/dL' : undefined),
                              test_date: new Date().toISOString().slice(0, 10),
                            });
                            onApproved();
                          } catch (err) {
                            setError(err.response?.data?.detail || 'Unable to analyse and add lab value.');
                          } finally {
                            setSaving('');
                          }
                        }}
                        className="btn-med-secondary text-[11px] py-1 px-2 whitespace-nowrap"
                      >
                        {saving === item.text ? 'Adding…' : 'Analyse in Lab Engine'}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            ) : <p className="text-xs text-slate-400">No matches found.</p>}
          </section>
        ))}
      </div>
    </div>
  );
};

// Viewer modal used by PatientReports above
const ReportViewerModal = ({ title, text, onClose }) => {
  if (!text) return null;
  return (
    <div className="fixed inset-0 bg-slate-900/40 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-2xl border border-[#e2ece7] max-w-3xl w-full p-6 shadow-xl overflow-auto max-h-[80vh]">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
          <h3 className="text-base font-bold text-slate-800">{title}</h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-700">✕</button>
        </div>
        <div className="text-xs text-slate-700 leading-relaxed whitespace-pre-wrap">
          {text}
        </div>
      </div>
    </div>
  );
};
