import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { 
  Activity, Upload, CheckCircle2, AlertCircle, FileText, 
  Clock, ShieldAlert, RefreshCw, Sparkles, ChevronRight,
  Eye, CheckCircle, AlertTriangle, Layers, Info, Filter, ArrowUpRight
} from 'lucide-react';
import { patientService } from '../services/api';
import ecgService from '../services/ecgService';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import { LeadCard, LeadOverlayMap } from '../components/EcgLeadAlignment';
import { EcgClassificationPanel, IntegratedGradientsAttributionPanel } from '../components/EcgExplainabilityPanels';
import { ClinicalMeasurementsPanel } from '../components/EcgClinicalMeasurements';

const EcgAnalysis = () => {
  const [searchParams] = useSearchParams();
  const urlPatientId = searchParams.get('patientId');
  const urlEcgId = searchParams.get('ecgId');

  // Patient & ECG Selection State
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(urlPatientId || '');
  const [patientHistory, setPatientHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [selectedEcgRecord, setSelectedEcgRecord] = useState(null);

  // Upload State
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [successRecord, setSuccessRecord] = useState(null);

  // Explainability Analysis State
  const [analyzing, setAnalyzing] = useState(false);
  const [explainabilityData, setExplainabilityData] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [targetClass, setTargetClass] = useState('');
  const [selectedLead, setSelectedLead] = useState(null);
  const [activeTab, setActiveTab] = useState('alignment'); // 'alignment' | 'attribution' | 'predictions' | 'measurements'

  // Image display state
  const [ecgImageUrl, setEcgImageUrl] = useState('');

  // 1. Initial Load: Fetch Patients
  useEffect(() => {
    loadPatients();
  }, []);

  // 2. Patient selection changed: Reload patient ECG records
  useEffect(() => {
    if (selectedPatientId) {
      loadHistory(selectedPatientId);
    } else {
      setPatientHistory([]);
      setSelectedEcgRecord(null);
      setExplainabilityData(null);
    }
  }, [selectedPatientId]);

  // 3. Selected ECG Record changed: Update image and reset analysis
  useEffect(() => {
    let objectUrl = null;
    let isSubscribed = true;

    if (selectedEcgRecord) {
      setExplainabilityData(null);
      setAnalysisError(null);
      setSelectedLead(null);

      // Fetch image with Bearer token using getEcgImageBlob and create an object URL
      ecgService.getEcgImageBlob(selectedEcgRecord.id)
        .then((res) => {
          if (isSubscribed) {
            objectUrl = URL.createObjectURL(res.data);
            setEcgImageUrl(objectUrl);
          }
        })
        .catch((err) => {
          console.warn('Failed to load image blob via auth, falling back to direct URL', err);
          if (isSubscribed) {
            setEcgImageUrl(ecgService.getEcgImageUrl(selectedEcgRecord.id));
          }
        });
    } else {
      setEcgImageUrl('');
      setExplainabilityData(null);
    }

    return () => {
      isSubscribed = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [selectedEcgRecord]);

  const loadPatients = async () => {
    try {
      const res = await patientService.getPatients('', 0, 100);
      setPatients(res.data || []);
      if (!selectedPatientId && res.data && res.data.length > 0) {
        setSelectedPatientId(res.data[0].id.toString());
      }
    } catch (err) {
      console.error('Failed to load patients', err);
    }
  };

  const loadHistory = async (patientId) => {
    setLoadingHistory(true);
    try {
      const res = await ecgService.getPatientEcgHistory(patientId);
      const records = res.data || [];
      setPatientHistory(records);
      if (records.length > 0) {
        if (urlEcgId) {
          const matched = records.find((r) => r.id.toString() === urlEcgId);
          setSelectedEcgRecord(matched || records[0]);
        } else {
          setSelectedEcgRecord(records[0]);
        }
      } else {
        setSelectedEcgRecord(null);
      }
    } catch (err) {
      console.error('Failed to load ECG history', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setUploadError(null);
    setSuccessRecord(null);

    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(selectedFile.type)) {
      setUploadError('Invalid file format. Please upload a standard PNG or JPEG ECG scan image.');
      setFile(null);
      return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (selectedFile.size > maxSize) {
      setUploadError(`File size exceeds 10MB limit (${(selectedFile.size / (1024 * 1024)).toFixed(1)}MB).`);
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file || !selectedPatientId) {
      setUploadError('Please select a patient and an ECG scan image.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const response = await ecgService.uploadEcg(
        selectedPatientId,
        file,
        (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        }
      );

      const record = response.data.record;
      setSuccessRecord(record);
      setFile(null);
      await loadHistory(selectedPatientId);
      setSelectedEcgRecord(record);
    } catch (err) {
      const msg = err.response?.data?.detail || 'Failed to upload ECG scan image.';
      setUploadError(msg);
    } finally {
      setUploading(false);
    }
  };

  const handleRunExplainability = async (targetClsOverride) => {
    if (!selectedEcgRecord) return;

    setAnalyzing(true);
    setAnalysisError(null);

    const chosenClass = targetClsOverride !== undefined ? targetClsOverride : targetClass;

    try {
      const res = await ecgService.runExplainability(selectedEcgRecord.id, {
        target_class: chosenClass || undefined,
        integration_steps: 50,
      });

      setExplainabilityData(res.data);
      if (res.data?.prediction?.queried_class) {
        setTargetClass(res.data.prediction.queried_class);
      }
    } catch (err) {
      console.error('Explainability error', err);
      let errorMsg = 'Failed to generate explainability analysis.';
      if (err.response) {
        const status = err.response.status;
        const detail = err.response.data?.detail || '';
        if (status === 401) {
          errorMsg = 'Session expired. Please re-authenticate as an authorized doctor.';
        } else if (status === 403) {
          errorMsg = 'Access Forbidden: You are not authorized to analyze records for this patient.';
        } else if (status === 404) {
          errorMsg = `Record not found: ${detail}`;
        } else if (status === 422) {
          errorMsg = `Analysis validation error: ${detail}`;
        } else {
          errorMsg = `Server error (${status}): ${detail || 'Internal processing failed.'}`;
        }
      }
      setAnalysisError(errorMsg);
    } finally {
      setAnalyzing(false);
    }
  };

  const selectedLeadData = selectedLead
    ? selectedLead === 'II_rhythm'
      ? explainabilityData?.rhythm_strip_alignment
      : explainabilityData?.lead_alignments?.[selectedLead]
    : null;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-[#e2ece7] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Activity className="w-6 h-6 text-[#0e7c5b]" />
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              12-Lead ECG Analysis & Explainability
            </h1>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Doctor-in-the-loop decision support: Deep-learning ResNet classification, Integrated Gradients lead attribution, and confidence-gated image alignment.
          </p>
        </div>
      </div>

      <ClinicalDisclaimer />

      {/* Patient & ECG Record Selection Toolbar */}
      <div className="bg-white rounded-xl border border-[#e2ece7] p-4 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs font-bold text-slate-600 uppercase tracking-wide">
            Patient:
          </label>
          <select
            value={selectedPatientId}
            onChange={(e) => setSelectedPatientId(e.target.value)}
            className="input-med text-xs w-48 py-1.5"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.first_name} {p.last_name} (#{p.id})
              </option>
            ))}
          </select>

          {patientHistory.length > 0 && (
            <>
              <label className="text-xs font-bold text-slate-600 uppercase tracking-wide ml-2">
                ECG Record:
              </label>
              <select
                value={selectedEcgRecord?.id || ''}
                onChange={(e) => {
                  const rec = patientHistory.find((r) => r.id.toString() === e.target.value);
                  setSelectedEcgRecord(rec || null);
                }}
                className="input-med text-xs w-56 py-1.5"
              >
                {patientHistory.map((rec) => (
                  <option key={rec.id} value={rec.id}>
                    #{rec.id} — {rec.original_filename} ({new Date(rec.uploaded_at).toLocaleDateString()})
                  </option>
                ))}
              </select>
            </>
          )}
        </div>

        {selectedEcgRecord && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleRunExplainability()}
              disabled={analyzing}
              className="btn-med-primary text-xs py-2 px-3.5 shadow-xs flex items-center gap-1.5"
            >
              {analyzing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Computing Attribution...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Run Explainability Analysis</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Main Grid: Upload & Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: ECG Overview & Metadata */}
        <div className="space-y-6">
          {/* Record Overview Card */}
          <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2 border-b border-slate-100 pb-2.5">
              <FileText className="w-4 h-4 text-[#0e7c5b]" />
              ECG Record Overview
            </h2>

            {selectedEcgRecord ? (
              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">Record ID</span>
                  <span className="font-semibold text-slate-800 font-mono">#{selectedEcgRecord.id}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">Filename</span>
                  <span className="font-medium text-slate-800 truncate max-w-[180px]" title={selectedEcgRecord.original_filename}>
                    {selectedEcgRecord.original_filename}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">Uploaded</span>
                  <span className="font-medium text-slate-700">
                    {new Date(selectedEcgRecord.uploaded_at).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">File Size</span>
                  <span className="font-medium text-slate-700">
                    {(selectedEcgRecord.file_size_bytes / 1024).toFixed(0)} KB
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">Format</span>
                  <span className="font-medium text-slate-700 uppercase">{selectedEcgRecord.mime_type}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-50">
                  <span className="text-slate-400">Processing Status</span>
                  <span className="font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-800 text-[10px]">
                    {selectedEcgRecord.processing_status}
                  </span>
                </div>

                {explainabilityData && (
                  <>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-400">Detected Layout</span>
                      <span className="font-medium text-slate-800">
                        {explainabilityData.global_metadata?.layout_type}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-400">Physical Calibration</span>
                      <span
                        className={`font-semibold px-2 py-0.5 rounded-full text-[10px] ${
                          explainabilityData.global_metadata?.calibration_verified
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {explainabilityData.global_metadata?.calibration_verified
                          ? 'VERIFIED (mV/s)'
                          : 'UNVERIFIED (Pixel)'}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-50">
                      <span className="text-slate-400">Image SHA-256</span>
                      <span className="font-mono text-[10px] text-slate-500 truncate max-w-[160px]" title={explainabilityData.global_metadata?.source_image_hash}>
                        {explainabilityData.global_metadata?.source_image_hash?.slice(0, 16)}...
                      </span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-400">Coverage</span>
                      <span className="font-bold text-slate-800">
                        {explainabilityData.overall_alignment_coverage_pct}%
                        <span className="text-slate-400 font-normal ml-1">
                          ({explainabilityData.confident_lead_count}/12 leads)
                        </span>
                      </span>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-400 py-4 text-center">
                No ECG recording selected.
              </p>
            )}
          </div>

          {/* Upload New Scan Box */}
          <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2 border-b border-slate-100 pb-2.5">
              <Upload className="w-4 h-4 text-[#0e7c5b]" />
              Upload New 12-Lead Scan
            </h2>

            <form onSubmit={handleUpload} className="space-y-3">
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files?.[0]) validateAndSetFile(e.dataTransfer.files[0]);
                }}
                className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors cursor-pointer ${
                  dragOver ? 'border-[#0e7c5b] bg-[#ecfdf5]' : 'border-slate-300 hover:border-slate-400'
                }`}
                onClick={() => document.getElementById('ecg-file-upload').click()}
              >
                <input
                  id="ecg-file-upload"
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && validateAndSetFile(e.target.files[0])}
                />
                <Activity className="w-6 h-6 text-slate-400 mx-auto mb-1.5" />
                <p className="text-xs font-semibold text-slate-700">
                  {file ? file.name : 'Select or drop ECG scan image'}
                </p>
                <p className="text-[10px] text-slate-400 mt-0.5">PNG or JPEG up to 10MB</p>
              </div>

              {uploadProgress > 0 && uploading && (
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-[#0e7c5b] h-1.5 rounded-full transition-all"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              )}

              {uploadError && (
                <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                  <span>{uploadError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={!file || uploading}
                className="btn-med-primary text-xs w-full py-2 shadow-xs"
              >
                {uploading ? 'Uploading...' : 'Upload & Register ECG'}
              </button>
            </form>
          </div>
        </div>

        {/* Right 2 Columns: ECG Scan Image & Alignment Viewer */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-[#0e7c5b]" />
                  Original ECG Scan & Lead Coordinate Alignment
                </h2>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Pixel-accurate lead region outlines gated by reconstruction confidence and geometric verification.
                </p>
              </div>

              {explainabilityData && (
                <div className="flex items-center gap-2">
                  <label className="text-xs font-semibold text-slate-600">Explain Target:</label>
                  <select
                    value={targetClass}
                    onChange={(e) => {
                      setTargetClass(e.target.value);
                      handleRunExplainability(e.target.value);
                    }}
                    className="input-med text-xs py-1 px-2.5 w-28"
                  >
                    {['NORM', 'MI', 'STTC', 'CD', 'HYP'].map((cls) => (
                      <option key={cls} value={cls}>{cls}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* Error Message if Analysis Failed */}
            {analysisError && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-900 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="font-bold">Explainability Analysis Failed</p>
                  <p className="text-red-800 leading-relaxed">{analysisError}</p>
                  <button
                    onClick={() => handleRunExplainability()}
                    className="mt-2 text-xs font-bold text-red-700 underline hover:text-red-900"
                  >
                    Retry Analysis
                  </button>
                </div>
              </div>
            )}

            {/* Interactive Overlay Map */}
            <LeadOverlayMap
              imageUrl={ecgImageUrl}
              leadAlignments={explainabilityData?.lead_alignments}
              rhythmStripAlignment={explainabilityData?.rhythm_strip_alignment}
              selectedLead={selectedLead}
              onSelectLead={setSelectedLead}
              imageWidth={1665}
              imageHeight={1065}
            />

            {/* Selected Lead Inspector Pill */}
            {selectedLeadData && (
              <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 text-sm">
                    {selectedLeadData.is_rhythm_strip ? 'Lead II Rhythm' : `Lead ${selectedLeadData.lead_name}`}
                  </span>
                  <span className="text-slate-400">|</span>
                  <span className="text-slate-600 font-medium">
                    Status: <strong>{selectedLeadData.alignment_status}</strong>
                  </span>
                  <span className="text-slate-400">|</span>
                  <span className="text-slate-600">
                    Confidence: <strong>{Math.round(selectedLeadData.reconstruction_confidence * 100)}%</strong>
                  </span>
                  {selectedLeadData.attribution_rank && (
                    <>
                      <span className="text-slate-400">|</span>
                      <span className="text-slate-600">
                        Rank: <strong>#{selectedLeadData.attribution_rank}</strong>
                      </span>
                    </>
                  )}
                </div>
                <button
                  onClick={() => setSelectedLead(null)}
                  className="text-xs text-slate-500 hover:text-slate-800 underline self-end sm:self-auto"
                >
                  Clear Selection
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tabs / Detailed Analysis Section */}
      {explainabilityData && (
        <div className="space-y-6">
          {/* Navigation Tabs */}
          <div className="flex border-b border-[#e2ece7] gap-4">
            <button
              onClick={() => setActiveTab('alignment')}
              className={`pb-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 flex items-center gap-1.5 ${
                activeTab === 'alignment'
                  ? 'border-[#0e7c5b] text-[#0e7c5b]'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>12-Lead Alignment Cards ({explainabilityData.confident_lead_count}/12 Confident)</span>
            </button>
            <button
              onClick={() => setActiveTab('predictions')}
              className={`pb-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 flex items-center gap-1.5 ${
                activeTab === 'predictions'
                  ? 'border-[#0e7c5b] text-[#0e7c5b]'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Multi-Label Predictions (5 Superclasses)</span>
            </button>
            <button
              onClick={() => setActiveTab('attribution')}
              className={`pb-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 flex items-center gap-1.5 ${
                activeTab === 'attribution'
                  ? 'border-[#0e7c5b] text-[#0e7c5b]'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Integrated Gradients Attribution</span>
            </button>
            <button
              onClick={() => setActiveTab('measurements')}
              className={`pb-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 flex items-center gap-1.5 ${
                activeTab === 'measurements'
                  ? 'border-[#0e7c5b] text-[#0e7c5b]'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Clinical Measurements & Intervals</span>
            </button>
          </div>

          {/* Tab 1: 12-Lead Alignment Grid */}
          {activeTab === 'alignment' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].map((leadName) => (
                  <LeadCard
                    key={leadName}
                    leadName={leadName}
                    alignment={explainabilityData.lead_alignments?.[leadName]}
                    isSelected={selectedLead === leadName}
                    onClick={() => setSelectedLead(leadName)}
                  />
                ))}
              </div>

              {/* Lead II Continuous Rhythm Strip Card */}
              {explainabilityData.rhythm_strip_alignment && (
                <div className="pt-2">
                  <LeadCard
                    leadName="II_rhythm"
                    alignment={explainabilityData.rhythm_strip_alignment}
                    isSelected={selectedLead === 'II_rhythm'}
                    onClick={() => setSelectedLead('II_rhythm')}
                  />
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Classification Results */}
          {activeTab === 'predictions' && (
            <EcgClassificationPanel
              prediction={explainabilityData.prediction}
              selectedClass={targetClass}
              onSelectClass={(cls) => {
                setTargetClass(cls);
                handleRunExplainability(cls);
              }}
            />
          )}

          {/* Tab 3: Integrated Gradients Attribution */}
          {activeTab === 'attribution' && (
            <IntegratedGradientsAttributionPanel
              attribution={explainabilityData.attribution}
              leadAlignments={explainabilityData.lead_alignments}
              selectedLead={selectedLead}
              onSelectLead={setSelectedLead}
            />
          )}

          {/* Tab 4: Clinical Measurements */}
          {activeTab === 'measurements' && (
            <ClinicalMeasurementsPanel
              measurements={explainabilityData.clinical_measurements}
              isCalibrated={explainabilityData.global_metadata?.calibration_verified}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default EcgAnalysis;
