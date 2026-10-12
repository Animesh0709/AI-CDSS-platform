import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, Brain, Activity, Sliders, Eye, RefreshCw, BarChart3, AlertCircle, 
  CheckCircle2, Info, ChevronRight, Download, ShieldCheck, Microscope,
  Upload, X, Image as ImageIcon, FileCode, Check
} from 'lucide-react';
import { segmentationService } from '../services/api';

const TumorSegmentation = () => {
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState('');
  const [selectedModel, setSelectedModel] = useState('attunet3d');
  const [activePlane, setActivePlane] = useState('axial');
  const [sliceIndex, setSliceIndex] = useState(78);
  const [viewMode, setViewMode] = useState('overlay'); // 'raw', 'overlay', 'uncertainty'
  
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [predictionData, setPredictionData] = useState(null);
  const [evalReport, setEvalReport] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);

  // Upload modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadModel, setUploadModel] = useState('attunet3d');
  const fileInputRef = useRef(null);

  // Fetch initial cases and eval report
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setInitialLoading(true);
    try {
      const [casesRes, evalRes, metaRes] = await Promise.all([
        segmentationService.getCases(),
        segmentationService.getEvalReport(),
        segmentationService.getMetadata()
      ]);
      setCases(casesRes.data || []);
      if (casesRes.data && casesRes.data.length > 0) {
        setSelectedCase(casesRes.data[0]);
      }
      setEvalReport(evalRes.data);
      setMetadata(metaRes.data);
    } catch (err) {
      console.error("Failed to load initial segmentation data:", err);
      setError("Failed to communicate with segmentation backend. Make sure the backend is running on port 8000.");
    } finally {
      setInitialLoading(false);
    }
  };

  const handleRunSegmentation = async () => {
    if (!selectedCase) return;
    setLoading(true);
    setError(null);
    try {
      const payload = {
        subject_identifier: selectedCase,
        model_name: selectedModel,
        axial_slice: sliceIndex,
        coronal_slice: 120,
        sagittal_slice: 120
      };
      const res = await segmentationService.predict(payload);
      setPredictionData(res.data);
    } catch (err) {
      console.error("Segmentation inference error:", err);
      setError(err.response?.data?.detail || "Segmentation pipeline failed.");
    } finally {
      setLoading(false);
    }
  };

  // Trigger prediction on initial case load
  useEffect(() => {
    if (selectedCase && !predictionData) {
      handleRunSegmentation();
    }
  }, [selectedCase]);

  const handleFileSelect = (file) => {
    if (!file) return;
    setUploadError(null);
    const validExts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.nii', '.nii.gz'];
    const lowerName = file.name.toLowerCase();
    const isValid = validExts.some(ext => lowerName.endsWith(ext));
    if (!isValid) {
      setUploadError("Invalid file type. Please upload a PNG, JPG, BMP, TIFF image or NIfTI (.nii, .nii.gz) MRI scan volume.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setUploadError("File exceeds 50 MB limit.");
      return;
    }
    setUploadFile(file);
    if (!lowerName.endsWith('.nii') && !lowerName.endsWith('.nii.gz')) {
      const reader = new FileReader();
      reader.onload = (e) => setUploadPreview(e.target.result);
      reader.readAsDataURL(file);
    } else {
      setUploadPreview(null);
    }
  };

  const handleUploadAndSegment = async () => {
    if (!uploadFile) return;
    setUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      formData.append('model_name', uploadModel);
      formData.append('axial_slice', 78);
      formData.append('coronal_slice', 120);
      formData.append('sagittal_slice', 120);

      const res = await segmentationService.uploadAndPredict(formData);
      setPredictionData(res.data);
      setSelectedModel(uploadModel);
      setSelectedCase(res.data.subject_identifier);

      // Refresh cases list to prioritize uploaded scan
      const casesRes = await segmentationService.getCases();
      setCases(casesRes.data || []);

      setShowUploadModal(false);
      setUploadFile(null);
      setUploadPreview(null);
    } catch (err) {
      console.error("Upload & segmentation error:", err);
      setUploadError(err.response?.data?.detail || "Upload & segmentation failed.");
    } finally {
      setUploading(false);
    }
  };

  const currentSlice = predictionData?.slices?.[activePlane];

  const getImageSrc = () => {
    if (!currentSlice) return null;
    if (viewMode === 'raw') return currentSlice.raw_mri_base64;
    if (viewMode === 'uncertainty') return currentSlice.uncertainty_map_base64;
    return currentSlice.overlay_mask_base64;
  };

  if (initialLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-slate-500">
        <div className="w-12 h-12 border-4 border-[#0e7c5b] border-t-transparent rounded-full animate-spin" />
        <div className="text-center">
          <p className="font-semibold text-slate-800">Loading Segmentation Workstation</p>
          <p className="text-sm text-slate-500 mt-1">Connecting to backend &amp; loading available MRI cases…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#e2ece7] pb-5">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#ecfdf5] text-[#0f4c3a] border border-[#a7f3d0]">
              Multimodal 3D Voxel Segmentation
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
              Primary Cohort: UPenn-GBM (TCIA)
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-200">
              External Cohort: MSD Task01 (Pending Remote Bandwidth)
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200">
              4-Channel (FLAIR, T1, T1ce, T2)
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Brain className="w-7 h-7 text-[#0e7c5b]" />
            Brain Tumor 3D Volumetric Segmentation Workstation
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Deep 3D voxel segmentation across 3 architectures with sub-tumor volume quantification (cm³) and spatial uncertainty maps.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowUploadModal(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-white border border-[#0e7c5b] text-[#0e7c5b] hover:bg-[#ecfdf5] rounded-xl text-sm font-medium shadow-xs transition-all"
          >
            <Upload className="w-4 h-4" />
            <span>Upload MRI Scan</span>
          </button>
          <button
            onClick={handleRunSegmentation}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#0e7c5b] hover:bg-[#0c6b4e] text-white rounded-xl text-sm font-medium shadow-sm transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Segmenting 3D Volume...' : 'Run Segmentation'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Clinical Grid: Left Controls & Viewer, Right Volumetrics & Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Interactive 3D Multi-Planar Viewer (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white rounded-2xl border border-[#e2ece7] shadow-xs overflow-hidden">
            {/* Viewer Top Bar */}
            <div className="p-4 border-b border-[#e2ece7] bg-slate-50 flex flex-wrap items-center justify-between gap-3">
              {/* Select Case & Model */}
              <div className="flex items-center gap-2">
                <select
                  value={selectedCase}
                  onChange={(e) => { setSelectedCase(e.target.value); }}
                  className="text-xs font-medium bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-[#0e7c5b] max-w-[200px] truncate"
                >
                  {cases.map((c) => (
                    <option key={c} value={c}>
                      {c.startsWith('UPLOAD_') ? `📁 [Uploaded] ${c.replace('UPLOAD_', '')}` : c}
                    </option>
                  ))}
                </select>

                <button
                  onClick={() => setShowUploadModal(true)}
                  title="Upload your own MRI scan"
                  className="p-1.5 text-[#0e7c5b] hover:bg-[#ecfdf5] border border-slate-300 rounded-lg transition-colors flex items-center gap-1 text-xs font-medium"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Upload</span>
                </button>

                <select
                  value={selectedModel}
                  onChange={(e) => { setSelectedModel(e.target.value); }}
                  className="text-xs font-medium bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-[#0e7c5b]"
                >
                  <option value="attunet3d">3D Attention U-Net</option>
                  <option value="unet3d">3D U-Net</option>
                  <option value="nnunet">3D nnU-Net v2 (DynUNet)</option>
                </select>
              </div>

              {/* View Modes */}
              <div className="flex items-center bg-white border border-slate-200 rounded-lg p-0.5">
                <button
                  onClick={() => setViewMode('raw')}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${viewMode === 'raw' ? 'bg-slate-800 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
                >
                  Raw MRI
                </button>
                <button
                  onClick={() => setViewMode('overlay')}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${viewMode === 'overlay' ? 'bg-[#0e7c5b] text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
                >
                  Sub-region Masks
                </button>
                <button
                  onClick={() => setViewMode('uncertainty')}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${viewMode === 'uncertainty' ? 'bg-amber-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
                >
                  Spatial Uncertainty
                </button>
              </div>
            </div>

            {/* Canvas / Viewport */}
            <div className="relative bg-slate-950 flex items-center justify-center min-h-[380px] p-4">
              {getImageSrc() ? (
                <div className="relative max-w-full">
                  <img
                    src={getImageSrc()}
                    alt="MRI Slice"
                    className="max-h-[420px] rounded-lg shadow-lg object-contain border border-slate-800"
                  />
                  {/* Overlay Legend Badge */}
                  <div className="absolute top-3 right-3 bg-black/75 backdrop-blur-md px-3 py-2 rounded-lg border border-white/10 text-[11px] text-white space-y-1">
                    <div className="font-semibold text-slate-300 uppercase tracking-wider text-[9px]">Legend</div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#10b981]" />
                      <span>Whole Tumor (WT)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]" />
                      <span>Tumor Core (TC)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" />
                      <span>Enhancing (ET)</span>
                    </div>
                  </div>

                  {/* Orientation marker */}
                  <div className="absolute bottom-3 left-3 bg-black/60 px-2 py-1 rounded text-[10px] text-slate-400 font-mono">
                    {activePlane.toUpperCase()} | Slice {sliceIndex}/{currentSlice?.total_slices || 155}
                  </div>
                </div>
              ) : (
                <div className="text-center text-slate-500 py-16 space-y-2">
                  <RefreshCw className="w-8 h-8 animate-spin mx-auto text-slate-600" />
                  <p className="text-sm">Generating 3D volume reconstruction...</p>
                </div>
              )}
            </div>

            {/* Navigation Plane & Slice Controls */}
            <div className="p-4 bg-white border-t border-[#e2ece7] space-y-3">
              <div className="flex items-center justify-between">
                {/* Plane Toggle */}
                <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl">
                  {['axial', 'coronal', 'sagittal'].map((plane) => (
                    <button
                      key={plane}
                      onClick={() => setActivePlane(plane)}
                      className={`px-3 py-1 text-xs font-semibold rounded-lg capitalize transition-all ${
                        activePlane === plane 
                          ? 'bg-white text-[#0e7c5b] shadow-xs border border-slate-200' 
                          : 'text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      {plane}
                    </button>
                  ))}
                </div>

                <div className="text-xs text-slate-500">
                  Latency: <span className="font-semibold text-slate-800">{predictionData?.inference_latency_ms || 0} ms</span>
                </div>
              </div>

              {/* Slice Slider */}
              <div className="flex items-center gap-3">
                <span className="text-xs font-medium text-slate-600 w-12">Slice:</span>
                <input
                  type="range"
                  min="10"
                  max={currentSlice?.total_slices ? currentSlice.total_slices - 10 : 140}
                  value={sliceIndex}
                  onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                  onMouseUp={handleRunSegmentation}
                  onTouchEnd={handleRunSegmentation}
                  className="flex-1 accent-[#0e7c5b] cursor-pointer"
                />
                <span className="text-xs font-mono bg-slate-100 px-2 py-1 rounded text-slate-700 w-10 text-center">
                  {sliceIndex}
                </span>
              </div>
            </div>
          </div>

          {/* Uncertainty Metrics Card */}
          <div className="bg-white rounded-2xl border border-[#e2ece7] p-5 shadow-xs space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-amber-600" />
                Spatial Uncertainty & Calibration (MC-Dropout)
              </h3>
              <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-amber-50 text-amber-700 border border-amber-200">
                {predictionData?.uncertainty?.confidence_level || 'Evaluating...'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500">Predictive Entropy</div>
                <div className="text-base font-bold text-slate-900 mt-0.5">
                  {predictionData?.uncertainty?.mean_predictive_entropy || 0.0}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500">Peak Boundary Entropy</div>
                <div className="text-base font-bold text-slate-900 mt-0.5">
                  {predictionData?.uncertainty?.max_predictive_entropy || 0.0}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500">Epistemic Variance</div>
                <div className="text-base font-bold text-slate-900 mt-0.5">
                  {predictionData?.uncertainty?.epistemic_variance || 0.0}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500">Expected Calib. Error</div>
                <div className="text-base font-bold text-slate-900 mt-0.5">
                  {predictionData?.uncertainty?.calibration_ece || 0.0}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Physical Quantification & 3-Model Comparison (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Physical Volume Quantification in cm³ */}
          <div className="bg-white rounded-2xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-[#e2ece7] pb-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#0e7c5b]" />
                Physical Volume Quantification (cm³)
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">1 mm³ = 0.001 cm³</span>
            </div>

            <div className="space-y-3">
              {/* WT */}
              <div className="p-3.5 bg-emerald-50/60 rounded-xl border border-emerald-200 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-emerald-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    Whole Tumor (WT)
                  </div>
                  <div className="text-[11px] text-emerald-700 mt-0.5">Edema + Non-enhancing + Enhancing</div>
                </div>
                <div className="text-lg font-bold text-emerald-950 font-mono">
                  {predictionData?.volumes?.whole_tumor_cm3 || 0.0} <span className="text-xs font-normal">cm³</span>
                </div>
              </div>

              {/* TC */}
              <div className="p-3.5 bg-amber-50/60 rounded-xl border border-amber-200 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-amber-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    Tumor Core (TC)
                  </div>
                  <div className="text-[11px] text-amber-700 mt-0.5">Necrotic Core + Active Enhancing</div>
                </div>
                <div className="text-lg font-bold text-amber-950 font-mono">
                  {predictionData?.volumes?.tumor_core_cm3 || 0.0} <span className="text-xs font-normal">cm³</span>
                </div>
              </div>

              {/* ET */}
              <div className="p-3.5 bg-rose-50/60 rounded-xl border border-rose-200 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-rose-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-500" />
                    Enhancing Tumor (ET)
                  </div>
                  <div className="text-[11px] text-rose-700 mt-0.5">Hypervascularized active rim</div>
                </div>
                <div className="text-lg font-bold text-rose-950 font-mono">
                  {predictionData?.volumes?.enhancing_tumor_cm3 || 0.0} <span className="text-xs font-normal">cm³</span>
                </div>
              </div>

              {/* ED */}
              <div className="p-3.5 bg-blue-50/60 rounded-xl border border-blue-200 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-blue-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    Peritumoral Edema (ED)
                  </div>
                  <div className="text-[11px] text-blue-700 mt-0.5">Fluid infiltration (WT - TC)</div>
                </div>
                <div className="text-lg font-bold text-blue-950 font-mono">
                  {predictionData?.volumes?.edema_cm3 || 0.0} <span className="text-xs font-normal">cm³</span>
                </div>
              </div>
            </div>

            {/* Clinical Interpretation Note */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-700 leading-relaxed">
              <span className="font-semibold text-slate-900">Radiological Impression: </span>
              {predictionData?.clinical_interpretation || "Awaiting segmentation calculation..."}
            </div>
          </div>

          {/* Three-Model Benchmark Comparison Table */}
          <div className="bg-white rounded-2xl border border-[#e2ece7] p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-[#e2ece7] pb-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-[#0e7c5b]" />
                Rigorous 3-Model Benchmark Comparison
              </h3>
              <span className="text-[11px] px-2 py-0.5 bg-slate-100 rounded text-slate-600 font-medium">Held-out Patients</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="pb-2 font-semibold">Model</th>
                    <th className="pb-2 font-semibold">Calibrated WT Dice (95% CI)</th>
                    <th className="pb-2 font-semibold">Calib TC Dice</th>
                    <th className="pb-2 font-semibold">Calib ET Dice</th>
                    <th className="pb-2 font-semibold">HD95 (mm)</th>
                    <th className="pb-2 font-semibold">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {evalReport?.models && Object.entries(evalReport.models).map(([key, m]) => {
                    const cal = m.calibrated_metrics;
                    return (
                      <tr 
                        key={key} 
                        className={`hover:bg-slate-50 transition-colors ${selectedModel === key ? 'bg-[#ecfdf5]/50 font-semibold' : ''}`}
                      >
                        <td className="py-2.5 text-slate-900">
                          <div className="flex items-center gap-1.5">
                            {selectedModel === key && <span className="w-1.5 h-1.5 rounded-full bg-[#0e7c5b]" />}
                            <div>
                              <div>{key === 'attunet3d' ? '3D Attention U-Net' : (key === 'unet3d' ? '3D U-Net' : 'DynUNet (nnU-Net)')}</div>
                              <div className="text-[10px] text-slate-400 font-normal">Th: WT {cal?.operating_thresholds?.WT || 0.5} / TC {cal?.operating_thresholds?.TC || 0.5}</div>
                            </div>
                          </div>
                        </td>
                        <td className="py-2.5 text-slate-700">
                          {cal?.WT_dice?.mean ?? m.subregions?.WT?.dice?.mean} <span className="text-[10px] text-slate-400">[{cal?.WT_dice?.ci95?.[0] ?? m.subregions?.WT?.dice?.ci95?.[0]}, {cal?.WT_dice?.ci95?.[1] ?? m.subregions?.WT?.dice?.ci95?.[1]}]</span>
                        </td>
                        <td className="py-2.5 text-slate-700">
                          {cal?.TC_dice?.mean ?? m.subregions?.TC?.dice?.mean}
                        </td>
                        <td className="py-2.5 text-slate-700">
                          {cal?.ET_dice?.mean ?? m.subregions?.ET?.dice?.mean}
                        </td>
                        <td className="py-2.5 text-slate-700">
                          {cal?.WT_hd95 ?? m.subregions?.WT?.hd95?.mean} mm
                        </td>
                        <td className="py-2.5 text-slate-700">
                          {m.mean_latency_ms} ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="p-3 bg-emerald-50/50 rounded-xl border border-emerald-100 text-[11px] text-emerald-800 space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                Patient-Level Leakage Prevention & Operating Thresholds Verified
              </div>
              <p className="text-emerald-700">
                Operating thresholds tuned strictly on 4 validation patients (<code className="bg-emerald-100/60 px-1 rounded">00020_11, 00021_11, 00026_11, 00029_11</code>) and tested on 4 held-out test patients (<code className="bg-emerald-100/60 px-1 rounded">00030_11, 00031_11, 00033_11, 00035_11</code>) with zero scan leakage.
              </p>
            </div>

            {/* Per-Patient Test Breakdown for Selected Model */}
            {evalReport?.models?.[selectedModel]?.per_patient_calibrated_breakdown && (
              <div className="pt-2 border-t border-slate-100 space-y-2">
                <div className="text-[11px] font-bold text-slate-700 uppercase tracking-wider flex items-center justify-between">
                  <span>Held-Out Test Patients (n=4)</span>
                  <span className="font-normal text-slate-400 capitalize">{selectedModel}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  {evalReport.models[selectedModel].per_patient_calibrated_breakdown.map((pt) => (
                    <div key={pt.patient_id} className="p-2.5 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
                      <div className="font-mono font-semibold text-slate-800 text-[10px]">{pt.patient_id}</div>
                      <div className="flex justify-between text-slate-600">
                        <span>WT Dice:</span>
                        <span className="font-semibold text-slate-900">{pt.calibrated_metrics?.WT?.dice != null ? pt.calibrated_metrics.WT.dice.toFixed(3) : '0.000'}</span>
                      </div>
                      <div className="flex justify-between text-slate-600">
                        <span>TC Dice:</span>
                        <span className="font-semibold text-slate-900">{pt.calibrated_metrics?.TC?.dice != null ? pt.calibrated_metrics.TC.dice.toFixed(3) : '0.000'}</span>
                      </div>
                      <div className="flex justify-between text-slate-600">
                        <span>ET Dice:</span>
                        <span className="font-semibold text-slate-900">{pt.calibrated_metrics?.ET?.dice != null ? pt.calibrated_metrics.ET.dice.toFixed(3) : '0.000'}</span>
                      </div>
                      <div className="flex justify-between text-[10px] text-slate-400 pt-0.5 border-t border-slate-200">
                        <span>Vol (Pred/True):</span>
                        <span className="font-mono">{pt.calibrated_metrics?.WT?.pred_vol_cm3 ?? 0} / {pt.true_volumes_cm3?.WT ?? 0} cm³</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      {/* Upload MRI Scan Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-100 space-y-5 animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-[#ecfdf5] flex items-center justify-center text-[#0e7c5b]">
                  <Upload className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">Upload MRI Scan</h3>
                  <p className="text-xs text-slate-500">2D MRI image (PNG/JPG) or 3D NIfTI volume (.nii/.nii.gz)</p>
                </div>
              </div>
              <button
                onClick={() => {
                  if (!uploading) {
                    setShowUploadModal(false);
                    setUploadFile(null);
                    setUploadPreview(null);
                    setUploadError(null);
                  }
                }}
                disabled={uploading}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dropzone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                isDragging 
                  ? 'border-[#0e7c5b] bg-[#ecfdf5]/40' 
                  : uploadFile 
                    ? 'border-emerald-300 bg-emerald-50/20' 
                    : 'border-slate-300 hover:border-[#0e7c5b] hover:bg-slate-50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif,.nii,.nii.gz"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                }}
                className="hidden"
              />

              {uploadPreview ? (
                <div className="space-y-3">
                  <img
                    src={uploadPreview}
                    alt="Scan Preview"
                    className="max-h-48 mx-auto rounded-lg shadow-sm border border-slate-200 object-contain"
                  />
                  <div className="text-xs font-medium text-slate-700">
                    {uploadFile?.name} ({(uploadFile?.size / 1024).toFixed(1)} KB)
                  </div>
                  <span className="text-[11px] text-[#0e7c5b] hover:underline">Click or drag to replace image</span>
                </div>
              ) : uploadFile ? (
                <div className="space-y-2 py-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 mx-auto flex items-center justify-center">
                    <FileCode className="w-6 h-6" />
                  </div>
                  <div className="font-semibold text-slate-800 text-sm">{uploadFile.name}</div>
                  <div className="text-xs text-slate-500">{(uploadFile.size / (1024 * 1024)).toFixed(2)} MB</div>
                  <span className="text-[11px] text-[#0e7c5b] hover:underline">Click to choose another scan</span>
                </div>
              ) : (
                <div className="space-y-3 py-4">
                  <div className="w-12 h-12 rounded-xl bg-slate-100 text-slate-500 mx-auto flex items-center justify-center">
                    <ImageIcon className="w-6 h-6" />
                  </div>
                  <div>
                    <span className="font-semibold text-[#0e7c5b] text-sm">Click to upload</span>
                    <span className="text-slate-600 text-sm"> or drag and drop</span>
                  </div>
                  <p className="text-xs text-slate-400">
                    PNG, JPG, JPEG, BMP, TIFF or NIfTI (.nii, .nii.gz) up to 50 MB
                  </p>
                </div>
              )}
            </div>

            {uploadError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Model Selection in Modal */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-700">Select Segmentation Model:</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'attunet3d', name: '3D Attention U-Net', badge: 'Recommended' },
                  { id: 'unet3d', name: '3D U-Net', badge: 'Fast' },
                  { id: 'nnunet', name: '3D nnU-Net v2', badge: 'DynUNet' }
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setUploadModel(m.id)}
                    className={`p-2.5 rounded-xl border text-left transition-all ${
                      uploadModel === m.id
                        ? 'border-[#0e7c5b] bg-[#ecfdf5] text-[#0f4c3a]'
                        : 'border-slate-200 hover:border-slate-300 text-slate-700'
                    }`}
                  >
                    <div className="text-xs font-semibold">{m.name}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">{m.badge}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 pt-2 border-t border-slate-100">
              <button
                type="button"
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadFile(null);
                  setUploadPreview(null);
                  setUploadError(null);
                }}
                disabled={uploading}
                className="px-4 py-2 border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-xl text-xs font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUploadAndSegment}
                disabled={!uploadFile || uploading}
                className="flex items-center gap-2 px-5 py-2 bg-[#0e7c5b] hover:bg-[#0c6b4e] text-white rounded-xl text-xs font-medium shadow-xs transition-all disabled:opacity-50"
              >
                {uploading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Reconstructing &amp; Segmenting...</span>
                  </>
                ) : (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>Segment 3D Volume</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TumorSegmentation;
