import React from 'react';
import { 
  Activity, CheckCircle2, AlertTriangle, AlertCircle, 
  HelpCircle, Sparkles, TrendingUp, Compass, Shield
} from 'lucide-react';

export const LeadCard = ({ leadName, alignment, isSelected, onClick }) => {
  if (!alignment) {
    return (
      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-400">
        <span className="font-semibold text-slate-600">Lead {leadName}</span>: Not mapped
      </div>
    );
  }

  const {
    geometry_status,
    reconstruction_status,
    calibration_status,
    attribution_status,
    alignment_status,
    attribution_importance,
    attribution_rank,
    reconstruction_confidence,
    uncertainty_reasons = [],
    units_status,
    is_rhythm_strip,
  } = alignment;

  // Status visual configurations
  const statusStyles = {
    CONFIDENT: {
      border: 'border-emerald-300 bg-emerald-50/40 text-emerald-900',
      badge: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      icon: CheckCircle2,
    },
    UNCERTAIN: {
      border: 'border-amber-300 bg-amber-50/40 text-amber-900 border-dashed',
      badge: 'bg-amber-100 text-amber-800 border-amber-200',
      icon: AlertTriangle,
    },
    UNAVAILABLE: {
      border: 'border-slate-300 bg-slate-50/60 text-slate-700 border-dotted',
      badge: 'bg-slate-200 text-slate-700 border-slate-300',
      icon: AlertCircle,
    },
  };

  const currentStyle = statusStyles[alignment_status] || statusStyles.UNAVAILABLE;
  const StatusIcon = currentStyle.icon;

  return (
    <div
      onClick={onClick}
      className={`p-3.5 rounded-xl border transition-all cursor-pointer relative ${
        currentStyle.border
      } ${
        isSelected
          ? 'ring-2 ring-[#0e7c5b] shadow-md bg-white'
          : 'hover:shadow-xs'
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5">
          <span className="font-bold text-sm text-slate-900">
            {is_rhythm_strip ? 'Lead II (Continuous Rhythm)' : `Lead ${leadName}`}
          </span>
          {attribution_rank !== null && attribution_rank !== undefined && attribution_status === 'AVAILABLE' && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-800 text-white">
              Rank #{attribution_rank}
            </span>
          )}
        </div>
        <span
          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border flex items-center gap-1 ${currentStyle.badge}`}
        >
          <StatusIcon className="w-3 h-3" />
          {alignment_status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600 mb-2">
        <div>
          <span className="text-slate-400 block text-[10px]">Reconstruction</span>
          <span className="font-medium text-slate-800">
            {reconstruction_status} ({reconstruction_confidence !== undefined ? Math.round(reconstruction_confidence * 100) : 0}%)
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px]">Attribution Score</span>
          <span className="font-mono font-medium text-slate-800">
            {attribution_status === 'AVAILABLE' && attribution_importance !== undefined
              ? Number(attribution_importance).toFixed(4)
              : 'Unavailable'}
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px]">Geometry Box</span>
          <span className="font-medium text-slate-700 truncate block">
            {geometry_status === 'VERIFIED' ? 'Verified' : geometry_status === 'ESTIMATED_STANDARD_LAYOUT' ? 'Standard 3x4' : 'Unknown'}
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px]">Physical Units</span>
          <span className="font-medium text-slate-700 truncate block">
            {calibration_status === 'VERIFIED' ? 'Calibrated (mV)' : 'Unverified (px)'}
          </span>
        </div>
      </div>

      {uncertainty_reasons.length > 0 && uncertainty_reasons[0] !== 'NONE' && (
        <div className="pt-2 border-t border-slate-200/60 mt-1">
          <span className="text-[10px] font-medium text-amber-800 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3 text-amber-600 shrink-0" />
            {uncertainty_reasons.join(', ')}
          </span>
        </div>
      )}
    </div>
  );
};

export const LeadOverlayMap = ({
  imageUrl,
  leadAlignments,
  rhythmStripAlignment,
  selectedLead,
  onSelectLead,
  imageWidth,
  imageHeight,
}) => {
  const [naturalDims, setNaturalDims] = React.useState({ w: imageWidth || 1665, h: imageHeight || 1065 });
  const [imgLoaded, setImgLoaded] = React.useState(false);
  const [imgError, setImgError] = React.useState(false);

  if (!imageUrl || imgError) {
    return (
      <div className="h-96 rounded-xl border border-dashed border-slate-300 bg-slate-50 flex flex-col items-center justify-center p-6 text-center">
        <Activity className="w-10 h-10 text-slate-300 mb-2" />
        <p className="text-sm font-medium text-slate-600">
          {imgError ? 'Unable to load ECG scan image' : 'No ECG Scan Image Available'}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          {imgError ? 'The image file could not be retrieved from the server storage.' : 'Upload a 12-lead ECG scan or select a recorded study.'}
        </p>
      </div>
    );
  }

  const baseW = naturalDims.w || 1665;
  const baseH = naturalDims.h || 1065;

  const leads = leadAlignments ? Object.values(leadAlignments) : [];
  const allRegions = [...leads];
  if (rhythmStripAlignment) {
    allRegions.push(rhythmStripAlignment);
  }

  const getOverlayStyle = (alignment, isSelected) => {
    const stat = alignment.alignment_status;
    let border = 'border-slate-400 border-dotted bg-slate-400/5';
    let textColor = 'text-slate-700 bg-slate-100/90';

    if (stat === 'CONFIDENT') {
      border = isSelected
        ? 'border-[#0e7c5b] bg-[#0e7c5b]/15 border-2 shadow-sm'
        : 'border-emerald-600 border-solid bg-emerald-600/10 hover:bg-emerald-600/20';
      textColor = 'text-emerald-950 bg-emerald-100/90 border border-emerald-300';
    } else if (stat === 'UNCERTAIN') {
      border = isSelected
        ? 'border-amber-600 bg-amber-500/20 border-2 shadow-sm'
        : 'border-amber-500 border-dashed bg-amber-500/10 hover:bg-amber-500/20';
      textColor = 'text-amber-950 bg-amber-100/90 border border-amber-300';
    } else if (stat === 'UNAVAILABLE') {
      border = 'border-slate-400 border-dotted bg-slate-500/5';
      textColor = 'text-slate-700 bg-slate-200/90 border border-slate-300';
    }

    return { border, textColor };
  };

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs py-2 px-3 bg-slate-50 border border-[#e2ece7] rounded-lg">
        <div className="flex items-center gap-4">
          <span className="font-semibold text-slate-700 text-[11px] uppercase tracking-wide">
            Lead Region Alignment:
          </span>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 rounded-xs border border-emerald-600 bg-emerald-600/20 inline-block" />
            <span className="text-slate-700 font-medium">Confident</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 rounded-xs border border-amber-500 border-dashed bg-amber-500/20 inline-block" />
            <span className="text-slate-700 font-medium">Uncertain</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 rounded-xs border border-slate-400 border-dotted bg-slate-400/20 inline-block" />
            <span className="text-slate-500 font-medium">Unavailable</span>
          </div>
        </div>
        <span className="text-[11px] text-slate-500 italic">
          Click any lead outline or card to inspect attribution metadata
        </span>
      </div>

      {/* Interactive Image Container */}
      <div className="relative rounded-xl overflow-hidden border border-slate-300 bg-white shadow-sm max-w-full">
        <img
          src={imageUrl}
          alt="12-Lead ECG scan"
          className="w-full h-auto block select-none pointer-events-none"
          onLoad={(e) => {
            if (e.target.naturalWidth && e.target.naturalHeight) {
              setNaturalDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
              setImgLoaded(true);
            }
          }}
          onError={() => {
            setImgError(true);
          }}
        />

        {/* Dynamic Lead Outlines */}
        {allRegions.map((aln) => {
          if (!aln.image_region) return null;
          const box = aln.image_region;
          const isSelected = selectedLead === aln.lead_name;
          const { border, textColor } = getOverlayStyle(aln, isSelected);

          const leftPct = (box.x_min / baseW) * 100;
          const topPct = (box.y_min / baseH) * 100;
          const widthPct = (box.width / baseW) * 100;
          const heightPct = (box.height / baseH) * 100;

          return (
            <div
              key={aln.lead_name}
              onClick={() => onSelectLead && onSelectLead(aln.lead_name)}
              style={{
                left: `${leftPct}%`,
                top: `${topPct}%`,
                width: `${widthPct}%`,
                height: `${heightPct}%`,
              }}
              className={`absolute transition-all cursor-pointer flex flex-col justify-between p-1.5 ${border}`}
            >
              <div className="flex items-center justify-between gap-1 pointer-events-none">
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded shadow-xs leading-none ${textColor}`}
                >
                  {aln.is_rhythm_strip ? 'II Rhythm' : aln.lead_name}
                </span>

                {aln.attribution_rank && aln.attribution_status === 'AVAILABLE' && (
                  <span className="text-[9px] font-mono font-bold px-1 rounded bg-slate-900/80 text-white leading-tight">
                    #{aln.attribution_rank}
                  </span>
                )}
              </div>

              {aln.uncertainty_reasons.length > 0 && aln.uncertainty_reasons[0] !== 'NONE' && (
                <div className="pointer-events-none">
                  <span className="text-[9px] font-semibold px-1 rounded bg-amber-100/90 text-amber-900 border border-amber-300 leading-tight inline-block truncate max-w-full">
                    {aln.uncertainty_reasons[0]}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
