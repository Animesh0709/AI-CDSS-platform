import React, { useState } from 'react';
import { Bot, FileText, Copy, Check, RotateCw, Sparkles, ShieldCheck, Eye, Code, AlertTriangle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import api from '../services/api';

export default function PatientClinicalDraft({ patientId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState('formatted'); // 'formatted' | 'raw'

  const create = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get(`/assistant/patients/${patientId}/draft`);
      setData(response.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Unable to create the clinical draft.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!data?.draft) return;
    try {
      await navigator.clipboard.writeText(data.draft);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy draft to clipboard', err);
    }
  };

  return (
    <section className="med-card p-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center shrink-0">
            <Sparkles className="w-4 h-4 text-[#0e7c5b]" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
              AI Clinical Summary
              {data?.summary_mode && (
                <span className="text-[10px] font-medium bg-emerald-100/70 text-emerald-800 px-2 py-0.5 rounded-full">
                  {data.summary_mode}
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-500">
              Synthesized case summary from medical history, lab biomarkers, and diagnostic reports
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          {data && (
            <div className="flex items-center bg-slate-100 rounded-lg p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setViewMode('formatted')}
                className={`px-2 py-1 rounded-md flex items-center gap-1 font-medium transition-all ${
                  viewMode === 'formatted'
                    ? 'bg-white text-slate-800 shadow-xs'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
                title="Formatted Clinical View"
              >
                <Eye className="w-3 h-3" />
                Formatted
              </button>
              <button
                type="button"
                onClick={() => setViewMode('raw')}
                className={`px-2 py-1 rounded-md flex items-center gap-1 font-medium transition-all ${
                  viewMode === 'raw'
                    ? 'bg-white text-slate-800 shadow-xs'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
                title="Plain Text View"
              >
                <Code className="w-3 h-3" />
                Raw Text
              </button>
            </div>
          )}
          <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded font-medium">
            Live
          </span>
        </div>
      </div>

      {/* Action Bar */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
        <button
          onClick={create}
          disabled={loading}
          className="btn-med-primary text-xs flex items-center gap-1.5"
        >
          {loading ? (
            <>
              <RotateCw className="w-3.5 h-3.5 animate-spin" />
              Synthesizing Clinical Summary…
            </>
          ) : (
            <>
              <FileText className="w-3.5 h-3.5" />
              {data ? 'Regenerate draft for review' : 'Create draft for review'}
            </>
          )}
        </button>

        {data && (
          <button
            type="button"
            onClick={handleCopy}
            className="btn-med-secondary text-xs flex items-center gap-1.5"
            title="Copy entire draft to clipboard"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700 font-semibold">Copied to Clipboard!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                Copy Summary
              </>
            )}
          </button>
        )}
      </div>

      {error && (
        <div className="mt-3 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-500" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !data && (
        <div className="mt-4 p-5 rounded-lg bg-slate-50 border border-slate-100 space-y-3 animate-pulse">
          <div className="h-4 bg-slate-200 rounded w-1/3"></div>
          <div className="h-3 bg-slate-200 rounded w-full"></div>
          <div className="h-3 bg-slate-200 rounded w-5/6"></div>
          <div className="h-3 bg-slate-200 rounded w-4/6"></div>
          <p className="text-[11px] text-slate-500 italic mt-2">
            Synthesizing patient history, laboratory metrics, and extracted reports into a structured case review…
          </p>
        </div>
      )}

      {/* Render Summary Data */}
      {data && (
        <div className="mt-4">
          {viewMode === 'formatted' ? (
            <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 text-slate-700">
              <ReactMarkdown
                components={{
                  h1: ({ node, ...props }) => (
                    <h1 className="text-sm font-bold text-slate-900 border-b border-slate-200 pb-2 mb-3 tracking-wide" {...props} />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 mt-4 mb-2" {...props} />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 className="text-xs font-bold text-[#0e7c5b] mt-3.5 mb-1.5 flex items-center gap-1.5" {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p className="text-xs leading-relaxed text-slate-700 mb-2.5" {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc pl-4 text-xs text-slate-700 space-y-1 mb-2.5" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal pl-4 text-xs text-slate-700 space-y-1 mb-2.5" {...props} />
                  ),
                  li: ({ node, ...props }) => (
                    <li className="leading-relaxed" {...props} />
                  ),
                  strong: ({ node, ...props }) => (
                    <strong className="font-semibold text-slate-900" {...props} />
                  ),
                  code: ({ node, ...props }) => (
                    <code className="bg-emerald-50 text-emerald-800 border border-emerald-200 text-[11px] px-1.5 py-0.5 rounded font-mono font-medium" {...props} />
                  ),
                  blockquote: ({ node, ...props }) => (
                    <blockquote className="border-l-3 border-[#0e7c5b] pl-3 py-1 my-2 text-xs text-slate-600 bg-emerald-50/50 rounded-r-lg" {...props} />
                  ),
                  table: ({ node, ...props }) => (
                    <div className="overflow-x-auto my-3">
                      <table className="min-w-full text-xs text-slate-700 border border-slate-200 rounded-lg" {...props} />
                    </div>
                  ),
                  th: ({ node, ...props }) => (
                    <th className="border border-slate-200 bg-slate-100/80 p-2 text-left font-semibold text-slate-800" {...props} />
                  ),
                  td: ({ node, ...props }) => (
                    <td className="border border-slate-200 p-2" {...props} />
                  ),
                  hr: () => (
                    <hr className="my-3 border-slate-200" />
                  ),
                }}
              >
                {data.draft}
              </ReactMarkdown>
            </div>
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-700 rounded-xl bg-slate-50 border border-slate-200 p-4 overflow-x-auto select-all">
              {data.draft}
            </pre>
          )}

          {/* Clinical Disclaimer Callout */}
          <div className="mt-3 p-2.5 rounded-lg bg-amber-50/80 border border-amber-200/80 flex items-start gap-2 text-amber-800">
            <ShieldCheck className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
            <div className="text-[11px] leading-relaxed">
              <span className="font-semibold">Clinician Verification Notice: </span>
              {data.disclaimer}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
