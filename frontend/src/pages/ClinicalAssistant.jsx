import React, { useEffect, useState } from 'react';
import { Bot, FileText, Send, ShieldCheck, Sparkles, Copy, Check, Eye, Code, RotateCw, AlertTriangle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import api, { patientService } from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';

export default function ClinicalAssistant() {
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState('');
  const [question, setQuestion] = useState('Summarize the relevant guideline evidence for diabetes screening.');
  const [result, setResult] = useState(null);
  const [draft, setDraft] = useState(null);
  const [loading, setLoading] = useState(false);
  const [draftLoading, setDraftLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState('formatted'); // 'formatted' | 'raw'

  useEffect(() => {
    patientService.getPatients()
      .then(({ data }) => setPatients(data))
      .catch(() => setError('Unable to load patients.'));
  }, []);

  const ask = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    setDraft(null);
    try {
      const res = await api.post('/assistant/ask', {
        question,
        patient_id: patientId ? Number(patientId) : null,
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'The assistant is unavailable.');
    } finally {
      setLoading(false);
    }
  };

  const createDraft = async () => {
    if (!patientId) {
      return setError('Select a patient before creating a draft for review.');
    }
    setDraftLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.get(`/assistant/patients/${patientId}/draft`);
      setDraft(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not create the clinical draft.');
    } finally {
      setDraftLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!draft?.draft) return;
    try {
      await navigator.clipboard.writeText(draft.draft);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy draft to clipboard', err);
    }
  };

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Bot className="w-6 h-6 text-[#0e7c5b]" />
          AI Clinical Assistant
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Controlled guideline evidence, patient-specific records, and synthesized clinical case summaries.
        </p>
      </div>

      <ClinicalDisclaimer />

      <section className="med-card p-5">
        <form onSubmit={ask} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Patient Context (Required for clinical summaries)
            </label>
            <select
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="input-med text-sm"
            >
              <option value="">Guideline evidence only (Select patient for summaries)</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.first_name} {p.last_name} (#{p.id})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Clinical Query / Question (Guidelines, Patient Records, or Reports)
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              required
              minLength="3"
              rows="3"
              className="input-med text-sm"
              placeholder="Ask any clinical question (e.g. guideline criteria, patient cholesterol, abnormal lab values, report impressions, medications)..."
            />
          </div>

          <div className="flex flex-wrap gap-2 pt-1">
            <button
              type="submit"
              disabled={loading || draftLoading}
              className="btn-med-primary text-xs flex items-center gap-1.5"
            >
              <Send className="w-3.5 h-3.5" />
              {loading ? 'Consulting guidelines & records…' : 'Ask from evidence'}
            </button>

            <button
              type="button"
              onClick={createDraft}
              disabled={loading || draftLoading}
              className="btn-med-secondary text-xs flex items-center gap-1.5"
              title="Generate a structured clinical case summary for this patient"
            >
              {draftLoading ? (
                <>
                  <RotateCw className="w-3.5 h-3.5 animate-spin text-[#0e7c5b]" />
                  <span>Synthesizing clinical summary…</span>
                </>
              ) : (
                <>
                  <FileText className="w-3.5 h-3.5" />
                  <span>{draft ? 'Regenerate draft for review' : 'Create draft for review'}</span>
                </>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-500" />
            <span>{error}</span>
          </div>
        )}
      </section>

      {/* RAG Evidence Grounded Result */}
      {result && (
        <section className="med-card p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                Clinical Answer
                {result.provider && (
                  <span className="text-[10px] font-medium bg-emerald-100/70 text-emerald-800 px-2 py-0.5 rounded-full">
                    {result.provider === 'local_clinical_engine' ? 'Guideline & Record Synthesis' : result.provider}
                  </span>
                )}
              </h2>
              <p className="text-[11px] text-slate-500">
                Direct answer to query: <span className="font-medium text-slate-700">"{result.question}"</span>
              </p>
            </div>

            {result.answer && (
              <button
                type="button"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(result.answer);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  } catch (e) {
                    console.error(e);
                  }
                }}
                className="btn-med-secondary text-xs flex items-center gap-1.5 py-1 px-2.5 self-start sm:self-auto"
                title="Copy answer to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span className="text-emerald-700 font-semibold">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy Answer
                  </>
                )}
              </button>
            )}
          </div>

          {result.answer ? (
            <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 text-slate-700">
              <ReactMarkdown
                components={{
                  h3: ({ node, ...props }) => (
                    <h3 className="text-xs font-bold text-[#0e7c5b] uppercase tracking-wide mt-3 mb-2 flex items-center gap-1.5" {...props} />
                  ),
                  h4: ({ node, ...props }) => (
                    <h4 className="text-xs font-bold text-slate-800 tracking-wide mt-3 mb-1.5" {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p className="text-xs leading-relaxed text-slate-700 mb-2.5" {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul className="list-disc pl-4 text-xs text-slate-700 space-y-1.5 mb-2.5" {...props} />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol className="list-decimal pl-4 text-xs text-slate-700 space-y-1.5 mb-2.5" {...props} />
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
                  hr: () => (
                    <hr className="my-3 border-slate-200" />
                  ),
                }}
              >
                {result.answer}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800">
              <p className="font-semibold">Notice:</p>
              <p className="mt-1">{result.message || 'No sufficient clinical guideline evidence found for this query.'}</p>
            </div>
          )}

          {result.patient_context && (
            <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-3.5 text-xs text-slate-700">
              <div className="flex items-center justify-between pb-2 border-b border-slate-200/60 mb-2">
                <span className="font-semibold text-slate-800 flex items-center gap-1.5 text-xs">
                  <span className="w-2 h-2 rounded-full bg-[#0e7c5b]" />
                  Supporting Patient Record Context: {result.patient_context.patient_name}
                </span>
                <span className="text-[10px] bg-white text-slate-600 border border-slate-200 px-2 py-0.5 rounded font-mono">
                  MRN #{result.patient_context.patient_id}
                </span>
              </div>
              <p className="mt-1 text-slate-600">
                <span className="font-semibold text-slate-700">Documented Conditions: </span>
                {result.patient_context.history.length ? result.patient_context.history.join(', ') : 'No chronic conditions recorded.'}
              </p>
              <div className="mt-2 space-y-1">
                <span className="font-semibold text-slate-700 block mb-1">Key Pertinent Findings in Record:</span>
                {result.patient_context.report_matches.length ? (
                  result.patient_context.report_matches.map((match, i) => (
                    <p key={i} className="text-slate-600 pl-2 border-l-2 border-emerald-400 text-[11px]">
                      • {match}
                    </p>
                  ))
                ) : (
                  <span className="text-slate-400">No active laboratory or report abnormalities registered.</span>
                )}
              </div>
            </div>
          )}

          <Evidence items={result.evidence} />
        </section>
      )}

      {/* Synthesized Clinical Summary Draft */}
      {draft && (
        <section className="med-card p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center shrink-0">
                <Sparkles className="w-4 h-4 text-[#0e7c5b]" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                  Synthesized Clinical Case Summary
                  {draft.summary_mode && (
                    <span className="text-[10px] font-medium bg-emerald-100/70 text-emerald-800 px-2 py-0.5 rounded-full">
                      {draft.summary_mode}
                    </span>
                  )}
                </h2>
                <p className="text-[11px] text-slate-500">
                  Synthesized case summary from medical history, lab biomarkers, and diagnostic reports
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto">
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

              <button
                type="button"
                onClick={handleCopy}
                className="btn-med-secondary text-xs flex items-center gap-1.5 py-1 px-2.5"
                title="Copy entire draft to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span className="text-emerald-700 font-semibold">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy
                  </>
                )}
              </button>
            </div>
          </div>

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
                  {draft.draft}
                </ReactMarkdown>
              </div>
            ) : (
              <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-700 rounded-xl bg-slate-50 border border-slate-200 p-4 overflow-x-auto select-all">
                {draft.draft}
              </pre>
            )}

            <div className="mt-3 p-2.5 rounded-lg bg-amber-50/80 border border-amber-200/80 flex items-start gap-2 text-amber-800">
              <ShieldCheck className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
              <div className="text-[11px] leading-relaxed">
                <span className="font-semibold">Clinician Verification Notice: </span>
                {draft.disclaimer}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function Evidence({ items = [] }) {
  if (!items.length) return null;
  return (
    <div className="border-t border-slate-100 pt-4">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Retrieved sources</p>
      {items.map((item, i) => (
        <article key={i} className="mt-3 rounded-lg border border-slate-100 p-3 text-xs">
          <p className="font-semibold text-slate-800">
            [{i + 1}] {item.source} <span className="font-normal text-slate-400">({item.score})</span>
          </p>
          <p className="mt-1 text-[#0e7c5b] font-medium">{item.section}</p>
          <p className="mt-2 leading-relaxed text-slate-600">{item.text}</p>
        </article>
      ))}
    </div>
  );
}
