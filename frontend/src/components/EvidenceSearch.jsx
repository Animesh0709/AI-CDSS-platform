import React, { useState } from 'react';
import { BookOpen, Bot, ExternalLink, Search } from 'lucide-react';
import api from '../services/api';

export default function EvidenceSearch({ patientId = null }) {
  const [query, setQuery] = useState('What does the guideline say about diabetes screening?');
  const [data, setData] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const request = async (path) => {
    setError(''); setLoading(true);
    try {
      const response = (await api.post(path, path === '/assistant/ask' ? { question: query, patient_id: patientId } : { query })).data;
      if (path === '/rag/answer' || path === '/assistant/ask') { setAnswer(response); setData({ results: response.evidence }); }
      else { setAnswer(null); setData(response); }
    } catch (err) { setError(err.response?.data?.detail || 'Evidence retrieval is unavailable.'); }
    finally { setLoading(false); }
  };

  return <section className="med-card p-5">
    <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-3">
      <div className="flex items-center gap-2"><BookOpen className="w-4 h-4 text-[#0e7c5b]" /><h2 className="text-sm font-bold text-slate-800">Medical Evidence Assistant</h2></div>
      <span className="text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-0.5 rounded">Live</span>
    </div>
    <p className="mt-3 text-xs text-slate-600">Retrieves curated guideline excerpts first. {patientId ? 'Matching stored-report passages are also used only to help navigate this patient’s record. ' : ''}Neither diagnoses nor prescribes.</p>
    <form onSubmit={(event) => { event.preventDefault(); request('/rag/search'); }} className="mt-3 flex gap-2">
      <input value={query} onChange={(e) => setQuery(e.target.value)} className="input-med text-xs" minLength="3" required />
      <button disabled={loading} className="btn-med-secondary text-xs px-3 flex items-center gap-1"><Search className="w-3.5 h-3.5" />Retrieve</button>
      <button type="button" onClick={() => request(patientId ? '/assistant/ask' : '/rag/answer')} disabled={loading || query.trim().length < 3} className="btn-med-primary text-xs px-3 flex items-center gap-1"><Bot className="w-3.5 h-3.5" />Answer</button>
    </form>
    {error && <p className="mt-2 text-xs text-rose-700">{error}</p>}
    {answer && <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50/50 p-3 text-xs">
      <p className="font-semibold text-emerald-900">{answer.status === 'evidence_summary' ? 'Evidence summary' : 'Generated explanation'}</p>
      <p className="mt-2 leading-relaxed whitespace-pre-wrap text-slate-700">{answer.answer || answer.message}</p>
      {answer.patient_context && <div className="mt-3 border-t border-emerald-100 pt-2 text-[11px] text-slate-600"><b>Patient record navigation: {answer.patient_context.patient_name}</b>{answer.patient_context.report_matches.length ? answer.patient_context.report_matches.map((match, i) => <p key={i} className="mt-1">• {match}</p>) : <p className="mt-1">No matching extracted report passages found.</p>}</div>}
      <p className="mt-2 text-[10px] text-slate-500">Status: {answer.status}{answer.model ? ` · ${answer.model}` : ''}. Evidence is below for clinician review.</p>
    </div>}
    {data && <div className="mt-4 space-y-3">
      {data.embedding_mode && <p className="text-[10px] text-slate-400">Index mode: {data.embedding_mode}</p>}
      <p className="text-[11px] font-semibold text-slate-600">Retrieved evidence and sources</p>
      {data.results.map((result, index) => <article key={index} className="rounded-lg border border-slate-100 p-3 text-xs">
        <div className="flex justify-between gap-3"><div className="font-semibold text-slate-800">[{index + 1}] {result.source}</div><span className="text-slate-400">relevance {result.score}</span></div>
        {result.section && <p className="mt-1 text-[11px] font-medium text-[#0e7c5b]">{result.section}</p>}
        <p className="mt-2 leading-relaxed text-slate-600">{result.text}</p>
        {result.source_url && <a href={result.source_url} target="_blank" rel="noreferrer" className="mt-2 inline-flex items-center gap-1 text-[#0e7c5b] hover:underline font-medium">View source <ExternalLink className="w-3 h-3" /></a>}
      </article>)}
    </div>}
  </section>;
}
