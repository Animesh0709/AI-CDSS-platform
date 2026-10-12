import React, { useEffect, useState } from 'react';
import { patientService } from '../services/api';

export default function PatientPicker({ onSelect }) {
  const [patients, setPatients] = useState([]), [id, setId] = useState(''), [error, setError] = useState('');
  useEffect(() => { patientService.getPatients().then(({ data }) => { setPatients(data); if (data[0]) { setId(String(data[0].id)); onSelect(data[0].id); } }).catch(() => setError('Unable to load patients.')); }, [onSelect]);
  const change = (value) => { setId(value); onSelect(value ? Number(value) : null); };
  return <section className="med-card p-5"><label className="block text-xs font-semibold text-slate-700 mb-1">Patient</label><select value={id} onChange={(e) => change(e.target.value)} className="input-med max-w-md text-sm"><option value="">Select a patient</option>{patients.map((p) => <option key={p.id} value={p.id}>{p.first_name} {p.last_name}</option>)}</select>{error && <p className="mt-2 text-xs text-rose-700">{error}</p>}{!patients.length && !error && <p className="mt-2 text-xs text-slate-500">Add a patient first.</p>}</section>;
}
