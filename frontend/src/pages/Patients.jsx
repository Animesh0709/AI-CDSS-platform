import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { patientService } from '../services/api';
import { 
  Users, Search, UserPlus, Trash2, Eye, Pencil,
  AlertCircle, Phone, Mail, Calendar, Droplets
} from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import useDebouncedValue from '../hooks/useDebouncedValue';
import { patientCode } from '../utils/format';

const Patients = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const debouncedSearch = useDebouncedValue(search, 300);
  const [genderFilter, setGenderFilter] = useState('ALL');
  const [error, setError] = useState('');

  useEffect(() => {
    fetchPatients();
  }, [debouncedSearch]);

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const res = await patientService.getPatients(debouncedSearch);
      setPatients(res.data);
    } catch (err) {
      console.error(err);
      setError('Unable to fetch patient directory.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id, name, e) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to remove the record for ${name}?`)) {
      try {
        await patientService.deletePatient(id);
        setPatients(patients.filter(p => p.id !== id));
      } catch (err) {
        console.error(err);
        alert('Failed to delete patient record.');
      }
    }
  };

  const filteredPatients = patients.filter(p => {
    if (genderFilter === 'ALL') return true;
    return p.gender.toUpperCase() === genderFilter;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[#e2ece7]">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight flex items-center space-x-2">
            <Users className="w-6 h-6 text-[#0e7c5b]" />
            <span>Patient Directory</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Manage electronic medical records, clinical encounters, and readmission risk tracking.
          </p>
        </div>

        <Link to="/patients/new" className="btn-med-primary text-xs flex items-center space-x-1.5 self-start sm:self-auto shadow-sm">
          <UserPlus className="w-4 h-4" />
          <span>Add New Patient</span>
        </Link>
      </div>

      <ClinicalDisclaimer compact={true} />

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center space-x-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Search & Filter Bar */}
      <div className="med-card p-4 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by patient name, phone, or email..."
            className="input-med !pl-9 text-xs"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto justify-end">
          <span className="text-xs text-slate-500 font-medium">Filter Gender:</span>
          <select
            value={genderFilter}
            onChange={(e) => setGenderFilter(e.target.value)}
            className="input-med text-xs py-1.5 w-32"
          >
            <option value="ALL">All Genders</option>
            <option value="MALE">Male</option>
            <option value="FEMALE">Female</option>
            <option value="OTHER">Other</option>
          </select>
        </div>
      </div>

      {/* Patient Table */}
      <div className="med-card overflow-hidden">
        {loading ? (
          <div className="py-16 text-center text-sm text-slate-400">Loading patients from PostgreSQL database...</div>
        ) : filteredPatients.length === 0 ? (
          <div className="py-16 text-center">
            <div className="w-12 h-12 rounded-full bg-emerald-50 text-[#0e7c5b] flex items-center justify-center mx-auto mb-3">
              <Users className="w-6 h-6" />
            </div>
            <p className="text-base font-semibold text-slate-700">No patients found</p>
            <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
              {search ? 'Try clearing your search criteria or add a new patient record.' : 'Your patient registry is currently empty.'}
            </p>
            <Link to="/patients/new" className="mt-4 btn-med-primary text-xs inline-flex items-center space-x-1.5">
              <UserPlus className="w-3.5 h-3.5" />
              <span>Create New Patient</span>
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#f8faf9] text-slate-500 uppercase tracking-wider border-b border-[#e2ece7]">
                <tr>
                  <th className="py-3.5 px-4 font-semibold">Patient</th>
                  <th className="py-3.5 px-4 font-semibold">Demographics</th>
                  <th className="py-3.5 px-4 font-semibold">Contact</th>
                  <th className="py-3.5 px-4 font-semibold">Blood Group</th>
                  <th className="py-3.5 px-4 font-semibold">Medical History</th>
                  <th className="py-3.5 px-4 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredPatients.map((p) => (
                  <tr key={p.id} className="hover:bg-emerald-50/30 transition-colors">
                    <td className="py-3.5 px-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-emerald-100 text-[#0f4c3a] flex items-center justify-center font-bold text-xs flex-shrink-0">
                          {p.first_name.charAt(0)}{p.last_name.charAt(0)}
                        </div>
                        <div>
                          <div className="font-bold text-slate-800 text-sm">
                            {p.first_name} {p.last_name}
                          </div>
                          <div className="text-[11px] text-slate-400">ID: {patientCode(p.id)}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="text-slate-700 font-medium">{p.gender}</div>
                      <div className="text-slate-400 text-[11px] flex items-center space-x-1 mt-0.5">
                        <Calendar className="w-3 h-3 text-slate-400" />
                        <span>{p.date_of_birth || 'Not specified'}</span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="text-slate-700 flex items-center space-x-1">
                        <Phone className="w-3 h-3 text-slate-400" />
                        <span>{p.phone || 'No phone'}</span>
                      </div>
                      {p.email && (
                        <div className="text-slate-400 text-[11px] flex items-center space-x-1 mt-0.5">
                          <Mail className="w-3 h-3 text-slate-400" />
                          <span className="truncate max-w-[150px]">{p.email}</span>
                        </div>
                      )}
                    </td>
                    <td className="py-3.5 px-4">
                      {p.blood_group ? (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-rose-50 text-rose-700 border border-rose-200 font-semibold text-[11px]">
                          <Droplets className="w-3 h-3 text-rose-500" />
                          <span>{p.blood_group}</span>
                        </span>
                      ) : (
                        <span className="text-slate-400">N/A</span>
                      )}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-[#0f4c3a] border border-emerald-200 text-xs font-semibold">
                        {p.history_count} {p.history_count === 1 ? 'condition' : 'conditions'}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <div className="flex items-center justify-end space-x-2">
                        <Link
                          to={`/patients/${p.id}`}
                          className="btn-med-secondary text-xs py-1 px-2.5 flex items-center space-x-1"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>Profile</span>
                        </Link>
                        <Link
                          to={`/patients/${p.id}/edit`}
                          className="btn-med-secondary text-xs py-1 px-2.5 flex items-center space-x-1"
                        >
                          <Pencil className="w-3.5 h-3.5" />
                          <span>Edit</span>
                        </Link>
                        <button
                          onClick={(e) => handleDelete(p.id, `${p.first_name} ${p.last_name}`, e)}
                          title="Delete Record"
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Patients;
