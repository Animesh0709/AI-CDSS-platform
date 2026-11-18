import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Pencil, ArrowLeft, AlertCircle } from 'lucide-react';
import { patientService } from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import PatientForm from '../components/PatientForm';
import { apiErrorMessage, patientCode } from '../utils/format';

const emptyForm = {
  first_name: '', last_name: '', date_of_birth: '', gender: 'Male',
  phone: '', email: '', address: '', blood_group: 'O+',
};

const EditPatient = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [formData, setFormData] = useState(emptyForm);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const res = await patientService.getPatientById(id);
        const p = res.data;
        setFormData({
          first_name: p.first_name || '',
          last_name: p.last_name || '',
          date_of_birth: p.date_of_birth || '',
          gender: p.gender || 'Male',
          phone: p.phone || '',
          email: p.email || '',
          address: p.address || '',
          blood_group: p.blood_group || 'O+',
        });
      } catch (err) {
        setError(apiErrorMessage(err, 'Unable to load patient for editing.'));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id]);

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.first_name.trim() || !formData.last_name.trim()) {
      setError('Please provide patient first and last name.');
      return;
    }
    setError('');
    setSubmitting(true);
    try {
      await patientService.updatePatient(id, {
        first_name: formData.first_name.trim(),
        last_name: formData.last_name.trim(),
        date_of_birth: formData.date_of_birth || null,
        gender: formData.gender,
        phone: formData.phone.trim() || null,
        email: formData.email.trim() || null,
        address: formData.address.trim() || null,
        blood_group: formData.blood_group || null,
      });
      navigate(`/patients/${id}`);
    } catch (err) {
      setError(apiErrorMessage(err, 'Failed to update patient record.'));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <div className="py-20 text-center text-sm text-slate-400">Loading patient record...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center space-x-3 pb-2 border-b border-[#e2ece7]">
        <Link to={`/patients/${id}`} className="p-2 text-slate-500 hover:text-slate-800 hover:bg-white rounded-lg border border-[#e2ece7]">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight flex items-center space-x-2">
            <Pencil className="w-6 h-6 text-[#0e7c5b]" />
            <span>Edit Patient {patientCode(id)}</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">Update demographics stored in PostgreSQL</p>
        </div>
      </div>
      <ClinicalDisclaimer compact />
      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      <PatientForm
        formData={formData}
        onChange={handleChange}
        onSubmit={handleSubmit}
        submitting={submitting}
        mode="edit"
        cancelTo={`/patients/${id}`}
      />
    </div>
  );
};

export default EditPatient;
