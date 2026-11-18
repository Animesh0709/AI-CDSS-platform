import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { UserPlus, ArrowLeft, AlertCircle } from 'lucide-react';
import { patientService } from '../services/api';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import PatientForm from '../components/PatientForm';
import { apiErrorMessage } from '../utils/format';

const AddPatient = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    first_name: '', last_name: '', date_of_birth: '', gender: 'Male',
    phone: '', email: '', address: '', blood_group: 'O+',
    initial_condition: '', initial_condition_desc: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

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
      const res = await patientService.createPatient({
        first_name: formData.first_name.trim(),
        last_name: formData.last_name.trim(),
        date_of_birth: formData.date_of_birth || null,
        gender: formData.gender,
        phone: formData.phone.trim() || null,
        email: formData.email.trim() || null,
        address: formData.address.trim() || null,
        blood_group: formData.blood_group || null,
        initial_condition: formData.initial_condition.trim() || null,
        initial_condition_desc: formData.initial_condition_desc.trim() || null,
      });
      navigate(`/patients/${res.data.id}`);
    } catch (err) {
      setError(apiErrorMessage(err, 'Failed to create patient record. Please check the form fields.'));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center space-x-3 pb-2 border-b border-[#e2ece7]">
        <Link to="/patients" className="p-2 text-slate-500 hover:text-slate-800 hover:bg-white rounded-lg border border-[#e2ece7]">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight flex items-center space-x-2">
            <UserPlus className="w-6 h-6 text-[#0e7c5b]" />
            <span>Register New Patient</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">Enter demographics and baseline clinical information</p>
        </div>
      </div>
      <ClinicalDisclaimer compact />
      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      <PatientForm formData={formData} onChange={handleChange} onSubmit={handleSubmit} submitting={submitting} mode="create" />
    </div>
  );
};

export default AddPatient;
