import React from 'react';
import { Link } from 'react-router-dom';
import { User, HeartPulse, Check } from 'lucide-react';

const BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'];

const PatientForm = ({
  formData,
  onChange,
  onSubmit,
  submitting,
  mode = 'create',
  cancelTo = '/patients',
}) => {
  const isEdit = mode === 'edit';

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <div className="med-card p-6">
        <h2 className="text-sm font-bold uppercase tracking-wider mb-4 flex items-center space-x-2 text-[#0f4c3a]">
          <User className="w-4 h-4 text-[#0e7c5b]" />
          <span>Patient Demographics</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">First Name *</label>
            <input type="text" name="first_name" required value={formData.first_name} onChange={onChange} className="input-med" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Last Name *</label>
            <input type="text" name="last_name" required value={formData.last_name} onChange={onChange} className="input-med" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Date of Birth</label>
            <input type="date" name="date_of_birth" value={formData.date_of_birth || ''} onChange={onChange} className="input-med" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Gender *</label>
            <select name="gender" value={formData.gender} onChange={onChange} className="input-med">
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Blood Group</label>
            <select name="blood_group" value={formData.blood_group || 'O+'} onChange={onChange} className="input-med">
              {BLOOD_GROUPS.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Phone Number</label>
            <input type="tel" name="phone" value={formData.phone || ''} onChange={onChange} className="input-med" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Email Address</label>
            <input type="email" name="email" value={formData.email || ''} onChange={onChange} className="input-med" />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Residential Address</label>
            <input type="text" name="address" value={formData.address || ''} onChange={onChange} className="input-med" />
          </div>
        </div>
      </div>

      {!isEdit && (
        <div className="med-card p-6">
          <h2 className="text-sm font-bold uppercase tracking-wider mb-4 flex items-center space-x-2 text-[#0f4c3a]">
            <HeartPulse className="w-4 h-4 text-[#0e7c5b]" />
            <span>Primary Diagnosis / Initial Condition</span>
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Primary Medical Condition</label>
              <input type="text" name="initial_condition" value={formData.initial_condition || ''} onChange={onChange} className="input-med" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Observations</label>
              <textarea name="initial_condition_desc" rows="3" value={formData.initial_condition_desc || ''} onChange={onChange} className="input-med" />
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center justify-end space-x-3 pt-2">
        <Link to={cancelTo} className="btn-med-secondary text-sm">Cancel</Link>
        <button type="submit" disabled={submitting} className="btn-med-primary text-sm flex items-center space-x-1.5">
          {submitting ? 'Saving to Database...' : isEdit ? 'Save Changes' : 'Save & Open Profile'}
          {!submitting && <Check className="w-4 h-4" />}
        </button>
      </div>
    </form>
  );
};

export default PatientForm;
