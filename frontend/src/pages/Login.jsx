import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Stethoscope, Lock, Mail, AlertCircle, ArrowRight, ShieldCheck, UserCheck } from 'lucide-react';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || '/dashboard';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(email, password);
      navigate(from, { replace: true });
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail || 'Authentication failed. Please check your credentials.'
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleDemoFill = () => {
    setEmail('doctor@hospital.org');
    setPassword('DoctorPass123!');
  };

  return (
    <div className="min-h-screen bg-[#f7faf8] flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-[#0f4c3a] to-[#0e7c5b] flex items-center justify-center text-white shadow-md">
            <Stethoscope className="w-8 h-8 text-emerald-100" />
          </div>
        </div>
        <h2 className="mt-4 text-center text-2xl font-bold tracking-tight text-slate-800">
          Doctor Clinical Portal
        </h2>
        <p className="mt-1 text-center text-sm text-slate-500">
          Explainable AI-Based Clinical Decision Support System (AI-CDSS)
        </p>
      </div>

      <div className="mt-6 sm:mx-auto sm:w-full sm:max-w-md px-4">
        <div className="bg-white py-8 px-6 sm:px-8 border border-[#e2ece7] rounded-2xl shadow-sm">
          {error && (
            <div className="mb-5 p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-start space-x-2.5">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                Hospital Email Address
              </label>
              <div className="relative rounded-lg shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="doctor@hospital.org"
                  className="input-med !pl-10"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
                Password
              </label>
              <div className="relative rounded-lg shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="input-med !pl-10"
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={submitting}
                className="w-full btn-med-primary py-2.5 text-sm"
              >
                {submitting ? 'Verifying Credentials...' : 'Sign In to Clinical Dashboard'}
                {!submitting && <ArrowRight className="w-4 h-4 ml-2" />}
              </button>
            </div>
          </form>

          {/* Quick Demo Helper */}
          <div className="mt-5 pt-4 border-t border-slate-100 flex flex-col space-y-3">
            <button
              type="button"
              onClick={handleDemoFill}
              className="w-full btn-med-secondary py-2 text-xs flex items-center justify-center space-x-1.5"
            >
              <UserCheck className="w-3.5 h-3.5" />
              <span>Fill Quick Demo Credentials</span>
            </button>
            
            <div className="text-center text-xs text-slate-500">
              New clinician?{' '}
              <Link to="/register" className="font-semibold text-[#0e7c5b] hover:underline">
                Register Doctor Account
              </Link>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <ClinicalDisclaimer compact={true} />
        </div>
      </div>
    </div>
  );
};

export default Login;
