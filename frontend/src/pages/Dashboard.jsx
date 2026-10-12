import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { patientService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import StatCard from '../components/StatCard';
import ClinicalDisclaimer from '../components/ClinicalDisclaimer';
import {
  Users, Activity, TrendingUp, UserPlus, ArrowRight,
  Clock, ShieldCheck, ChevronRight, AlertCircle, FileText, FlaskConical, Bot
} from 'lucide-react';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const res = await patientService.getStats();
      setStats(res.data);
    } catch (err) {
      console.error("Dashboard error:", err);
      setError('Unable to load clinical statistics.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[#e2ece7]">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
            Clinical Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Welcome back, <span className="font-semibold text-[#0e7c5b]">Dr. {user?.full_name}</span> • {user?.hospital_affiliation || 'City Central Hospital'}
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <Link to="/patients/new" className="btn-med-primary text-xs flex items-center space-x-1.5 shadow-sm">
            <UserPlus className="w-4 h-4" />
            <span>Add New Patient</span>
          </Link>
          <Link to="/patients" className="btn-med-secondary text-xs flex items-center space-x-1.5">
            <Users className="w-4 h-4" />
            <span>Patient Directory</span>
          </Link>
        </div>
      </div>

      {/* Doctor-in-the-Loop Safety Banner */}
      <ClinicalDisclaimer />

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center space-x-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Assigned Patients"
          value={loading ? '...' : (stats?.total_patients || 0)}
          subtitle="Database Records"
          trend="Live"
          icon={Users}
          color="green"
        />
        <StatCard
          title="Tracked Diagnoses"
          value={loading ? '...' : (stats?.total_conditions || 0)}
          subtitle="Medical history records"
          trend="Live"
          icon={Activity}
          color="blue"
        />
        <StatCard
          title="Readmission Model"
          value="XGBoost"
          subtitle="30-Day Risk Classifier"
          trend="Live"
          icon={TrendingUp}
          color="purple"
        />
        <StatCard
          title="Clinical Safety"
          value="Supervised"
          subtitle="Doctor review mandated"
          trend="100% In-Loop"
          icon={ShieldCheck}
          color="amber"
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {[
          { to: '/patients', label: 'Patients', hint: 'Live', icon: Users, live: true },
          { label: 'Reports', hint: 'Coming Soon', icon: FileText },
          { label: 'Lab Results', hint: 'Coming Soon', icon: FlaskConical },
          { label: 'Risk Prediction UI', hint: 'Coming Soon · W9', icon: TrendingUp },
          { label: 'AI Assistant', hint: 'Coming Soon', icon: Bot },
        ].map((item) => {
          const Icon = item.icon;
          const live = {
            Patients: { to: '/patients', hint: 'Live' }, Reports: { to: '/reports', hint: 'Live' },
            'Lab Results': { to: '/laboratory', hint: 'Live' }, 'Risk Prediction UI': { to: '/readmission', hint: 'Live' },
            'AI Assistant': { to: '/assistant', hint: 'Live' },
          }[item.label] || item;
          const inner = (
            <div className="med-card p-3 flex items-center space-x-3 hover:border-emerald-300 hover:bg-emerald-50/30 transition-colors">
              <Icon className="w-4 h-4 text-[#0e7c5b]" />
              <div>
                <div className="text-xs font-semibold text-slate-800">{item.label}</div>
                <div className="text-[10px] text-emerald-700">{live.hint}</div>
              </div>
            </div>
          );
          return <Link key={item.label} to={live.to}>{inner}</Link>;
        })}
      </div>
      {/* Full-Width Recent Patients Table */}
      <div className="med-card p-6 w-full">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-800 flex items-center space-x-2">
              <Clock className="w-4 h-4 text-[#0e7c5b]" />
              <span>Recently Admitted Patients</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">Direct clinical records under your supervision</p>
          </div>
          <Link to="/patients" className="text-xs font-semibold text-[#0e7c5b] hover:text-[#0a5c43] flex items-center space-x-1">
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {loading ? (
          <div className="py-12 text-center text-sm text-slate-400">Loading patient records...</div>
        ) : stats?.recent_patients?.length === 0 ? (
          <div className="py-12 text-center">
            <div className="w-12 h-12 rounded-full bg-emerald-50 text-[#0e7c5b] flex items-center justify-center mx-auto mb-3">
              <Users className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-slate-700">No patients recorded yet.</p>
            <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
              Add your first patient to start tracking medical history and testing XGBoost readmission risk predictions.
            </p>
            <Link to="/patients/new" className="mt-4 btn-med-primary text-xs inline-flex items-center space-x-1">
              <UserPlus className="w-3.5 h-3.5" />
              <span>Create Patient Record</span>
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400 uppercase tracking-wider border-b border-slate-100">
                  <th className="pb-3 font-semibold">Patient Name</th>
                  <th className="pb-3 font-semibold">Gender</th>
                  <th className="pb-3 font-semibold">DOB</th>
                  <th className="pb-3 font-semibold">Conditions</th>
                  <th className="pb-3 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {stats?.recent_patients?.map((p) => (
                  <tr key={p.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3 font-medium text-slate-800 flex items-center space-x-2.5">
                      <div className="w-7 h-7 rounded-full bg-emerald-100 text-[#0f4c3a] flex items-center justify-center font-bold text-xs">
                        {p.full_name.charAt(0)}
                      </div>
                      <span className="font-semibold">{p.full_name}</span>
                    </td>
                    <td className="py-3 text-slate-600">{p.gender}</td>
                    <td className="py-3 text-slate-600">{p.date_of_birth || 'N/A'}</td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-[11px] font-medium">
                        {p.conditions_count} tracked
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <Link
                        to={`/patients/${p.id}`}
                        className="inline-flex items-center space-x-1 text-[#0e7c5b] hover:text-[#0b6348] font-semibold text-xs bg-emerald-50/60 hover:bg-emerald-100/60 px-2.5 py-1 rounded-md transition-colors"
                      >
                        <span>Open Profile</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </Link>
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

export default Dashboard;
