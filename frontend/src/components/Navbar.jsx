import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Stethoscope, LogOut, Building2, ShieldCheck } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  return (
    <header className="bg-white border-b border-[#e2ece7] sticky top-0 z-30 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Brand Logo & Name */}
          <Link to="/dashboard" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#0f4c3a] to-[#0e7c5b] flex items-center justify-center text-white shadow-sm transition-transform group-hover:scale-105">
              <Stethoscope className="w-5 h-5 text-emerald-100" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-slate-800">
                  AI<span className="text-[#0e7c5b]">-CDSS</span>
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded border border-emerald-200">
                  Doctor Portal
                </span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">Explainable Clinical Decision Support System</p>
            </div>
          </Link>

          {/* Right: Doctor Profile & Actions */}
          <div className="flex items-center space-x-4">
            {user && (
              <>
                <div className="hidden md:flex items-center space-x-2 text-xs text-slate-600 bg-[#f7faf8] px-3 py-1.5 rounded-lg border border-[#e2ece7]">
                  <Building2 className="w-3.5 h-3.5 text-[#0e7c5b]" />
                  <span className="font-medium truncate max-w-[180px]">{user.hospital_affiliation || 'City Central Hospital'}</span>
                </div>

                <div className="flex items-center space-x-3 pl-2 border-l border-slate-200">
                  <div className="w-9 h-9 rounded-full bg-emerald-100 border border-emerald-300 flex items-center justify-center text-[#0f4c3a] font-semibold text-sm">
                    {user.full_name ? user.full_name.charAt(0).toUpperCase() : 'D'}
                  </div>
                  <div className="hidden lg:block text-left">
                    <div className="text-sm font-semibold text-slate-800 leading-tight flex items-center space-x-1">
                      <span>Dr. {user.full_name}</span>
                      <ShieldCheck className="w-3.5 h-3.5 text-[#0e7c5b]" />
                    </div>
                    <div className="text-[11px] text-slate-500">{user.specialization || 'Attending Physician'}</div>
                  </div>
                </div>

                <button
                  onClick={handleLogout}
                  title="Log Out"
                  className="p-2 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
