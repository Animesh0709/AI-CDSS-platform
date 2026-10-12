import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, Users, UserPlus, FileText, FlaskConical, 
  TrendingUp, Sparkles, BookOpen, Bot, ClipboardPlus, ShieldCheck, HelpCircle, Microscope, GitCompare, Layers
} from 'lucide-react';

const Sidebar = () => {
  const activeClass = "flex items-center space-x-3 px-3.5 py-2.5 rounded-xl bg-[#ecfdf5] text-[#0f4c3a] font-medium border border-[#a7f3d0] shadow-xs";
  const inactiveClass = "flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-slate-600 hover:bg-[#f1f7f4] hover:text-[#0f4c3a] font-medium transition-colors";

  return (
    <aside className="w-64 bg-white border-r border-[#e2ece7] min-h-[calc(100vh-4rem)] p-4 flex flex-col justify-between hidden md:flex">
      <div className="space-y-6">
        {/* Main Navigation */}
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 px-3.5 mb-2">
            Core Modules
          </div>
          <nav className="space-y-1">
            <NavLink to="/dashboard" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <LayoutDashboard className="w-4 h-4 text-[#0e7c5b]" />
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/patients" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <Users className="w-4 h-4 text-[#0e7c5b]" />
              <span>Patients</span>
            </NavLink>
            <NavLink to="/patients/new" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <UserPlus className="w-4 h-4 text-[#0e7c5b]" />
              <span>Add Patient</span>
            </NavLink>
          </nav>
        </div>

        {/* Clinical Intelligence Modules */}
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 px-3.5 mb-2">
            AI Modules
          </div>
          <nav className="space-y-1">
            <NavLink to="/reports" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <FileText className="w-4 h-4 text-[#0e7c5b]" />
              <span>Medical Reports</span>
            </NavLink>

            <NavLink to="/laboratory" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <FlaskConical className="w-4 h-4 text-[#0e7c5b]" />
              <span>Lab Value Engine</span>
            </NavLink>

            <NavLink to="/readmission" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <TrendingUp className="w-4 h-4 text-[#0e7c5b]" />
              <span>Readmission Risk</span>
            </NavLink>

            <NavLink to="/explainability" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <Sparkles className="w-4 h-4 text-[#0e7c5b]" />
              <span>Risk Explanations</span>
            </NavLink>

            <NavLink to="/evidence" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <BookOpen className="w-4 h-4 text-[#0e7c5b]" />
              <span>Clinical Guidelines</span>
            </NavLink>

            <NavLink to="/assistant" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <Bot className="w-4 h-4 text-[#0e7c5b]" />
              <span>AI Clinical Assistant</span>
            </NavLink>

            <NavLink to="/clinical-summary" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <ClipboardPlus className="w-4 h-4 text-[#0e7c5b]" />
              <span>AI Clinical Summary</span>
            </NavLink>

            <NavLink to="/disease-diagnosis" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <Microscope className="w-4 h-4 text-[#0e7c5b]" />
              <span>Disease Diagnosis</span>
            </NavLink>

            <NavLink to="/tumor-progression" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <GitCompare className="w-4 h-4 text-[#0e7c5b]" />
              <span>Tumor Progression</span>
            </NavLink>

            <NavLink to="/tumor-segmentation" className={({ isActive }) => isActive ? activeClass : inactiveClass}>
              <Layers className="w-4 h-4 text-[#0e7c5b]" />
              <span>3D Tumor Segmentation</span>
            </NavLink>
          </nav>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
