import React from 'react';

const StatCard = ({ title, value, subtitle, icon: Icon, color = 'green', trend }) => {
  const colorStyles = {
    green: {
      bg: 'bg-emerald-50 text-[#0e7c5b] border-emerald-200',
      badge: 'bg-emerald-100 text-emerald-800'
    },
    blue: {
      bg: 'bg-sky-50 text-sky-700 border-sky-200',
      badge: 'bg-sky-100 text-sky-800'
    },
    amber: {
      bg: 'bg-amber-50 text-amber-700 border-amber-200',
      badge: 'bg-amber-100 text-amber-800'
    },
    purple: {
      bg: 'bg-indigo-50 text-indigo-700 border-indigo-200',
      badge: 'bg-indigo-100 text-indigo-800'
    }
  };

  const style = colorStyles[color] || colorStyles.green;

  return (
    <div className="med-card p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{title}</p>
          <h3 className="text-2xl font-bold text-slate-800 mt-1">{value}</h3>
        </div>
        <div className={`w-11 h-11 rounded-xl flex items-center justify-center border ${style.bg}`}>
          {Icon && <Icon className="w-5 h-5" />}
        </div>
      </div>
      {(subtitle || trend) && (
        <div className="mt-3 flex items-center justify-between text-xs text-slate-500 border-t border-slate-100 pt-2.5">
          <span>{subtitle}</span>
          {trend && (
            <span className={`px-1.5 py-0.5 rounded font-medium text-[10px] ${style.badge}`}>
              {trend}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default StatCard;
