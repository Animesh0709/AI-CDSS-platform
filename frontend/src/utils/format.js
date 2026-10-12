export const patientCode = (id) => `#PT-${String(id).padStart(4, '0')}`;

export const apiErrorMessage = (err, fallback) => {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${d.loc ? d.loc.join('->') + ': ' : ''}${d.msg}`).join('; ');
  }
  return fallback;
};
