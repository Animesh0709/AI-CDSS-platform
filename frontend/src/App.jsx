import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import AddPatient from './pages/AddPatient';
import EditPatient from './pages/EditPatient';
import PatientDetails from './pages/PatientDetails';
import ClinicalAssistant from './pages/ClinicalAssistant';
import Explainability from './pages/Explainability';
import Reports from './pages/Reports';
import Laboratory from './pages/Laboratory';
import MedicalEvidence from './pages/MedicalEvidence';
import Readmission from './pages/Readmission';
import ClinicalSummary from './pages/ClinicalSummary';
import DiseaseDiagnosis from './pages/DiseaseDiagnosis';
import TumorProgression from './pages/TumorProgression';
import TumorSegmentation from './pages/TumorSegmentation';
import ErrorBoundary from './components/ErrorBoundary';

// Layout for protected pages
const AppLayout = ({ children }) => {
  return (
    <div className="min-h-screen bg-[#f7faf8] flex flex-col">
      <Navbar />
      <div className="flex-1 flex max-w-7xl w-full mx-auto">
        <Sidebar />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Protected Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <Dashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/patients"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <Patients />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/patients/new"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <AddPatient />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/patients/:id/edit"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <EditPatient />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/patients/:id"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <PatientDetails />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route path="/assistant" element={<ProtectedRoute><AppLayout><ClinicalAssistant /></AppLayout></ProtectedRoute>} />
          <Route path="/explainability" element={<ProtectedRoute><AppLayout><Explainability /></AppLayout></ProtectedRoute>} />
          <Route path="/reports" element={<ProtectedRoute><AppLayout><Reports /></AppLayout></ProtectedRoute>} />
          <Route path="/laboratory" element={<ProtectedRoute><AppLayout><Laboratory /></AppLayout></ProtectedRoute>} />
          <Route path="/evidence" element={<ProtectedRoute><AppLayout><MedicalEvidence /></AppLayout></ProtectedRoute>} />
          <Route path="/readmission" element={<ProtectedRoute><AppLayout><Readmission /></AppLayout></ProtectedRoute>} />
          <Route path="/clinical-summary" element={<ProtectedRoute><AppLayout><ClinicalSummary /></AppLayout></ProtectedRoute>} />
          <Route path="/disease-diagnosis" element={<ProtectedRoute><AppLayout><DiseaseDiagnosis /></AppLayout></ProtectedRoute>} />
          <Route path="/tumor-progression" element={<ProtectedRoute><AppLayout><TumorProgression /></AppLayout></ProtectedRoute>} />
          <Route path="/tumor-segmentation" element={<ProtectedRoute><AppLayout><TumorSegmentation /></AppLayout></ProtectedRoute>} />

          {/* Root Redirect */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
