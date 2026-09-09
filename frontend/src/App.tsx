import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RiskAnalysis from './pages/RiskAnalysis'
import ModelComparison from './pages/ModelComparison'
import AnomalyExplorer from './pages/AnomalyExplorer'
import RLTrainer from './pages/RLTrainer'
import Recommendations from './pages/Recommendations'
import EthicsAudit from './pages/EthicsAudit'

function App() {
    return (
        <Routes>
            <Route path="/" element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="risk" element={<RiskAnalysis />} />
                <Route path="models" element={<ModelComparison />} />
                <Route path="anomalies" element={<AnomalyExplorer />} />
                <Route path="rl" element={<RLTrainer />} />
                <Route path="recommendations" element={<Recommendations />} />
                <Route path="ethics" element={<EthicsAudit />} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
        </Routes>
    )
}

export default App