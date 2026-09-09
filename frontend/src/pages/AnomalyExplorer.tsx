import { useState } from 'react'
import { Radar, Scan, ShieldAlert, Cpu, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { apiClient } from '../api/client'

interface AnomalyResult {
    anomalies: Record<string, { is_anomaly: boolean; anomaly_score: number }>
}

const defaultFeatures = {
    attempts: 15,
    success: 0,
    hour_of_day: 2,
    is_weekend: 1,
    unusual_hour: 1,
    failed_attempts_24h: 30,
    cpu_usage: 94.2,
    memory_usage: 89.1,
    network_connections: 62,
    file_operations: 410,
    registry_changes: 78,
    unusual_process: 1,
    high_privilege: 1,
    files_accessed: 120,
    sensitive_files_accessed: 35,
    data_transferred_mb: 850.0,
    applications_used: 1,
    new_device: 1,
    new_location: 1,
    login_frequency_7d: 1,
}

export default function AnomalyExplorer() {
    const [features, setFeatures] = useState<Record<string, number>>(defaultFeatures)
    const [result, setResult] = useState<AnomalyResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleDetect = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await apiClient.detectAnomalies(features)
            setResult(response)
        } catch (e) {
            setError('Failed to detect anomalies. Make sure the API server is running.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2.5">
                        <Radar className="w-7 h-7 text-accent-yellow" />
                        Zero-Day Anomaly Radar
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Unsupervised multi-detector anomaly scan using Isolation Forest and Local Outlier Factor (LOF).
                    </p>
                </div>
                <button
                    onClick={handleDetect}
                    disabled={loading}
                    className="btn-gradient-yellow flex items-center gap-2"
                >
                    <Scan className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    {loading ? 'Scanning Anomalies...' : 'Execute Anomaly Scan'}
                </button>
            </div>

            {error && (
                <div className="p-4 bg-accent-red/10 border border-accent-red/30 rounded-xl text-accent-red text-sm flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Feature Input Panel */}
                <div className="lg:col-span-7 card space-y-4">
                    <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                        <span className="font-bold text-sm text-white">Suspicious Vector Signals</span>
                        <span className="text-xs font-mono text-slate-400">Zero-Day Simulation</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 max-h-[500px] overflow-y-auto pr-2">
                        {Object.entries(features).map(([key, val]) => (
                            <div key={key} className="p-3 rounded-lg bg-[#0a0f19] border border-white/[0.05] space-y-1">
                                <div className="flex justify-between items-center text-xs font-mono">
                                    <span className="text-slate-300 capitalize">{key.replace(/_/g, ' ')}</span>
                                    <span className="text-accent-yellow font-bold">{val}</span>
                                </div>
                                <input
                                    type="number"
                                    value={val}
                                    onChange={(e) => setFeatures({ ...features, [key]: parseFloat(e.target.value) || 0 })}
                                    className="input-field w-full py-1 text-xs"
                                />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Detector Result Gauges */}
                <div className="lg:col-span-5 space-y-6">
                    {result ? (
                        <div className="space-y-4">
                            {Object.entries(result.anomalies).map(([detector, info]) => {
                                const isAno = info.is_anomaly
                                return (
                                    <div
                                        key={detector}
                                        className={`card ${isAno ? 'card-glow-red' : 'card-glow-green'} p-5 space-y-3`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <Cpu className={`w-4 h-4 ${isAno ? 'text-accent-red' : 'text-accent-green'}`} />
                                                <h3 className="font-bold text-sm text-white uppercase font-mono">{detector.replace(/_/g, ' ')}</h3>
                                            </div>
                                            <span className={isAno ? 'badge-critical' : 'badge-minimal'}>
                                                {isAno ? 'ANOMALY DETECTED' : 'NORMAL PATTERN'}
                                            </span>
                                        </div>

                                        <div className="flex items-baseline justify-between text-xs font-mono">
                                            <span className="text-slate-400">Outlier Deviation Score</span>
                                            <span className={`text-xl font-bold ${isAno ? 'text-accent-red' : 'text-accent-green'}`}>
                                                {(info.anomaly_score * 100).toFixed(1)}%
                                            </span>
                                        </div>

                                        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full ${isAno ? 'bg-gradient-to-r from-orange-500 to-accent-red' : 'bg-accent-green'}`}
                                                style={{ width: `${Math.min(100, Math.max(0, info.anomaly_score * 100))}%` }}
                                            />
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    ) : (
                        <div className="card text-center py-16 text-slate-400 space-y-3">
                            <Radar className="w-10 h-10 text-accent-yellow mx-auto opacity-70 animate-pulse" />
                            <p className="text-sm font-mono">Click <strong>Execute Anomaly Scan</strong> to evaluate the host and network signals against zero-day boundary baselines.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}