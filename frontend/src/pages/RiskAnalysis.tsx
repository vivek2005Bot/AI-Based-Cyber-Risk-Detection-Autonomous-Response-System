import { useState } from 'react'
import { Activity, AlertTriangle, ShieldCheck, Flame, Cpu, Sliders, RefreshCw, BarChart2 } from 'lucide-react'
import { apiClient, RiskAssessmentResponse } from '../api/client'

const defaultFeatures = {
    attempts: 12,
    success: 0,
    hour_of_day: 3,
    is_weekend: 1,
    unusual_hour: 1,
    failed_attempts_24h: 18,
    cpu_usage: 88.5,
    memory_usage: 92.4,
    network_connections: 54,
    file_operations: 310,
    registry_changes: 45,
    unusual_process: 1,
    high_privilege: 1,
    files_accessed: 95,
    sensitive_files_accessed: 14,
    data_transferred_mb: 480.5,
    applications_used: 3,
    new_device: 1,
    new_location: 1,
    login_frequency_7d: 2,
}

export default function RiskAnalysis() {
    const [features, setFeatures] = useState<Record<string, number>>(defaultFeatures)
    const [result, setResult] = useState<RiskAssessmentResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleAssess = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await apiClient.assessRisk({
                features,
                source_type: 'cicids2017',
                event_type: 'auth',
                cvss_score: 8.8,
                asset_criticality: 90,
                threat_score: 0.85,
                incident_count: 4,
                hours_since_event: 0.8,
            })
            setResult(response)
        } catch (e) {
            setError('Failed to assess risk. Make sure the API server is running.')
        } finally {
            setLoading(false)
        }
    }

    const getScoreColor = (score: number) => {
        if (score >= 0.7) return 'text-accent-red'
        if (score >= 0.4) return 'text-accent-yellow'
        return 'text-accent-green'
    }

    const getScoreBg = (score: number) => {
        if (score >= 0.7) return 'bg-accent-red'
        if (score >= 0.4) return 'bg-accent-yellow'
        return 'bg-accent-green'
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2.5">
                        <Flame className="w-7 h-7 text-accent-red" />
                        Evidence-Based Multimodal Risk Fusion
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Unified risk estimation combining supervised confidence, unsupervised anomaly scores, and contextual CVSS threat metrics.
                    </p>
                </div>
                <button
                    onClick={handleAssess}
                    disabled={loading}
                    className="btn-gradient-red flex items-center gap-2"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    {loading ? 'Evaluating Fusion...' : 'Compute Fused Risk'}
                </button>
            </div>

            {error && (
                <div className="p-4 bg-accent-red/10 border border-accent-red/30 rounded-xl text-accent-red text-sm flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    {error}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Left: Input Telemetry Sliders / Values */}
                <div className="lg:col-span-7 card space-y-5">
                    <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                        <div className="flex items-center gap-2">
                            <Sliders className="w-4 h-4 text-accent-yellow" />
                            <h2 className="font-bold text-sm text-white">Interactive Host & Network Signals</h2>
                        </div>
                        <button
                            onClick={() => setFeatures(defaultFeatures)}
                            className="text-[11px] font-mono text-slate-400 hover:text-accent-yellow"
                        >
                            Reset Defaults
                        </button>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[500px] overflow-y-auto pr-2">
                        {Object.entries(features).map(([key, val]) => (
                            <div key={key} className="p-3 rounded-lg bg-[#0a0f19] border border-white/[0.05] space-y-1.5">
                                <div className="flex justify-between items-center text-xs font-mono">
                                    <span className="text-slate-300 capitalize">{key.replace(/_/g, ' ')}</span>
                                    <span className="text-accent-yellow font-bold">{val}</span>
                                </div>
                                <input
                                    type="number"
                                    step={key.includes('mb') || key.includes('usage') ? '0.1' : '1'}
                                    value={val}
                                    onChange={(e) => setFeatures({ ...features, [key]: parseFloat(e.target.value) || 0 })}
                                    className="input-field w-full py-1 text-xs"
                                />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right: Fusion Gauge & Decomposed Results */}
                <div className="lg:col-span-5 space-y-6">
                    {result ? (
                        <>
                            {/* Main Fused Score Card */}
                            <div className={`card ${result.risk_level === 'critical' || result.risk_level === 'high' ? 'card-glow-red' : result.risk_level === 'medium' ? 'card-glow-yellow' : 'card-glow-green'} text-center p-6 space-y-4`}>
                                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                                    <span>FUSED RISK LEVEL</span>
                                    <span className={`badge-${result.risk_level} uppercase`}>{result.risk_level}</span>
                                </div>

                                <div className="py-2">
                                    <div className={`text-6xl font-black font-mono tracking-tight ${getScoreColor(result.risk_score)}`}>
                                        {(result.risk_score * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-xs font-mono text-slate-400 mt-2">
                                        Estimated Composite Exploitation Probability
                                    </div>
                                </div>

                                <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5 border border-white/10">
                                    <div
                                        className={`h-full rounded-full transition-all duration-700 ${getScoreBg(result.risk_score)}`}
                                        style={{ width: `${Math.round(result.risk_score * 100)}%` }}
                                    />
                                </div>
                            </div>

                            {/* Evidence Decomposition Breakdown */}
                            <div className="card space-y-4">
                                <div className="flex items-center gap-2 border-b border-white/[0.06] pb-3">
                                    <BarChart2 className="w-4 h-4 text-accent-green" />
                                    <h3 className="font-bold text-sm text-white">Tri-Pillar Evidence Decomposition</h3>
                                </div>

                                <div className="space-y-3.5">
                                    {/* Supervised Component */}
                                    <div>
                                        <div className="flex justify-between text-xs font-mono mb-1">
                                            <span className="text-slate-300 flex items-center gap-1.5">
                                                <Cpu className="w-3.5 h-3.5 text-accent-red" />
                                                Supervised ML Probability (40%)
                                            </span>
                                            <span className="text-white font-bold">{(result.supervised_score * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-accent-red rounded-full" style={{ width: `${result.supervised_score * 100}%` }} />
                                        </div>
                                    </div>

                                    {/* Anomaly Component */}
                                    <div>
                                        <div className="flex justify-between text-xs font-mono mb-1">
                                            <span className="text-slate-300 flex items-center gap-1.5">
                                                <Activity className="w-3.5 h-3.5 text-accent-yellow" />
                                                Unsupervised Anomaly Score (35%)
                                            </span>
                                            <span className="text-white font-bold">{(result.anomaly_score * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-accent-yellow rounded-full" style={{ width: `${result.anomaly_score * 100}%` }} />
                                        </div>
                                    </div>

                                    {/* Contextual CVSS Component */}
                                    <div>
                                        <div className="flex justify-between text-xs font-mono mb-1">
                                            <span className="text-slate-300 flex items-center gap-1.5">
                                                <ShieldCheck className="w-3.5 h-3.5 text-accent-green" />
                                                Contextual Asset Impact (25%)
                                            </span>
                                            <span className="text-white font-bold">{(result.contextual_score * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-accent-green rounded-full" style={{ width: `${result.contextual_score * 100}%` }} />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="card text-center py-16 text-slate-400 space-y-3">
                            <Flame className="w-10 h-10 text-accent-yellow mx-auto opacity-70 animate-pulse" />
                            <p className="text-sm font-mono">Adjust security indicators on the left and click <strong>Compute Fused Risk</strong> to evaluate composite vulnerability.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}