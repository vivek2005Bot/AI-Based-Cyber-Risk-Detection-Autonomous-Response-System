import { useState } from 'react'
import { ThumbsUp, Shield, AlertTriangle, Lock, UserX, Wrench, Search, ArrowUpRight, Activity, CheckCircle2, Zap } from 'lucide-react'
import { apiClient, RiskAssessmentResponse } from '../api/client'

const actionIcons: Record<string, any> = {
    monitor: Activity,
    alert: AlertTriangle,
    block_ip: Shield,
    quarantine_host: Lock,
    reset_credentials: UserX,
    patch_vulnerability: Wrench,
    investigate: Search,
    escalate: ArrowUpRight,
}

export default function Recommendations() {
    const [result, setResult] = useState<RiskAssessmentResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleAnalyze = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await apiClient.assessRisk({
                features: {
                    attempts: 25,
                    success: 0,
                    hour_of_day: 3,
                    is_weekend: 1,
                    unusual_hour: 1,
                    failed_attempts_24h: 45,
                    cpu_usage: 85.5,
                    memory_usage: 90.2,
                    network_connections: 45,
                    file_operations: 200,
                    registry_changes: 50,
                    unusual_process: 1,
                    high_privilege: 1,
                    files_accessed: 80,
                    sensitive_files_accessed: 20,
                    data_transferred_mb: 500,
                    applications_used: 2,
                    new_device: 1,
                    new_location: 1,
                    login_frequency_7d: 3,
                },
                source_type: 'cicids2017',
                event_type: 'auth',
                cvss_score: 9.2,
                asset_criticality: 95,
                threat_score: 0.92,
                incident_count: 5,
                hours_since_event: 0.3,
            })
            setResult(response)
        } catch (e) {
            setError('Failed to get recommendations. Make sure the API server is running.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2.5">
                        <ThumbsUp className="w-7 h-7 text-accent-green" />
                        Automated Incident Remediation Engine
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Hybrid recommendation system (Content-Based + Collaborative Filtering) for autonomous countermeasure deployment.
                    </p>
                </div>
                <button
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="btn-gradient-green flex items-center gap-2"
                >
                    <Zap className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    {loading ? 'Generating Strategy...' : 'Synthesize Remediation Playbook'}
                </button>
            </div>

            {error && (
                <div className="p-4 bg-accent-red/10 border border-accent-red/30 rounded-xl text-accent-red text-sm flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    {error}
                </div>
            )}

            {result && (
                <div className="space-y-6">
                    {/* Event Context Card */}
                    <div className="card-glow-red p-6">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                            <div>
                                <span className="text-xs font-mono text-accent-red font-bold uppercase tracking-wider">
                                    HIGH-SEVERITY INCIDENT PROFILE
                                </span>
                                <h3 className="text-xl font-bold text-white mt-1">Brute-Force & Privilege Escalation Attempt</h3>
                                <p className="text-xs text-slate-400 font-mono mt-1">Origin IP: 192.168.1.105 | Target Host: Core-Database-Server</p>
                            </div>
                            <div className="text-right">
                                <div className="text-2xl font-black font-mono text-accent-red">{(result.risk_score * 100).toFixed(1)}% RISK</div>
                                <span className="badge-critical uppercase text-[10px] mt-1">{result.risk_level}</span>
                            </div>
                        </div>
                    </div>

                    {/* Prescribed Countermeasures */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-mono font-bold text-slate-300 uppercase tracking-wider">
                            Ranked Countermeasure Playbook
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {result.recommendations.map((rec, i) => {
                                const Icon = actionIcons[rec.action] || Shield
                                return (
                                    <div
                                        key={i}
                                        className="card hover:border-accent-green/40 transition-all p-5 space-y-3"
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className="w-9 h-9 rounded-lg bg-accent-green/15 text-accent-green flex items-center justify-center">
                                                    <Icon className="w-5 h-5" />
                                                </div>
                                                <div>
                                                    <h4 className="font-bold text-sm text-white capitalize">{rec.action.replace(/_/g, ' ')}</h4>
                                                    <span className="text-[10px] font-mono text-slate-400">Hybrid Prescriptive Match</span>
                                                </div>
                                            </div>
                                            <span className="badge-minimal text-[10px] font-mono">
                                                {(rec.score * 100).toFixed(0)}% MATCH
                                            </span>
                                        </div>

                                        <p className="text-xs text-slate-300 leading-relaxed">{rec.description}</p>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>
            )}

            {!result && (
                <div className="card text-center py-20 text-slate-400 space-y-3">
                    <ThumbsUp className="w-10 h-10 text-accent-green mx-auto opacity-70 animate-pulse" />
                    <p className="text-sm font-mono">Click <strong>Synthesize Remediation Playbook</strong> to evaluate an active attack scenario and generate prioritized mitigation steps.</p>
                </div>
            )}
        </div>
    )
}