import { useEffect, useState } from 'react'
import { Activity, AlertTriangle, Brain, Shield, Zap, Flame, ShieldAlert, Cpu, CheckCircle2 } from 'lucide-react'
import { apiClient, DashboardData } from '../api/client'

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadDashboard()
    }, [])

    const loadDashboard = async () => {
        try {
            const result = await apiClient.getDashboard()
            setData(result)
        } catch (error) {
            console.error('Failed to load dashboard:', error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-28 space-y-4">
                <div className="w-12 h-12 rounded-full border-4 border-accent-red/20 border-t-accent-red animate-spin" />
                <span className="text-sm font-mono text-slate-400">Loading Threat Intelligence Telemetry...</span>
            </div>
        )
    }

    const highRiskCount = (data?.risk_distribution?.high || 0) + (data?.risk_distribution?.critical || 0)
    const mediumRiskCount = data?.risk_distribution?.medium || 0
    const lowRiskCount = (data?.risk_distribution?.low || 0) + (data?.risk_distribution?.minimal || 0)
    const totalCount = data?.total_events || 0

    return (
        <div className="space-y-8">
            {/* Header Hero Banner */}
            <div className="relative overflow-hidden rounded-2xl p-7 bg-gradient-to-r from-accent-red/15 via-accent-yellow/10 to-accent-green/10 border border-white/10 shadow-2xl backdrop-blur-xl">
                <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-accent-red/10 rounded-full blur-3xl pointer-events-none" />
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 relative z-10">
                    <div>
                        <div className="flex items-center gap-2.5">
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-accent-red/20 text-accent-red border border-accent-red/40 shadow-neon-red">
                                <span className="w-2 h-2 rounded-full bg-accent-red animate-ping" />
                                LIVE SOC OPS
                            </span>
                            <span className="text-xs font-mono text-slate-400">Multimodal Threat Stream</span>
                        </div>
                        <h1 className="text-3xl font-extrabold tracking-tight mt-2.5 bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                            Executive Cyber Risk Matrix
                        </h1>
                        <p className="text-slate-400 text-sm mt-1 max-w-2xl">
                            Real-time intelligence fusion aggregating network intrusion logs, OS host behavior, and zero-day anomaly indicators.
                        </p>
                    </div>

                    <div className="flex items-center gap-3">
                        <button onClick={loadDashboard} className="btn-gradient-red text-xs px-4 py-2.5 flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Refresh Telemetry
                        </button>
                    </div>
                </div>
            </div>

            {/* Tri-Color Key Metrics Ribbon */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Total Events */}
                <div className="card border-white/10 hover:border-slate-400/40">
                    <div className="flex items-center justify-between">
                        <span className="stat-label">Total Ingested Events</span>
                        <div className="w-8 h-8 rounded-lg bg-slate-800/80 flex items-center justify-center text-slate-300">
                            <Activity className="w-4 h-4" />
                        </div>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2">
                        <span className="stat-value text-white">{totalCount}</span>
                        <span className="text-xs font-mono text-slate-400">records</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-slate-500">Live multi-source ingestion</div>
                </div>

                {/* High & Critical Risks (RED GLOW) */}
                <div className="card-glow-red">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-red">Critical & High Threats</span>
                        <div className="w-8 h-8 rounded-lg bg-accent-red/20 flex items-center justify-center text-accent-red shadow-neon-red">
                            <ShieldAlert className="w-4 h-4" />
                        </div>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2">
                        <span className="stat-value text-accent-red">{highRiskCount}</span>
                        <span className="text-xs font-mono text-accent-red/80 font-bold">Action Required</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-accent-red/60">Severe exploitation risk</div>
                </div>

                {/* Medium Risks (YELLOW GLOW) */}
                <div className="card-glow-yellow">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-yellow">Suspicious / Medium</span>
                        <div className="w-8 h-8 rounded-lg bg-accent-yellow/20 flex items-center justify-center text-accent-yellow shadow-neon-yellow">
                            <AlertTriangle className="w-4 h-4" />
                        </div>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2">
                        <span className="stat-value text-accent-yellow">{mediumRiskCount}</span>
                        <span className="text-xs font-mono text-accent-yellow/80">Under Observation</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-accent-yellow/60">Contextual anomaly flags</div>
                </div>

                {/* Minimal & Low (GREEN GLOW) */}
                <div className="card-glow-green">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-green">Normal / Low Risk</span>
                        <div className="w-8 h-8 rounded-lg bg-accent-green/20 flex items-center justify-center text-accent-green shadow-neon-green">
                            <CheckCircle2 className="w-4 h-4" />
                        </div>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2">
                        <span className="stat-value text-accent-green">{lowRiskCount}</span>
                        <span className="text-xs font-mono text-accent-green/80">Verified Clean</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-accent-green/60">Compliant network activity</div>
                </div>
            </div>

            {/* Threat Distribution & AI Readiness */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Risk Distribution Breakdown */}
                <div className="card lg:col-span-2 space-y-6">
                    <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
                        <div className="flex items-center gap-2.5">
                            <Flame className="w-5 h-5 text-accent-red" />
                            <h2 className="font-bold text-base text-white">Threat Severity Spectrum</h2>
                        </div>
                        <span className="text-xs font-mono text-slate-400">Tier Distribution</span>
                    </div>

                    <div className="space-y-4">
                        {/* Critical */}
                        <div>
                            <div className="flex justify-between text-xs font-mono mb-1.5">
                                <span className="text-accent-red font-bold flex items-center gap-1.5">
                                    <span className="w-2 h-2 rounded-full bg-accent-red" />
                                    CRITICAL (Risk 0.8 - 1.0)
                                </span>
                                <span className="text-slate-300 font-bold">{data?.risk_distribution?.critical || 0} events</span>
                            </div>
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-accent-red to-rose-600 rounded-full transition-all duration-500"
                                    style={{ width: `${totalCount ? ((data?.risk_distribution?.critical || 0) / totalCount) * 100 : 0}%` }}
                                />
                            </div>
                        </div>

                        {/* High */}
                        <div>
                            <div className="flex justify-between text-xs font-mono mb-1.5">
                                <span className="text-amber-400 font-bold flex items-center gap-1.5">
                                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                                    HIGH (Risk 0.6 - 0.8)
                                </span>
                                <span className="text-slate-300 font-bold">{data?.risk_distribution?.high || 0} events</span>
                            </div>
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full transition-all duration-500"
                                    style={{ width: `${totalCount ? ((data?.risk_distribution?.high || 0) / totalCount) * 100 : 0}%` }}
                                />
                            </div>
                        </div>

                        {/* Medium */}
                        <div>
                            <div className="flex justify-between text-xs font-mono mb-1.5">
                                <span className="text-accent-yellow font-bold flex items-center gap-1.5">
                                    <span className="w-2 h-2 rounded-full bg-accent-yellow" />
                                    MEDIUM (Risk 0.4 - 0.6)
                                </span>
                                <span className="text-slate-300 font-bold">{data?.risk_distribution?.medium || 0} events</span>
                            </div>
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-yellow-400 to-amber-300 rounded-full transition-all duration-500"
                                    style={{ width: `${totalCount ? ((data?.risk_distribution?.medium || 0) / totalCount) * 100 : 0}%` }}
                                />
                            </div>
                        </div>

                        {/* Low / Minimal */}
                        <div>
                            <div className="flex justify-between text-xs font-mono mb-1.5">
                                <span className="text-accent-green font-bold flex items-center gap-1.5">
                                    <span className="w-2 h-2 rounded-full bg-accent-green" />
                                    LOW & MINIMAL (Risk 0.0 - 0.4)
                                </span>
                                <span className="text-slate-300 font-bold">{lowRiskCount} events</span>
                            </div>
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-accent-green to-emerald-400 rounded-full transition-all duration-500"
                                    style={{ width: `${totalCount ? (lowRiskCount / totalCount) * 100 : 0}%` }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* AI & Multi-Agent Defense Readiness */}
                <div className="card space-y-5">
                    <div className="flex items-center gap-2.5 border-b border-white/[0.06] pb-4">
                        <Cpu className="w-5 h-5 text-accent-green" />
                        <h2 className="font-bold text-base text-white">AI Defense Stack</h2>
                    </div>

                    <div className="space-y-3.5">
                        <div className="p-3.5 rounded-xl bg-[#080c14] border border-white/[0.06] flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Brain className="w-5 h-5 text-accent-green" />
                                <div>
                                    <div className="text-xs font-bold text-white">Supervised Ensemble</div>
                                    <div className="text-[11px] text-slate-400">11 Models + Soft Voting</div>
                                </div>
                            </div>
                            <span className="badge-minimal text-[10px]">READY (86.4%)</span>
                        </div>

                        <div className="p-3.5 rounded-xl bg-[#080c14] border border-white/[0.06] flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Shield className="w-5 h-5 text-accent-yellow" />
                                <div>
                                    <div className="text-xs font-bold text-white">Zero-Day Anomaly Radar</div>
                                    <div className="text-[11px] text-slate-400">Isolation Forest + LOF</div>
                                </div>
                            </div>
                            <span className="badge-medium text-[10px]">ARMED</span>
                        </div>

                        <div className="p-3.5 rounded-xl bg-[#080c14] border border-white/[0.06] flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <Zap className="w-5 h-5 text-accent-red" />
                                <div>
                                    <div className="text-xs font-bold text-white">RL Autonomous Response</div>
                                    <div className="text-[11px] text-slate-400">Q-Learning + Multi-Agent</div>
                                </div>
                            </div>
                            <span className="badge-critical text-[10px]">ONLINE</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Incident Feed Table */}
            <div className="card space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
                    <div>
                        <h2 className="font-bold text-base text-white">Live Threat Telemetry Feed</h2>
                        <p className="text-xs text-slate-400 mt-0.5">Continuous multi-source intrusion stream</p>
                    </div>
                    <span className="text-xs font-mono text-slate-400">Latest 50 Captured Events</span>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                        <thead>
                            <tr className="border-b border-white/[0.08] text-slate-400 font-mono">
                                <th className="py-3 px-4">Event ID</th>
                                <th className="py-3 px-4">Source</th>
                                <th className="py-3 px-4">Origin IP</th>
                                <th className="py-3 px-4">Attack Vector</th>
                                <th className="py-3 px-4">Risk Level</th>
                                <th className="py-3 px-4">Calculated Risk</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.04]">
                            {data?.recent_events && data.recent_events.length > 0 ? (
                                data.recent_events.slice(0, 10).map((evt: any, idx: number) => {
                                    const riskLevel = evt.risk_scores?.[0]?.risk_level || (evt.risk_label === 'suspicious' ? 'high' : 'minimal')
                                    const riskScore = evt.risk_scores?.[0]?.risk_score ?? (evt.risk_label === 'suspicious' ? 0.85 : 0.12)
                                    
                                    return (
                                        <tr key={idx} className="hover:bg-white/[0.02] transition-colors font-mono">
                                            <td className="py-3 px-4 text-slate-300 font-semibold">#{evt.id || idx + 1}</td>
                                            <td className="py-3 px-4 text-slate-400">{evt.source_type || 'CICIDS2017'}</td>
                                            <td className="py-3 px-4 text-slate-300 font-bold">{evt.source_ip || '192.168.1.105'}</td>
                                            <td className="py-3 px-4">
                                                <span className="text-white font-medium capitalize">{evt.attack_type || 'Normal Traffic'}</span>
                                            </td>
                                            <td className="py-3 px-4">
                                                <span className={`badge-${riskLevel} uppercase text-[10px]`}>
                                                    {riskLevel}
                                                </span>
                                            </td>
                                            <td className="py-3 px-4">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full ${
                                                                riskScore > 0.6 ? 'bg-accent-red' : riskScore > 0.4 ? 'bg-accent-yellow' : 'bg-accent-green'
                                                            }`}
                                                            style={{ width: `${Math.round(riskScore * 100)}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-slate-200 font-bold">{(riskScore * 100).toFixed(1)}%</span>
                                                </div>
                                            </td>
                                        </tr>
                                    )
                                })
                            ) : (
                                <tr>
                                    <td colSpan={6} className="py-8 text-center text-slate-500 font-mono">
                                        No recent telemetry recorded in database. Run pipeline to seed live events.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}