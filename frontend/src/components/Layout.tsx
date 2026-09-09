import { Outlet, NavLink } from 'react-router-dom'
import { Shield, LayoutDashboard, Activity, Brain, Radar, Bot, ThumbsUp, Scale, Zap } from 'lucide-react'

const navItems = [
    { path: '/', label: 'Executive Threat Ops', icon: LayoutDashboard, glow: 'hover:text-accent-red' },
    { path: '/risk', label: 'Multimodal Risk Fusion', icon: Activity, glow: 'hover:text-accent-yellow' },
    { path: '/models', label: 'Supervised & DNN Lab', icon: Brain, glow: 'hover:text-accent-green' },
    { path: '/anomalies', label: 'Zero-Day Anomaly Radar', icon: Radar, glow: 'hover:text-accent-yellow' },
    { path: '/rl', label: 'Autonomous RL Agent', icon: Bot, glow: 'hover:text-accent-red' },
    { path: '/recommendations', label: 'Automated Remediation', icon: ThumbsUp, glow: 'hover:text-accent-green' },
    { path: '/ethics', label: 'Ethics & Defense Audit', icon: Scale, glow: 'hover:text-accent-yellow' },
]

export default function Layout() {
    return (
        <div className="min-h-screen flex text-slate-100 font-sans">
            {/* Cyber Sidebar */}
            <aside className="w-72 bg-[#090d16]/95 backdrop-blur-xl border-r border-white/[0.08] flex flex-col relative z-20">
                {/* Header Logo */}
                <div className="p-6 border-b border-white/[0.08]">
                    <div className="flex items-center gap-3.5">
                        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-accent-red via-accent-yellow to-accent-green p-[1.5px] shadow-neon-red flex items-center justify-center">
                            <div className="w-full h-full bg-[#0d111b] rounded-[10px] flex items-center justify-center">
                                <Shield className="w-6 h-6 text-accent-red animate-pulse-slow" />
                            </div>
                        </div>
                        <div>
                            <div className="flex items-center gap-1.5">
                                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-accent-red via-accent-yellow to-accent-green bg-clip-text text-transparent">
                                    CYBERSHIELD
                                </span>
                                <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-accent-red/20 text-accent-red border border-accent-red/30">
                                    AI
                                </span>
                            </div>
                            <p className="text-[11px] font-mono text-slate-400">SOC Autonomous Defense</p>
                        </div>
                    </div>
                </div>

                {/* Navigation Links */}
                <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
                    <div className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-slate-500 font-semibold">
                        Defense Modules
                    </div>
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            end={item.path === '/'}
                            className={({ isActive }) =>
                                `group flex items-center justify-between px-3.5 py-3 rounded-xl text-xs font-semibold tracking-wide transition-all duration-300 ${isActive
                                    ? 'bg-gradient-to-r from-accent-red/20 via-accent-yellow/10 to-transparent text-white border-l-4 border-accent-red shadow-lg shadow-black/40 font-bold'
                                    : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'
                                }`
                            }
                        >
                            <div className="flex items-center gap-3">
                                <item.icon className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" />
                                <span>{item.label}</span>
                            </div>
                            <div className="w-1.5 h-1.5 rounded-full bg-slate-700 group-hover:bg-accent-yellow transition-colors" />
                        </NavLink>
                    ))}
                </nav>

                {/* Threat Telemetry System Status Indicator */}
                <div className="p-4 m-3 rounded-xl bg-[#0e1422] border border-white/[0.06] space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
                            <Zap className="w-3.5 h-3.5 text-accent-yellow animate-pulse" />
                            Engine Status
                        </span>
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-accent-green/10 text-accent-green border border-accent-green/30">
                            <span className="w-1.5 h-1.5 rounded-full bg-accent-green animate-ping" />
                            ACTIVE
                        </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-accent-green via-accent-yellow to-accent-red w-full" />
                    </div>
                    <div className="flex justify-between text-[10px] font-mono text-slate-500">
                        <span>CICIDS+UNSW+BETH</span>
                        <span>Multi-Modal</span>
                    </div>
                </div>
            </aside>

            {/* Main view container */}
            <main className="flex-1 overflow-y-auto">
                <div className="p-8 max-w-7xl mx-auto space-y-6">
                    <Outlet />
                </div>
            </main>
        </div>
    )
}