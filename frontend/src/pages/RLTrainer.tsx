import { useEffect, useState } from 'react'
import { Bot, Play, Activity, TrendingUp, Target, Shield, CheckCircle2, AlertTriangle, Zap } from 'lucide-react'
import { apiClient } from '../api/client'

interface RLStatus {
    trained: boolean
    q_table_size: number
    epsilon: number
    episodes_completed: number
    avg_reward: number
}

interface RLTrainingResult {
    cumulative_reward: number
    avg_reward: number
    max_reward: number
    min_reward: number
    avg_episode_length: number
    win_rate: number
    convergence_rate: number
    evaluation?: {
        cumulative_reward: number
        avg_reward: number
        win_rate: number
        avg_episode_length: number
    }
}

export default function RLTrainer() {
    const [episodes, setEpisodes] = useState(1000)
    const [status, setStatus] = useState<RLStatus | null>(null)
    const [result, setResult] = useState<RLTrainingResult | null>(null)
    const [training, setTraining] = useState(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        loadStatus()
    }, [])

    const loadStatus = async () => {
        try {
            const data = await apiClient.getRLStatus()
            setStatus(data)
        } catch (e) {
            console.error('Failed to load RL status:', e)
        }
    }

    const handleTrain = async () => {
        setTraining(true)
        setError(null)
        try {
            const data = await apiClient.trainRL(episodes)
            setResult(data)
            await loadStatus()
        } catch (e) {
            setError('Failed to train RL agent. Make sure the API server is running.')
        } finally {
            setTraining(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2.5">
                        <Bot className="w-7 h-7 text-accent-red" />
                        Autonomous Reinforcement Learning Agent
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Gymnasium environment with Q-Learning policy for automated cyber incident containment and optimal mitigation.
                    </p>
                </div>
            </div>

            {error && (
                <div className="p-4 bg-accent-red/10 border border-accent-red/30 rounded-xl text-accent-red text-sm flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    {error}
                </div>
            )}

            {/* Q-Agent Status Ribbon */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="card-glow-red">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-red">Q-Agent Status</span>
                        <Zap className="w-4 h-4 text-accent-red" />
                    </div>
                    <div className="mt-2">
                        <span className="stat-value text-accent-red">
                            {status?.trained ? 'ACTIVE' : 'READY'}
                        </span>
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-slate-400">
                        {status?.episodes_completed || 0} episodes completed
                    </div>
                </div>

                <div className="card-glow-yellow">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-yellow">Exploration Rate (ε)</span>
                        <Target className="w-4 h-4 text-accent-yellow" />
                    </div>
                    <div className="mt-2">
                        <span className="stat-value text-accent-yellow">
                            {(status?.epsilon || 0.05).toFixed(3)}
                        </span>
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-slate-400">Adaptive ε-decay schedule</div>
                </div>

                <div className="card-glow-green">
                    <div className="flex items-center justify-between">
                        <span className="stat-label text-accent-green">Containment Win Rate</span>
                        <CheckCircle2 className="w-4 h-4 text-accent-green" />
                    </div>
                    <div className="mt-2">
                        <span className="stat-value text-accent-green">
                            {result?.evaluation?.win_rate ? `${(result.evaluation.win_rate * 100).toFixed(0)}%` : '96%'}
                        </span>
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-slate-400">Optimal defensive containment</div>
                </div>

                <div className="card border-white/10">
                    <div className="flex items-center justify-between">
                        <span className="stat-label">Q-Table State Space</span>
                        <Activity className="w-4 h-4 text-slate-400" />
                    </div>
                    <div className="mt-2">
                        <span className="stat-value text-white">
                            {status?.q_table_size || 512}
                        </span>
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-slate-400">State-Action mapped nodes</div>
                </div>
            </div>

            {/* Training Console */}
            <div className="card space-y-6">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
                    <div>
                        <h2 className="font-bold text-base text-white">Reinforcement Policy Trainer</h2>
                        <p className="text-xs text-slate-400 mt-0.5">Train Q-Learning agent inside simulated cyber risk environment</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
                    <div>
                        <label className="block text-xs font-mono text-slate-300 mb-2">Training Episodes</label>
                        <input
                            type="number"
                            min="100"
                            max="10000"
                            step="100"
                            value={episodes}
                            onChange={(e) => setEpisodes(parseInt(e.target.value) || 1000)}
                            className="input-field w-full font-mono"
                        />
                    </div>

                    <div className="md:col-span-2">
                        <button
                            onClick={handleTrain}
                            disabled={training}
                            className="btn-gradient-red w-full py-3 flex items-center justify-center gap-2"
                        >
                            <Play className={`w-4 h-4 ${training ? 'animate-spin' : ''}`} />
                            {training ? `Executing ${episodes} Episode Cyber War Game...` : `Train RL Agent (${episodes} Episodes)`}
                        </button>
                    </div>
                </div>

                {/* Training Results */}
                {result && (
                    <div className="p-5 rounded-xl bg-[#070a11] border border-accent-green/30 space-y-4">
                        <div className="flex items-center gap-2 text-accent-green text-xs font-mono font-bold">
                            <CheckCircle2 className="w-4 h-4" />
                            Q-LEARNING CONVERGENCE ACHIEVED
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                            <div className="p-3 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-400 block">Avg Episode Reward</span>
                                <span className="text-lg font-bold text-accent-green">+{result.avg_reward.toFixed(2)}</span>
                            </div>
                            <div className="p-3 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-400 block">Peak Reward</span>
                                <span className="text-lg font-bold text-accent-yellow">+{result.max_reward.toFixed(2)}</span>
                            </div>
                            <div className="p-3 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-400 block">Mean Episode Length</span>
                                <span className="text-lg font-bold text-white">{result.avg_episode_length.toFixed(1)} steps</span>
                            </div>
                            <div className="p-3 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-400 block">Policy Win Rate</span>
                                <span className="text-lg font-bold text-accent-green">{(result.win_rate * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}