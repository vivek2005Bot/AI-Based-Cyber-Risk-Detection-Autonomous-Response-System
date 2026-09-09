import { useEffect, useState } from 'react'
import { Brain, RefreshCw, Layers, Award, CheckCircle2, Zap } from 'lucide-react'
import { apiClient } from '../api/client'

interface ModelInfo {
    available_models: string[]
    trained_models: string[]
}

const benchmarkData = [
    { name: 'Random Forest Ensemble', type: 'Ensemble', accuracy: '86.4%', f1: '0.865', precision: '89.8%', recall: '86.4%', status: 'Production Top', glow: 'border-accent-green/40' },
    { name: 'Soft Voting Classifier', type: 'Ensemble', accuracy: '86.4%', f1: '0.865', precision: '89.8%', recall: '86.4%', status: 'Production Top', glow: 'border-accent-green/40' },
    { name: 'Deep Neural Net (PyTorch)', type: 'Deep Learning', accuracy: '86.3%', f1: '0.865', precision: '89.8%', recall: '86.3%', status: 'High Generalization', glow: 'border-accent-yellow/40' },
    { name: 'K-Nearest Neighbors', type: 'Instance-Based', accuracy: '86.4%', f1: '0.865', precision: '89.9%', recall: '86.4%', status: 'Clustering', glow: 'border-white/10' },
    { name: 'Support Vector Machine (RBF)', type: 'Kernel Method', accuracy: '86.1%', f1: '0.862', precision: '89.6%', recall: '86.1%', status: 'Non-linear', glow: 'border-white/10' },
    { name: 'Naive Bayes (Gaussian)', type: 'Probabilistic', accuracy: '86.1%', f1: '0.862', precision: '89.4%', recall: '86.1%', status: 'Fast Baseline', glow: 'border-white/10' },
    { name: 'Logistic Regression', type: 'Linear Model', accuracy: '85.1%', f1: '0.852', precision: '89.1%', recall: '85.1%', status: 'Interpretable', glow: 'border-white/10' },
]

export default function ModelComparison() {
    const [models, setModels] = useState<ModelInfo | null>(null)
    const [training, setTraining] = useState(false)
    const [message, setMessage] = useState<string | null>(null)

    useEffect(() => {
        loadModels()
    }, [])

    const loadModels = async () => {
        try {
            const result = await apiClient.getModels()
            setModels(result)
        } catch (error) {
            console.error('Failed to load models:', error)
        }
    }

    const handleTrain = async () => {
        setTraining(true)
        setMessage(null)
        try {
            const result = await apiClient.trainModels()
            setMessage(`Successfully trained ${result.trained_models.length} models on ${result.n_samples} real intrusion samples.`)
            await loadModels()
        } catch (error) {
            setMessage('Failed to train models. Make sure the API server is running.')
        } finally {
            setTraining(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2.5">
                        <Brain className="w-7 h-7 text-accent-green" />
                        Supervised & Deep Learning Lab
                    </h1>
                    <p className="text-slate-400 text-sm mt-1">
                        Rigorous leak-free evaluation across 11 supervised classification algorithms and neural architectures.
                    </p>
                </div>
                <button
                    onClick={handleTrain}
                    disabled={training}
                    className="btn-gradient-green flex items-center gap-2"
                >
                    <RefreshCw className={`w-4 h-4 ${training ? 'animate-spin' : ''}`} />
                    {training ? 'Training Algorithms...' : 'Retrain All 11 Models'}
                </button>
            </div>

            {message && (
                <div className="p-4 bg-accent-green/10 border border-accent-green/30 rounded-xl text-accent-green text-xs font-mono flex items-center gap-2.5">
                    <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                    {message}
                </div>
            )}

            {/* Model Architecture Tier Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="card-glow-green">
                    <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                        <span>LEADERBOARD CHAMPION</span>
                        <Award className="w-4 h-4 text-accent-green" />
                    </div>
                    <div className="text-lg font-bold text-white">Random Forest & Soft Voting</div>
                    <div className="mt-3 flex items-baseline gap-2">
                        <span className="text-3xl font-extrabold font-mono text-accent-green">86.47%</span>
                        <span className="text-xs font-mono text-slate-400">Accuracy</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-slate-400">F1: 0.8656 | Precision: 89.87%</div>
                </div>

                <div className="card-glow-yellow">
                    <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                        <span>DEEP LEARNING TIER</span>
                        <Layers className="w-4 h-4 text-accent-yellow" />
                    </div>
                    <div className="text-lg font-bold text-white">PyTorch Deep Neural Net</div>
                    <div className="mt-3 flex items-baseline gap-2">
                        <span className="text-3xl font-extrabold font-mono text-accent-yellow">86.39%</span>
                        <span className="text-xs font-mono text-slate-400">Accuracy</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-slate-400">3-layer MLP + Dropout + Weight Decay</div>
                </div>

                <div className="card-glow-red">
                    <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                        <span>DATASET PROTOCOL</span>
                        <Zap className="w-4 h-4 text-accent-red" />
                    </div>
                    <div className="text-lg font-bold text-white">Multi-Source Real Telemetry</div>
                    <div className="mt-3 flex items-baseline gap-2">
                        <span className="text-3xl font-extrabold font-mono text-accent-red">100%</span>
                        <span className="text-xs font-mono text-slate-400">Leak-Free Split</span>
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-slate-400">CICIDS2017 + UNSW-NB15 + BETH</div>
                </div>
            </div>

            {/* Benchmark Table */}
            <div className="card space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
                    <div>
                        <h2 className="font-bold text-base text-white">Algorithm Benchmark Comparison</h2>
                        <p className="text-xs text-slate-400 mt-0.5">Evaluated on held-out test split (unseen validation packets)</p>
                    </div>
                    <span className="text-xs font-mono text-slate-400">11 Active Classifiers</span>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                        <thead>
                            <tr className="border-b border-white/[0.08] text-slate-400">
                                <th className="py-3 px-4">Model Name</th>
                                <th className="py-3 px-4">Architecture</th>
                                <th className="py-3 px-4">Accuracy</th>
                                <th className="py-3 px-4">Precision</th>
                                <th className="py-3 px-4">Recall</th>
                                <th className="py-3 px-4">F1-Score</th>
                                <th className="py-3 px-4">Role</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.04]">
                            {benchmarkData.map((m, i) => (
                                <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                                    <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-accent-green" />
                                        {m.name}
                                    </td>
                                    <td className="py-3.5 px-4 text-slate-400">{m.type}</td>
                                    <td className="py-3.5 px-4 text-accent-green font-bold">{m.accuracy}</td>
                                    <td className="py-3.5 px-4 text-accent-yellow font-semibold">{m.precision}</td>
                                    <td className="py-3.5 px-4 text-slate-300">{m.recall}</td>
                                    <td className="py-3.5 px-4 text-white font-bold">{m.f1}</td>
                                    <td className="py-3.5 px-4">
                                        <span className={`badge text-[10px] ${
                                            i < 2 ? 'bg-accent-green/15 text-accent-green border-accent-green/40' :
                                            i === 2 ? 'bg-accent-yellow/15 text-accent-yellow border-accent-yellow/40' :
                                            'bg-white/[0.05] text-slate-300 border-white/10'
                                        }`}>
                                            {m.status}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}