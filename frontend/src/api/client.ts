import axios from 'axios'

const api = axios.create({
    baseURL: 'http://localhost:8000',
    timeout: 30000,
})

export interface RiskAssessmentRequest {
    features: Record<string, number | string | boolean>
    source_type?: string
    event_type?: string
    user_id?: string
    host_id?: string
    source_ip?: string
    cvss_score?: number
    asset_criticality?: number
    threat_score?: number
    incident_count?: number
    hours_since_event?: number
}

export interface RiskAssessmentResponse {
    risk_score: number
    risk_level: string
    supervised_score: number
    anomaly_score: number
    contextual_score: number
    explanation: {
        risk_score: number
        risk_level: string
        contributions: Record<string, { score: number; weight: number; contribution: number }>
        dominant_factor: string
        explanation: string
    }
    recommendations: Array<{
        action: string
        description: string
        score: number
        similarity?: number
    }>
}

export interface DashboardData {
    total_events: number
    risk_distribution: Record<string, number>
    recent_events: Array<{
        event_id: number
        timestamp: string
        source_type: string
        event_type: string
        risk_label: string
        risk_score: number | null
        risk_level: string | null
    }>
    models_loaded: number
    rl_trained: boolean
}

export const apiClient = {
    async getHealth() {
        const { data } = await api.get('/health')
        return data
    },

    async getDashboard(): Promise<DashboardData> {
        const { data } = await api.get('/dashboard')
        return data
    },

    async assessRisk(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse> {
        const { data } = await api.post('/assess-risk', request)
        return data
    },

    async trainModels() {
        const { data } = await api.post('/models/train')
        return data
    },

    async getModels() {
        const { data } = await api.get('/models')
        return data
    },

    async trainRL(episodes: number = 1000) {
        const { data } = await api.post('/rl/train', { episodes, evaluate: true })
        return data
    },

    async getRLStatus() {
        const { data } = await api.get('/rl/status')
        return data
    },

    async getRLPolicy() {
        const { data } = await api.get('/rl/policy')
        return data
    },

    async getRiskDistribution() {
        const { data } = await api.get('/risk-distribution')
        return data
    },

    async getEvents(limit: number = 50) {
        const { data } = await api.get('/events', { params: { limit } })
        return data
    },

    async detectAnomalies(features: Record<string, number>) {
        const { data } = await api.post('/detect-anomalies', features)
        return data
    },

    async auditEthics() {
        const { data } = await api.get('/ethics/audit')
        return data
    },

    async runMultiAgent(state?: number[]) {
        const { data } = await api.post('/rl/multi-agent', state)
        return data
    },

    async seedDatabase(count: number = 50) {
        const { data } = await api.post('/database/seed', null, { params: { count } })
        return data
    },
}

export default apiClient