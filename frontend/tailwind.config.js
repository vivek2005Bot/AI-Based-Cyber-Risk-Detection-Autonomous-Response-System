/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cyber: {
                    black: '#07090e',
                    dark: '#0d1117',
                    surface: '#131823',
                    card: 'rgba(19, 24, 35, 0.75)',
                    border: 'rgba(255, 255, 255, 0.08)',
                },
                accent: {
                    red: '#ff3366',
                    yellow: '#ffd60a',
                    green: '#00f59b',
                    amber: '#ff9f1c',
                    emerald: '#10b981',
                },
                risk: {
                    minimal: '#00f59b',
                    low: '#a3e635',
                    medium: '#ffd60a',
                    high: '#ff9f1c',
                    critical: '#ff3366',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            boxShadow: {
                'neon-red': '0 0 25px -5px rgba(255, 51, 102, 0.35)',
                'neon-yellow': '0 0 25px -5px rgba(255, 214, 10, 0.35)',
                'neon-green': '0 0 25px -5px rgba(0, 245, 155, 0.35)',
                'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
            },
            keyframes: {
                glow: {
                    '0%': { opacity: 0.6 },
                    '100%': { opacity: 1 },
                }
            }
        },
    },
    plugins: [],
}