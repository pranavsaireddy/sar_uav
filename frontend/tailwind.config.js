/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Rajdhani', 'sans-serif'],
      },
      colors: {
        sar: {
          bg: '#050b14',
          surface: '#0a1628',
          border: '#1a2d4a',
          accent: '#00d4ff',
          heat: '#ff4500',
          warn: '#ffaa00',
          safe: '#00ff88',
          danger: '#ff2244',
          muted: '#4a6080',
        },
      },
    },
  },
  plugins: [],
}
