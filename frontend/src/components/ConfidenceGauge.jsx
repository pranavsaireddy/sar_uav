import React from 'react'

function getColor(v) {
  if (v >= 0.7) return 'var(--danger)'
  if (v >= 0.5) return 'var(--warn)'
  return 'var(--safe)'
}

export default function ConfidenceGauge({ value = 0, label = 'CONFIDENCE' }) {
  const r = 52
  const cx = 70, cy = 70
  const circumference = Math.PI * r          // half circle
  const stroke = circumference * Math.min(value, 1)
  const color = getColor(value)
  const pct = Math.round(value * 100)

  return (
    <div style={{ textAlign: 'center', userSelect: 'none' }}>
      <svg width="140" height="80" viewBox="0 0 140 80" style={{ overflow: 'visible' }}>
        {/* Track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke="var(--border)" strokeWidth="8" strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${stroke} ${circumference}`}
          style={{
            filter: `drop-shadow(0 0 6px ${color})`,
            transition: 'stroke-dasharray 0.5s ease, stroke 0.3s',
          }}
        />
        {/* Center text */}
        <text x={cx} y={cy - 8} textAnchor="middle"
          fill={color} fontSize="22" fontFamily="Rajdhani,sans-serif" fontWeight="700"
          style={{ filter: `drop-shadow(0 0 8px ${color})` }}>
          {pct}%
        </text>
      </svg>
      <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', marginTop: -8 }}>
        {label}
      </div>
    </div>
  )
}
