import React from 'react'

export default function MetricCard({ label, value, unit = '', color = 'var(--accent)', sub = '' }) {
  return (
    <div className="panel" style={{ padding: '12px 16px', minWidth: 100 }}>
      <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
        <span
          className="metric-value"
          style={{ color, textShadow: `0 0 16px ${color}` }}
        >
          {value}
        </span>
        {unit && (
          <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>{unit}</span>
        )}
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>{sub}</div>
      )}
    </div>
  )
}
