import React from 'react'

export default function ConsistencyBar({ value = 0, label = 'CROSS-MODAL CONSISTENCY' }) {
  const pct = Math.round(value * 100)
  let color = 'var(--danger)'
  if (value > 0.65) color = 'var(--safe)'
  else if (value > 0.4) color = 'var(--warn)'

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 9, letterSpacing: '1.5px', color: 'var(--text-dim)' }}>
        <span>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: `0 0 8px ${color}`,
          }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 9, color: 'var(--text-dim)' }}>
        <span>FALSE POS</span>
        <span>TRUE POSITIVE</span>
      </div>
    </div>
  )
}
