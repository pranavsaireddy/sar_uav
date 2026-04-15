import React, { useState, useEffect } from 'react'
import { getHistory, clearHistory } from '../services/api'

export default function History() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')   // all | detected | suppressed
  const [selected, setSelected] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const data = await getHistory(200)
      setRecords(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleClear = async () => {
    if (!window.confirm('Clear all detection history?')) return
    await clearHistory()
    setRecords([])
  }

  const handleExport = () => {
    const headers = ['frame_id', 'timestamp', 'detected', 'confidence', 'consistency_score', 'survival_likelihood', 'explanation']
    const rows = filtered.map(r => headers.map(h => r[h] ?? '').join(','))
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `sar_history_${Date.now()}.csv`
    a.click()
  }

  const filtered = records.filter(r => {
    if (filter === 'detected') return r.detected
    if (filter === 'suppressed') return !r.detected && r.confidence > 0.2
    return true
  })

  return (
    <div style={{ padding: 20, height: '100%', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <h2 style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 20, letterSpacing: '3px', color: 'var(--accent)' }}>
          DETECTION HISTORY
        </h2>
        <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>{filtered.length} RECORDS</span>

        {/* Filter tabs */}
        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
          {[['all','ALL'], ['detected','DETECTIONS'], ['suppressed','FP SUPPRESSED']].map(([v,l]) => (
            <button key={v} onClick={() => setFilter(v)} style={{
              padding: '4px 12px', fontSize: 9, letterSpacing: '1.5px',
              background: filter === v ? 'var(--accent)' : 'transparent',
              color: filter === v ? '#000' : 'var(--text-dim)',
              border: `1px solid ${filter === v ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: 2, cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
            }}>
              {l}
            </button>
          ))}
        </div>

        <button onClick={handleExport} style={{ padding: '4px 12px', fontSize: 9, letterSpacing: '1.5px', background: 'transparent', color: 'var(--safe)', border: '1px solid var(--safe)', borderRadius: 2, cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace' }}>
          EXPORT CSV
        </button>
        <button onClick={handleClear} style={{ padding: '4px 12px', fontSize: 9, letterSpacing: '1.5px', background: 'transparent', color: 'var(--danger)', border: '1px solid var(--danger)', borderRadius: 2, cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace' }}>
          CLEAR
        </button>
        <button onClick={load} style={{ padding: '4px 12px', fontSize: 9, letterSpacing: '1.5px', background: 'transparent', color: 'var(--text-dim)', border: '1px solid var(--border)', borderRadius: 2, cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace' }}>
          REFRESH
        </button>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)', fontSize: 11 }}>LOADING…</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)', fontSize: 11 }}>NO RECORDS FOUND</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>FRAME</th>
                <th>TIMESTAMP</th>
                <th>STATUS</th>
                <th>CONFIDENCE</th>
                <th>CONSISTENCY</th>
                <th>SURVIVAL</th>
                <th>GPS</th>
                <th>LATENCY</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr
                  key={i}
                  onClick={() => setSelected(selected?.frame_id === r.frame_id ? null : r)}
                  style={{ cursor: 'pointer', background: selected?.frame_id === r.frame_id ? 'rgba(0,212,255,0.06)' : undefined }}
                >
                  <td style={{ color: 'var(--accent)', fontWeight: 500 }}>{r.frame_id}</td>
                  <td style={{ color: 'var(--text-dim)' }}>
                    {r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '—'}
                  </td>
                  <td>
                    <span className={`badge ${r.detected ? 'badge-detected' : 'badge-clear'}`}>
                      {r.detected ? 'DETECTED' : 'CLEAR'}
                    </span>
                  </td>
                  <td style={{ color: r.confidence > 0.7 ? 'var(--danger)' : r.confidence > 0.5 ? 'var(--warn)' : 'var(--text)' }}>
                    {Math.round(r.confidence * 100)}%
                  </td>
                  <td style={{ color: r.consistency_score > 0.65 ? 'var(--safe)' : 'var(--warn)' }}>
                    {Math.round(r.consistency_score * 100)}%
                  </td>
                  <td>{Math.round(r.survival_likelihood * 100)}%</td>
                  <td style={{ color: 'var(--text-dim)', fontSize: 11 }}>
                    {r.gps_location ? `${r.gps_location.lat?.toFixed(3)}, ${r.gps_location.lon?.toFixed(3)}` : '—'}
                  </td>
                  <td style={{ color: 'var(--text-dim)' }}>{r.inference_ms?.toFixed(1)}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Expanded row detail */}
        {selected && (
          <div className="panel fade-in" style={{ margin: '12px 0', padding: 16 }}>
            <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--accent)', marginBottom: 8 }}>
              DETAIL — {selected.frame_id}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.8 }}>
              {selected.explanation}
            </div>
            {selected.bounding_boxes?.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 10 }}>
                {selected.bounding_boxes.map((b, i) => (
                  <span key={i} style={{ marginRight: 16, color: 'var(--danger)' }}>
                    Box {i+1}: cx={b.cx?.toFixed(3)} cy={b.cy?.toFixed(3)} w={b.w?.toFixed(3)} h={b.h?.toFixed(3)}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
