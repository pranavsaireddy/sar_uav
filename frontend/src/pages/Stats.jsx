import React, { useState, useEffect } from 'react'
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  PieChart, Pie, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { getStats } from '../services/api'
import useDetectionStore from '../store/detectionStore'
import MetricCard from '../components/MetricCard'

const COLORS = ['var(--accent)', 'var(--heat)', 'var(--safe)', 'var(--warn)']

const darkTooltip = {
  contentStyle: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 2, fontFamily: 'JetBrains Mono', fontSize: 10 },
  labelStyle: { color: 'var(--text-dim)' },
  itemStyle: { color: 'var(--accent)' },
}

export default function Stats() {
  const [serverStats, setServerStats] = useState(null)
  const {
    detectionRateHistory,
    confidenceHistory,
    consistencyHistory,
    latencyHistory,
    totalFrames,
    totalDetections,
    fpSuppressed,
    avgLatency,
  } = useDetectionStore()

  useEffect(() => {
    getStats().then(setServerStats).catch(() => {})
    const t = setInterval(() => getStats().then(setServerStats).catch(() => {}), 5000)
    return () => clearInterval(t)
  }, [])

  // Confidence distribution histogram
  const confBuckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i*10}-${(i+1)*10}%`,
    count: confidenceHistory.filter(c => c.value >= i/10 && c.value < (i+1)/10).length,
  }))

  // Scatter: consistency vs confidence
  const scatterData = confidenceHistory.map((c, i) => ({
    confidence: Math.round(c.value * 100),
    consistency: Math.round((consistencyHistory[i]?.value ?? 0) * 100),
  }))

  // Modality pie
  const modalityData = serverStats ? [
    { name: 'RGB', value: 52 },
    { name: 'Thermal', value: 48 },
  ] : []

  // Latency histogram
  const latBuckets = Array.from({ length: 8 }, (_, i) => {
    const lo = i * 5, hi = (i+1) * 5
    return {
      range: `${lo}-${hi}ms`,
      count: latencyHistory.filter(v => v >= lo && v < hi).length,
    }
  })

  return (
    <div style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16, height: '100%', overflow: 'auto' }}>
      <h2 style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 20, letterSpacing: '3px', color: 'var(--accent)' }}>
        MISSION STATISTICS
      </h2>

      {/* Top metrics */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <MetricCard label="Total Frames" value={totalFrames} color="var(--accent)" />
        <MetricCard label="Detections" value={totalDetections} color="var(--danger)" />
        <MetricCard label="FP Suppressed" value={fpSuppressed} color="var(--warn)" />
        <MetricCard label="Avg Latency" value={avgLatency.toFixed(1)} unit="ms" color="var(--safe)" />
        {serverStats && (
          <>
            <MetricCard label="Det / Min" value={serverStats.detections_per_minute} color="var(--accent)" />
            <MetricCard label="Avg Confidence" value={`${Math.round(serverStats.avg_confidence * 100)}%`} color="var(--warn)" />
            <MetricCard label="Uptime" value={`${Math.round(serverStats.uptime_seconds / 60)}m`} color="var(--text-dim)" />
          </>
        )}
      </div>

      {/* Charts grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Detection rate over time */}
        <div className="panel">
          <div className="panel-header">DETECTION RATE OVER TIME</div>
          <div style={{ padding: '12px 8px' }}>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={detectionRateHistory}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fill: 'var(--text-dim)', fontSize: 9 }} />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-dim)', fontSize: 9 }} unit="%" />
                <Tooltip {...darkTooltip} />
                <Line type="monotone" dataKey="rate" stroke="var(--accent)" strokeWidth={1.5} dot={false} unit="%" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confidence distribution */}
        <div className="panel">
          <div className="panel-header">CONFIDENCE DISTRIBUTION</div>
          <div style={{ padding: '12px 8px' }}>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={confBuckets}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fill: 'var(--text-dim)', fontSize: 8 }} />
                <YAxis tick={{ fill: 'var(--text-dim)', fontSize: 9 }} />
                <Tooltip {...darkTooltip} />
                <Bar dataKey="count" fill="var(--accent)" opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Consistency vs Confidence scatter */}
        <div className="panel">
          <div className="panel-header">CONSISTENCY VS CONFIDENCE</div>
          <div style={{ padding: '12px 8px' }}>
            <ResponsiveContainer width="100%" height={160}>
              <ScatterChart>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="confidence" name="Confidence" unit="%" tick={{ fill: 'var(--text-dim)', fontSize: 9 }} />
                <YAxis dataKey="consistency" name="Consistency" unit="%" tick={{ fill: 'var(--text-dim)', fontSize: 9 }} />
                <Tooltip {...darkTooltip} cursor={{ stroke: 'var(--border)' }} />
                <Scatter data={scatterData} fill="var(--accent)" opacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency histogram */}
        <div className="panel">
          <div className="panel-header">INFERENCE LATENCY DISTRIBUTION</div>
          <div style={{ padding: '12px 8px' }}>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={latBuckets}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fill: 'var(--text-dim)', fontSize: 8 }} />
                <YAxis tick={{ fill: 'var(--text-dim)', fontSize: 9 }} />
                <Tooltip {...darkTooltip} />
                <Bar dataKey="count" fill="var(--safe)" opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Model info + modality pie */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px', gap: 16 }}>
        <div className="panel" style={{ padding: 16 }}>
          <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--accent)', marginBottom: 12 }}>MODEL INFORMATION</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {[
              ['Architecture', 'SARFusionModel'],
              ['Backbone', 'EfficientNet-B0 x2'],
              ['Fusion Layers', '3 CrossModalAttention'],
              ['Parameters', '~8M (full)'],
              ['Input Size', '320 × 320 px'],
              ['Dataset', 'LLVIP (30k pairs)'],
              ['Confidence Thresh', '0.45'],
              ['Device', serverStats?.device ?? '—'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '1px' }}>{k}</span>
                <span style={{ fontSize: 11, color: 'var(--text)' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">MODALITY CONTRIBUTION</div>
          <div style={{ padding: 12, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <ResponsiveContainer width="100%" height={120}>
              <PieChart>
                <Pie data={modalityData.length ? modalityData : [{name:'RGB',value:52},{name:'IR',value:48}]}
                  cx="50%" cy="50%" innerRadius={35} outerRadius={55}
                  dataKey="value" paddingAngle={3}>
                  <Cell fill="var(--accent)" />
                  <Cell fill="var(--heat)" />
                </Pie>
                <Tooltip {...darkTooltip} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 16, fontSize: 9 }}>
              <span style={{ color: 'var(--accent)' }}>■ RGB</span>
              <span style={{ color: 'var(--heat)' }}>■ THERMAL</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
