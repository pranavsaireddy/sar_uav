import React, { useEffect, useRef } from 'react'
import useDetectionStore from '../store/detectionStore'
import { useDetection } from '../hooks/useDetection'
import ConfidenceGauge from '../components/ConfidenceGauge'
import ConsistencyBar from '../components/ConsistencyBar'
import CameraFeed from '../components/CameraFeed'
import DetectionOverlay from '../components/DetectionOverlay'
import MetricCard from '../components/MetricCard'
import GPSMap from '../components/GPSMap'

export default function Dashboard() {
  const {
    latestResult,
    wsConnected,
    totalFrames,
    totalDetections,
    fpSuppressed,
    avgLatency,
    gpsPoints,
  } = useDetectionStore()

  const { startDemoStream, stopDemoStream } = useDetection()

  useEffect(() => {
    // Start demo stream when WS connects but no real UAV
    const t = setTimeout(() => startDemoStream(1.5), 1000)
    return () => { clearTimeout(t); stopDemoStream() }
  }, [startDemoStream, stopDemoStream])

  const result = latestResult
  const detected = result?.detected ?? false
  const confidence = result?.confidence ?? 0
  const consistency = result?.consistency_score ?? 0
  const survival = result?.survival_likelihood ?? 0
  const boxes = result?.bounding_boxes ?? []
  const explanation = result?.explanation ?? 'Awaiting frames…'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gridTemplateRows: 'auto 1fr auto', gap: 1, height: '100%', background: '#000' }}>

      {/* ── Camera feeds ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
        {/* RGB Feed */}
        <div className="panel" style={{ position: 'relative', overflow: 'hidden' }}>
          <div className="panel-header">
            <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`} />
            RGB CAMERA
            <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-dim)' }}>
              {result?.frame_id ?? '—'}
            </span>
          </div>
          <div style={{ position: 'relative' }}>
            <CameraFeed src={null} label="RGB LIVE" width="100%" />
            <DetectionOverlay boxes={boxes} detected={detected} />
          </div>
        </div>

        {/* Thermal Feed */}
        <div className="panel" style={{ overflow: 'hidden' }}>
          <div className="panel-header" style={{ color: 'var(--heat)' }}>
            <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`} style={{ background: 'var(--heat)', boxShadow: '0 0 6px var(--heat)' }} />
            THERMAL IR
            <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-dim)' }}>
              {result?.inference_ms ? `${result.inference_ms.toFixed(1)} ms` : '—'}
            </span>
          </div>
          <CameraFeed src={null} label="IR" isThermal width="100%" />
        </div>
      </div>

      {/* ── Right sidebar ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1, background: '#000' }}>

        {/* Detection status */}
        <div className="panel" style={{ padding: 16 }}>
          <div style={{ textAlign: 'center', marginBottom: 12 }}>
            <span className={`badge ${detected ? 'badge-detected' : 'badge-clear'}`} style={{ fontSize: 12, padding: '4px 16px' }}>
              {detected ? '⚠ SURVIVOR DETECTED' : '◉ AREA CLEAR'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ConfidenceGauge value={confidence} />
          </div>
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <ConsistencyBar value={consistency} label="CROSS-MODAL CONSISTENCY" />
            <ConsistencyBar value={survival} label="SURVIVAL LIKELIHOOD" />
          </div>
        </div>

        {/* Explanation */}
        <div className="panel" style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', marginBottom: 6 }}>SYSTEM ANALYSIS</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: detected ? 'var(--warn)' : 'var(--text-dim)' }}>
            {explanation}
          </div>
        </div>

        {/* Modality weights */}
        {result?.modality_weights && (
          <div className="panel" style={{ padding: '10px 14px' }}>
            <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', marginBottom: 8 }}>MODALITY WEIGHTS</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 9, color: 'var(--accent)', width: 28 }}>RGB</span>
              <div className="bar-track" style={{ flex: 1 }}>
                <div className="bar-fill" style={{ width: `${(result.modality_weights.rgb * 100).toFixed(0)}%`, background: 'var(--accent)' }} />
              </div>
              <span style={{ fontSize: 9, color: 'var(--text-dim)', width: 32 }}>
                {(result.modality_weights.rgb * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <span style={{ fontSize: 9, color: 'var(--heat)', width: 28 }}>IR</span>
              <div className="bar-track" style={{ flex: 1 }}>
                <div className="bar-fill" style={{ width: `${(result.modality_weights.thermal * 100).toFixed(0)}%`, background: 'var(--heat)' }} />
              </div>
              <span style={{ fontSize: 9, color: 'var(--text-dim)', width: 32 }}>
                {(result.modality_weights.thermal * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}

        {/* GPS Map */}
        <div className="panel" style={{ flex: 1, overflow: 'hidden' }}>
          <div className="panel-header" style={{ color: 'var(--safe)' }}>
            GPS SURVIVOR MAP
            <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-dim)' }}>
              {gpsPoints.length} PINS
            </span>
          </div>
          <GPSMap points={gpsPoints} height={200} />
        </div>
      </div>

      {/* ── Bottom metrics bar ── */}
      <div style={{ gridColumn: '1/-1', display: 'flex', gap: 1, background: '#000' }}>
        <MetricCard label="Total Frames" value={totalFrames} color="var(--accent)" />
        <MetricCard label="Detections" value={totalDetections} color="var(--danger)" />
        <MetricCard label="FP Suppressed" value={fpSuppressed} color="var(--warn)" />
        <MetricCard label="Avg Latency" value={avgLatency.toFixed(1)} unit="ms" color="var(--safe)" />
        <MetricCard
          label="Detection Rate"
          value={totalFrames > 0 ? `${Math.round(totalDetections / totalFrames * 100)}` : '0'}
          unit="%"
          color="var(--accent)"
        />
        <div className="panel" style={{ flex: 1, display: 'flex', alignItems: 'center', padding: '0 16px', gap: 8 }}>
          <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`} />
          <span style={{ fontSize: 9, letterSpacing: '2px', color: wsConnected ? 'var(--safe)' : 'var(--danger)' }}>
            {wsConnected ? 'WEBSOCKET LIVE' : 'DISCONNECTED — RECONNECTING'}
          </span>
        </div>
      </div>
    </div>
  )
}
