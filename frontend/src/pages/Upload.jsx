import React, { useState, useRef, useEffect } from 'react'
import FileUploadPanel from '../components/FileUploadPanel'
import ConfidenceGauge from '../components/ConfidenceGauge'
import ConsistencyBar from '../components/ConsistencyBar'
import useDetectionStore from '../store/detectionStore'

function ResultCard({ result, rgbPreview }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!rgbPreview || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      // Draw bounding boxes
      if (result.detected && result.bounding_boxes?.length) {
        result.bounding_boxes.forEach(box => {
          const x = (box.cx - box.w / 2) * canvas.width
          const y = (box.cy - box.h / 2) * canvas.height
          const bw = box.w * canvas.width
          const bh = box.h * canvas.height
          ctx.strokeStyle = '#ff2244'
          ctx.lineWidth = 2
          ctx.strokeRect(x, y, bw, bh)
          ctx.fillStyle = '#ff2244'
          ctx.fillRect(x, y - 18, 80, 18)
          ctx.fillStyle = 'white'
          ctx.font = '10px JetBrains Mono'
          ctx.fillText(`HUMAN ${Math.round(box.confidence * 100)}%`, x + 4, y - 5)
        })
      }
    }
    img.src = rgbPreview
  }, [result, rgbPreview])

  const detected = result.detected

  return (
    <div className="panel fade-in" style={{ padding: 0, overflow: 'hidden' }}>
      <div className="panel-header" style={{ color: detected ? 'var(--danger)' : 'var(--safe)' }}>
        {detected ? '⚠ SURVIVOR DETECTED' : '◉ NO DETECTION'}
        <span style={{ marginLeft: 'auto', fontSize: 9, color: 'var(--text-dim)' }}>
          {result.frame_id} · {result.inference_ms}ms
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: 16 }}>
        {/* Left: Image with bbox */}
        <div>
          <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', marginBottom: 6 }}>
            RGB + DETECTION OVERLAY
          </div>
          {rgbPreview ? (
            <canvas
              ref={canvasRef}
              style={{ width: '100%', display: 'block', borderRadius: 2, border: '1px solid var(--border)' }}
            />
          ) : (
            <div style={{ height: 150, background: 'var(--surface2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: 'var(--text-dim)' }}>
              NO PREVIEW
            </div>
          )}
        </div>

        {/* Right: Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ConfidenceGauge value={result.confidence} />
          </div>
          <ConsistencyBar value={result.consistency_score} />
          <ConsistencyBar value={result.survival_likelihood} label="SURVIVAL LIKELIHOOD" />
        </div>
      </div>

      {/* Explanation */}
      <div style={{ padding: '8px 16px', borderTop: '1px solid var(--border)', fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.6 }}>
        {result.explanation}
      </div>

      {/* Bounding boxes */}
      {result.bounding_boxes?.length > 0 && (
        <div style={{ padding: '8px 16px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 9, letterSpacing: '2px', color: 'var(--text-dim)', marginBottom: 8 }}>BOUNDING BOXES</div>
          {result.bounding_boxes.map((b, i) => (
            <div key={i} style={{ display: 'flex', gap: 16, fontSize: 10, color: 'var(--text)' }}>
              <span>cx:{b.cx.toFixed(3)}</span>
              <span>cy:{b.cy.toFixed(3)}</span>
              <span>w:{b.w.toFixed(3)}</span>
              <span>h:{b.h.toFixed(3)}</span>
              <span style={{ color: 'var(--danger)' }}>conf:{Math.round(b.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* GPS */}
      {result.gps_location && (
        <div style={{ padding: '8px 16px', borderTop: '1px solid var(--border)', display: 'flex', gap: 24, fontSize: 10 }}>
          <span style={{ color: 'var(--text-dim)' }}>LAT</span>
          <span>{result.gps_location.lat?.toFixed(4)}</span>
          <span style={{ color: 'var(--text-dim)' }}>LON</span>
          <span>{result.gps_location.lon?.toFixed(4)}</span>
          <span style={{ color: 'var(--text-dim)' }}>ALT</span>
          <span>{result.gps_location.altitude?.toFixed(1)}m</span>
        </div>
      )}
    </div>
  )
}

export default function Upload() {
  const [result, setResult] = useState(null)
  const [rgbPreview, setRgbPreview] = useState(null)
  const addDetection = useDetectionStore(s => s.addDetection)

  const handleResult = (res, rgb) => {
    setResult(res)
    setRgbPreview(rgb)
    addDetection(res)
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '600px', gap: 16, padding: 20, maxWidth: 1400, margin: '0 auto', justifyContent: 'center' }}>
      {/* Upload panel */}
      <div>
        <div style={{ marginBottom: 16 }}>
          <h2 style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 20, letterSpacing: '3px', color: 'var(--accent)', marginBottom: 4 }}>
            UPLOAD INFERENCE
          </h2>
          <p style={{ fontSize: 10, color: 'var(--text-dim)', lineHeight: 1.6 }}>
            Upload a paired RGB + thermal image to run the SAR fusion model and detect survivors.
          </p>
        </div>
        <div className="panel" style={{ padding: 16 }}>
          <FileUploadPanel onResult={handleResult} />
        </div>
      </div>

      {/* Result panel */}
      {result && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <h2 style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 20, letterSpacing: '3px', color: result.detected ? 'var(--danger)' : 'var(--safe)' }}>
              INFERENCE RESULT
            </h2>
          </div>
          <ResultCard result={result} rgbPreview={rgbPreview} />
        </div>
      )}
    </div>
  )
}
