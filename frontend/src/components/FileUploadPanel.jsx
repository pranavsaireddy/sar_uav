import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function DropZone({ label, accept, onFile, preview, color }) {
  const onDrop = useCallback((accepted) => {
    if (accepted[0]) onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  })

  return (
    <div
      {...getRootProps()}
      style={{
        border: `1px dashed ${isDragActive ? color : 'var(--border)'}`,
        borderRadius: 2,
        padding: 20,
        textAlign: 'center',
        cursor: 'pointer',
        background: isDragActive ? `${color}10` : 'var(--surface)',
        transition: 'all 0.2s',
        position: 'relative',
        minHeight: 140,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
      }}
    >
      <input {...getInputProps()} />

      {preview ? (
        <>
          <img
            src={preview}
            alt={label}
            style={{
              maxHeight: 100,
              maxWidth: '100%',
              objectFit: 'contain',
              filter: label === 'THERMAL' ? 'grayscale(1) contrast(1.2)' : 'none',
            }}
          />
          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '1px' }}>
            {label} LOADED ✓
          </div>
        </>
      ) : (
        <>
          <div style={{ fontSize: 28, opacity: 0.3 }}>
            {label === 'RGB' ? '📷' : '🌡️'}
          </div>
          <div style={{ fontSize: 10, letterSpacing: '2px', color, textTransform: 'uppercase' }}>
            {label}
          </div>
          <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>
            {isDragActive ? 'DROP FILE HERE' : 'DRAG & DROP OR CLICK'}
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)' }}>JPG / PNG — max 10 MB</div>
        </>
      )}
    </div>
  )
}

export default function FileUploadPanel({ onResult }) {
  const [rgbFile, setRgbFile] = useState(null)
  const [thermalFile, setThermalFile] = useState(null)
  const [rgbPreview, setRgbPreview] = useState(null)
  const [thermalPreview, setThermalPreview] = useState(null)
  const [gps, setGps] = useState({ lat: '', lon: '', altitude: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleRgb = (file) => {
    setRgbFile(file)
    setRgbPreview(URL.createObjectURL(file))
  }

  const handleThermal = (file) => {
    setThermalFile(file)
    setThermalPreview(URL.createObjectURL(file))
  }

  const handleSubmit = async () => {
    if (!rgbFile || !thermalFile) {
      setError('Please provide both RGB and thermal images')
      return
    }
    setLoading(true)
    setError(null)

    try {
      const { uploadFiles } = await import('../services/api.js')
      const gpsData = gps.lat && gps.lon
        ? { lat: parseFloat(gps.lat), lon: parseFloat(gps.lon), altitude: parseFloat(gps.altitude) || 0 }
        : null
      const result = await uploadFiles(rgbFile, thermalFile, gpsData)
      onResult && onResult(result, rgbPreview, thermalPreview)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setRgbFile(null); setThermalFile(null)
    setRgbPreview(null); setThermalPreview(null)
    setError(null)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <DropZone
          label="RGB"
          accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
          onFile={handleRgb}
          preview={rgbPreview}
          color="var(--accent)"
        />
        <DropZone
          label="THERMAL"
          accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
          onFile={handleThermal}
          preview={thermalPreview}
          color="var(--heat)"
        />
      </div>

      {/* GPS Inputs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {['lat', 'lon', 'altitude'].map(field => (
          <div key={field}>
            <label style={{ fontSize: 9, letterSpacing: '1.5px', color: 'var(--text-dim)', textTransform: 'uppercase', display: 'block', marginBottom: 4 }}>
              {field}
            </label>
            <input
              type="number"
              placeholder={field === 'lat' ? '17.385' : field === 'lon' ? '78.487' : '50'}
              value={gps[field]}
              onChange={e => setGps(g => ({ ...g, [field]: e.target.value }))}
              style={{
                width: '100%', padding: '6px 8px',
                background: 'var(--surface2)', border: '1px solid var(--border)',
                color: 'var(--text)', borderRadius: 2, fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace', outline: 'none',
              }}
            />
          </div>
        ))}
      </div>

      {error && (
        <div style={{ padding: '8px 12px', background: 'rgba(255,34,68,0.1)', border: '1px solid rgba(255,34,68,0.3)', borderRadius: 2, fontSize: 11, color: 'var(--danger)' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleSubmit}
          disabled={loading || !rgbFile || !thermalFile}
          style={{
            flex: 1, padding: '10px 0',
            background: loading ? 'var(--border)' : 'var(--accent)',
            color: loading ? 'var(--text-dim)' : '#000',
            border: 'none', borderRadius: 2, cursor: loading ? 'wait' : 'pointer',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 11, letterSpacing: '2px', fontWeight: 600,
            textTransform: 'uppercase',
            transition: 'all 0.2s',
          }}
        >
          {loading ? 'ANALYZING...' : 'RUN INFERENCE'}
        </button>
        <button
          onClick={handleClear}
          style={{
            padding: '10px 16px',
            background: 'transparent', color: 'var(--text-dim)',
            border: '1px solid var(--border)', borderRadius: 2, cursor: 'pointer',
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '1px',
          }}
        >
          CLEAR
        </button>
      </div>
    </div>
  )
}
