import React, { useRef, useEffect } from 'react'

/**
 * Renders an image (base64 or URL) on a canvas.
 * For thermal: applies inferno-like colormap.
 */
export default function CameraFeed({ src = null, label = 'RGB', isThermal = false, width = '100%' }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(new Image())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (!src) {
      // Draw placeholder
      ctx.fillStyle = '#0a1628'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#1a2d4a'
      ctx.font = '11px JetBrains Mono'
      ctx.textAlign = 'center'
      ctx.fillText('NO SIGNAL', canvas.width / 2, canvas.height / 2)
      return
    }

    const img = imgRef.current
    img.onload = () => {
      canvas.width = img.width || 320
      canvas.height = img.height || 240

      if (isThermal) {
        // Draw to offscreen, read pixel data, apply inferno colormap
        const offscreen = document.createElement('canvas')
        offscreen.width = canvas.width
        offscreen.height = canvas.height
        const octx = offscreen.getContext('2d')
        octx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const imageData = octx.getImageData(0, 0, canvas.width, canvas.height)
        applyInferno(imageData)
        ctx.putImageData(imageData, 0, 0)
      } else {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      }
    }

    if (src.startsWith('data:') || src.startsWith('blob:')) {
      img.src = src
    } else {
      img.src = `data:image/png;base64,${src}`
    }
  }, [src, isThermal])

  return (
    <div style={{ position: 'relative', background: '#050b14' }}>
      <canvas
        ref={canvasRef}
        width={320} height={240}
        style={{ width, display: 'block', imageRendering: 'pixelated' }}
      />
      <div style={{
        position: 'absolute', top: 6, left: 8,
        fontSize: 9, letterSpacing: '2px',
        color: isThermal ? 'var(--heat)' : 'var(--accent)',
        textShadow: isThermal ? '0 0 8px var(--heat)' : '0 0 8px var(--accent)',
        textTransform: 'uppercase',
      }}>
        {label}
      </div>
    </div>
  )
}

/** Applies an inferno-like colormap to grayscale ImageData in-place */
function applyInferno(imageData) {
  const d = imageData.data
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] / 255  // grayscale value
    const [r, g, b] = infernoColor(v)
    d[i] = r; d[i+1] = g; d[i+2] = b; d[i+3] = 255
  }
}

function infernoColor(t) {
  // Simplified inferno colormap control points
  const stops = [
    [0, [0, 0, 4]],
    [0.13, [31, 12, 72]],
    [0.25, [85, 15, 109]],
    [0.38, [139, 23, 99]],
    [0.5, [188, 55, 84]],
    [0.63, [229, 107, 60]],
    [0.75, [252, 166, 54]],
    [0.88, [254, 224, 139]],
    [1, [252, 255, 164]],
  ]

  let i = 0
  while (i < stops.length - 1 && stops[i+1][0] < t) i++
  if (i >= stops.length - 1) return stops[stops.length-1][1]

  const [t0, c0] = stops[i]
  const [t1, c1] = stops[i+1]
  const alpha = (t - t0) / (t1 - t0)
  return c0.map((v, j) => Math.round(v + alpha * (c1[j] - v)))
}
