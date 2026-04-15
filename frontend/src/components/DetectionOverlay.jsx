import React from 'react'

/**
 * SVG overlay that draws bounding boxes over a camera feed.
 * boxes: [{cx, cy, w, h, confidence}] all normalized 0-1
 */
export default function DetectionOverlay({ boxes = [], width = 320, height = 240, detected = false }) {
  if (!boxes || boxes.length === 0) return null

  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      {boxes.map((box, i) => {
        const x = (box.cx - box.w / 2) * width
        const y = (box.cy - box.h / 2) * height
        const bw = box.w * width
        const bh = box.h * height
        const conf = Math.round((box.confidence || 0) * 100)

        return (
          <g key={i}>
            {/* Outer glow */}
            <rect
              x={x-2} y={y-2} width={bw+4} height={bh+4}
              fill="none"
              stroke="var(--danger)"
              strokeWidth="1"
              opacity="0.3"
            />
            {/* Main box */}
            <rect
              x={x} y={y} width={bw} height={bh}
              fill="rgba(255,34,68,0.08)"
              stroke="var(--danger)"
              strokeWidth="1.5"
              strokeDasharray="4 2"
              style={{ filter: 'drop-shadow(0 0 6px var(--danger))' }}
            />
            {/* Corner marks */}
            {[
              [x, y, 8, 0, 0, 8],
              [x+bw, y, -8, 0, 0, 8],
              [x, y+bh, 8, 0, 0, -8],
              [x+bw, y+bh, -8, 0, 0, -8],
            ].map(([px, py, dx1, dy1, dx2, dy2], ci) => (
              <g key={ci}>
                <line x1={px} y1={py} x2={px+dx1} y2={py+dy1} stroke="var(--danger)" strokeWidth="2" />
                <line x1={px} y1={py} x2={px+dx2} y2={py+dy2} stroke="var(--danger)" strokeWidth="2" />
              </g>
            ))}
            {/* Label */}
            <rect x={x} y={y-16} width={72} height={16} fill="var(--danger)" rx="1" />
            <text x={x+4} y={y-4}
              fill="white" fontSize="9"
              fontFamily="JetBrains Mono, monospace"
              fontWeight="600"
              letterSpacing="1">
              HUMAN {conf}%
            </text>
          </g>
        )
      })}
    </svg>
  )
}
