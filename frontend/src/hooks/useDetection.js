import { useCallback, useRef } from 'react'
import { useWebSocket } from './useWebSocket'

/**
 * Generates and sends synthetic demo frames via WebSocket
 * when no real UAV is connected.
 */
export function useDetection() {
  const { send } = useWebSocket()
  const intervalRef = useRef(null)

  const startDemoStream = useCallback((fps = 2) => {
    if (intervalRef.current) return

    intervalRef.current = setInterval(() => {
      // Generate synthetic 1x1 pixel base64 images as placeholders
      // In production this would be real UAV camera frames
      const canvas = document.createElement('canvas')
      canvas.width = 4; canvas.height = 4
      const ctx = canvas.getContext('2d')

      // RGB: random debris color
      ctx.fillStyle = `rgb(${80+Math.random()*80|0},${60+Math.random()*60|0},${40+Math.random()*40|0})`
      ctx.fillRect(0,0,4,4)
      const rgbB64 = canvas.toDataURL('image/png').split(',')[1]

      // Thermal: grayscale
      const v = 30 + Math.random() * 180 | 0
      ctx.fillStyle = `rgb(${v},${v},${v})`
      ctx.fillRect(0,0,4,4)
      const thmB64 = canvas.toDataURL('image/png').split(',')[1]

      send({
        rgb_b64: rgbB64,
        thermal_b64: thmB64,
        gps: {
          lat: 17.385 + (Math.random() - 0.5) * 0.01,
          lon: 78.487 + (Math.random() - 0.5) * 0.01,
          altitude: 45 + Math.random() * 15,
        },
      })
    }, 1000 / fps)
  }, [send])

  const stopDemoStream = useCallback(() => {
    clearInterval(intervalRef.current)
    intervalRef.current = null
  }, [])

  return { startDemoStream, stopDemoStream }
}
