import React, { useEffect, useRef } from 'react'

/**
 * Leaflet map showing survivor GPS pins.
 * Uses dynamic import to avoid SSR issues.
 */
export default function GPSMap({ points = [], height = 280 }) {
  const mapRef = useRef(null)
  const mapInstance = useRef(null)
  const markersRef = useRef([])

  useEffect(() => {
    if (mapInstance.current) return

    import('leaflet').then((L) => {
      if (!mapRef.current) return

      const map = L.map(mapRef.current, {
        center: [17.385, 78.487],
        zoom: 14,
        zoomControl: true,
        attributionControl: false,
      })

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OSM',
      }).addTo(map)

      mapInstance.current = { map, L }
    })

    return () => {
      if (mapInstance.current) {
        mapInstance.current.map.remove()
        mapInstance.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!mapInstance.current) return
    const { map, L } = mapInstance.current

    // Clear old markers
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []

    points.forEach((pt) => {
      const conf = Math.round(pt.confidence * 100)
      const color = conf >= 70 ? '#ff2244' : conf >= 50 ? '#ffaa00' : '#00ff88'

      const icon = L.divIcon({
        html: `<div style="
          width:14px;height:14px;border-radius:50%;
          background:${color};
          box-shadow:0 0 10px ${color};
          border:2px solid white;
          cursor:pointer;
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
        className: '',
      })

      const marker = L.marker([pt.lat, pt.lon], { icon })
        .bindPopup(`
          <div style="font-family:monospace;font-size:11px;color:#333;min-width:160px">
            <b style="color:#ff2244">SURVIVOR ${pt.frame_id}</b><br>
            Confidence: ${conf}%<br>
            Alt: ${pt.altitude?.toFixed(1)}m<br>
            ${new Date(pt.timestamp).toLocaleTimeString()}
          </div>
        `)
        .addTo(map)

      markersRef.current.push(marker)
    })

    if (points.length > 0) {
      const last = points[points.length - 1]
      map.setView([last.lat, last.lon], 15, { animate: true })
    }
  }, [points])

  return (
    <div
      ref={mapRef}
      style={{ height, width: '100%', background: '#050b14' }}
    />
  )
}
