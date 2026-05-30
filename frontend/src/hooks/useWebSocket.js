import { useEffect, useRef, useCallback } from 'react'
import useDetectionStore from '../store/detectionStore'

const WS_URL = `ws://${window.location.hostname}:8000/ws/live`

export function useWebSocket() {
  const ws = useRef(null)
  const retryTimeout = useRef(null)
  const retryDelay = useRef(1000)
  const mounted = useRef(true)

  const { setWsConnected, addDetection } = useDetectionStore()

  const connect = useCallback(() => {
    if (!mounted.current) return

    try {
      ws.current = new WebSocket(WS_URL)

      ws.current.onopen = () => {
        if (!mounted.current) return
        setWsConnected(true)
        retryDelay.current = 1000
        console.log('[WS] Connected')
      }

      ws.current.onmessage = (evt) => {
        if (!mounted.current) return
        try {
          const data = JSON.parse(evt.data)
          if (data.type !== 'ping') {
            addDetection(data)
          }
        } catch (e) {
          console.warn('[WS] Parse error', e)
        }
      }

      ws.current.onclose = () => {
        if (!mounted.current) return
        setWsConnected(false)
        console.log(`[WS] Disconnected, retry in ${retryDelay.current}ms`)
        retryTimeout.current = setTimeout(() => {
          retryDelay.current = Math.min(retryDelay.current * 2, 30000)
          connect()
        }, retryDelay.current)
      }

      ws.current.onerror = () => {
        ws.current?.close()
      }
    } catch (e) {
      console.warn('[WS] Connection error', e)
      retryTimeout.current = setTimeout(connect, retryDelay.current)
    }
  }, [setWsConnected, addDetection])

  useEffect(() => {
    mounted.current = true
    connect()
    return () => {
      mounted.current = false
      clearTimeout(retryTimeout.current)
      ws.current?.close()
    }
  }, [connect])

  const send = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    }
  }, [])

  return { send }
}
