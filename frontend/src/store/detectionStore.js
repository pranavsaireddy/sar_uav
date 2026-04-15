import { create } from 'zustand'

const useDetectionStore = create((set, get) => ({
  // State
  detections: [],
  latestResult: null,
  wsConnected: false,
  totalFrames: 0,
  totalDetections: 0,
  fpSuppressed: 0,
  avgLatency: 0,
  latencyHistory: [],
  confidenceHistory: [],
  consistencyHistory: [],
  detectionRateHistory: [],   // [{time, rate}]
  gpsPoints: [],
  modelInfo: null,

  // Actions
  addDetection: (result) => set((state) => {
    const newDetections = [result, ...state.detections].slice(0, 1000)
    const newLatency = [...state.latencyHistory, result.inference_ms].slice(-60)
    const avgLat = newLatency.reduce((a, b) => a + b, 0) / newLatency.length

    const newConf = [...state.confidenceHistory, {
      time: new Date(result.timestamp).toLocaleTimeString(),
      value: result.confidence,
    }].slice(-60)

    const newCons = [...state.consistencyHistory, {
      time: new Date(result.timestamp).toLocaleTimeString(),
      value: result.consistency_score,
    }].slice(-60)

    // Detection rate over last 10
    const recentN = newDetections.slice(0, 10)
    const rate = recentN.filter(d => d.detected).length / Math.max(1, recentN.length)
    const newRate = [...state.detectionRateHistory, {
      time: new Date(result.timestamp).toLocaleTimeString(),
      rate: Math.round(rate * 100),
    }].slice(-60)

    // GPS
    const newGps = result.detected && result.gps_location
      ? [...state.gpsPoints, { ...result.gps_location, frame_id: result.frame_id, confidence: result.confidence, timestamp: result.timestamp }]
      : state.gpsPoints

    return {
      detections: newDetections,
      latestResult: result,
      totalFrames: state.totalFrames + 1,
      totalDetections: result.detected ? state.totalDetections + 1 : state.totalDetections,
      fpSuppressed: !result.detected && result.confidence > 0.2 ? state.fpSuppressed + 1 : state.fpSuppressed,
      avgLatency: avgLat,
      latencyHistory: newLatency,
      confidenceHistory: newConf,
      consistencyHistory: newCons,
      detectionRateHistory: newRate,
      gpsPoints: newGps.slice(-200),
    }
  }),

  setWsConnected: (v) => set({ wsConnected: v }),
  setModelInfo: (info) => set({ modelInfo: info }),
  clearHistory: () => set({
    detections: [],
    latestResult: null,
    totalFrames: 0,
    totalDetections: 0,
    fpSuppressed: 0,
    latencyHistory: [],
    confidenceHistory: [],
    consistencyHistory: [],
    detectionRateHistory: [],
    gpsPoints: [],
  }),
}))

export default useDetectionStore
