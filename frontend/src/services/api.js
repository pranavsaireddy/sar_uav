import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export const detectBase64 = (rgbB64, thermalB64, gps = null) =>
  api.post('/detect', { rgb_b64: rgbB64, thermal_b64: thermalB64, gps }).then(r => r.data)

export const uploadFiles = (rgbFile, thermalFile, gps = null) => {
  const form = new FormData()
  form.append('rgb_file', rgbFile)
  form.append('thermal_file', thermalFile)
  if (gps?.lat != null) form.append('lat', gps.lat)
  if (gps?.lon != null) form.append('lon', gps.lon)
  if (gps?.altitude != null) form.append('altitude', gps.altitude)
  return api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const getStats = () => api.get('/stats').then(r => r.data)
export const getHistory = (limit = 100) => api.get(`/history?limit=${limit}`).then(r => r.data)
export const clearHistory = () => api.delete('/history').then(r => r.data)
export const getGpsPoints = () => api.get('/detections/map').then(r => r.data)
export const getHealth = () => api.get('/health').then(r => r.data)

export default api
