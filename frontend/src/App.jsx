import React from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import History from './pages/History'
import Stats from './pages/Stats'
import useDetectionStore from './store/detectionStore'
import { useWebSocket } from './hooks/useWebSocket'

function NavBar() {
  const { wsConnected, totalDetections } = useDetectionStore()

  // Mount WS at root so it's always active
  useWebSocket()

  return (
    <nav className="nav-bar">
      <span className="nav-logo">SAR·UAV</span>

      <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
        ◉ LIVE FEED
      </NavLink>
      <NavLink to="/upload" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
        ↑ UPLOAD
      </NavLink>
      <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
        ≡ HISTORY
      </NavLink>
      <NavLink to="/stats" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
        ∿ STATS
      </NavLink>

      <div className="nav-spacer" />

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {totalDetections > 0 && (
          <span style={{ fontSize: 9, letterSpacing: '1.5px', color: 'var(--danger)', padding: '2px 8px', background: 'rgba(255,34,68,0.1)', border: '1px solid rgba(255,34,68,0.3)', borderRadius: 1 }}>
            {totalDetections} DETECTIONS
          </span>
        )}
        <span className={`status-dot ${wsConnected ? 'online' : 'offline'}`} />
        <span style={{ fontSize: 9, letterSpacing: '2px', color: wsConnected ? 'var(--safe)' : 'var(--danger)' }}>
          {wsConnected ? 'LIVE' : 'OFFLINE'}
        </span>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <NavBar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/history" element={<History />} />
            <Route path="/stats" element={<Stats />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
