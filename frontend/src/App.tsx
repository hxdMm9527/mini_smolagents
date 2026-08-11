import { useEffect, useState } from 'react'
import type { AgentInfo, MemoryHit, SessionMessage } from './types'
import Header from './components/Header'
import ChatPanel from './components/ChatPanel'
import MemoryPanel from './components/MemoryPanel'

export default function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [selected, setSelected] = useState('')
  const [memoryHits, setMemoryHits] = useState<MemoryHit[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'light')
  const [resumeSession, setResumeSession] = useState<{
    session_id: string
    messages: SessionMessage[]
  } | null>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    fetch('/api/agents')
      .then((r) => r.json())
      .then((list: AgentInfo[]) => {
        setAgents(list)
        if (list.length) setSelected((cur) => cur || list[0].name)
      })
      .catch(() => {})
  }, [])

  const resume = (session_id: string) => {
    fetch(`/api/sessions/${session_id}`)
      .then((r) => r.json())
      .then((data: { session_id: string; messages: SessionMessage[] }) => {
        setResumeSession({ session_id: data.session_id, messages: data.messages })
      })
      .catch(() => {})
  }

  return (
    <div className="app">
      <Header
        agents={agents}
        selected={selected}
        onSelect={setSelected}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />
      <div className="main">
        <ChatPanel
          key={resumeSession?.session_id ?? 'new'}
          agentId={selected}
          initialSession={resumeSession}
          onMemoryHits={setMemoryHits}
          onSessionId={setSessionId}
        />
        <MemoryPanel hits={memoryHits} onResume={resume} />
      </div>
      <div style={{ display: 'none' }}>{sessionId}</div>
    </div>
  )
}
