import { useEffect, useState } from 'react'
import type { AgentInfo, MemoryHit } from './types'
import Header from './components/Header'
import ChatPanel from './components/ChatPanel'
import MemoryPanel from './components/MemoryPanel'

export default function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [selected, setSelected] = useState('')
  const [memoryHits, setMemoryHits] = useState<MemoryHit[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'light')

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
          agentId={selected}
          onMemoryHits={setMemoryHits}
          onSessionId={setSessionId}
        />
        <MemoryPanel hits={memoryHits} />
      </div>
      <div style={{ display: 'none' }}>{sessionId}</div>
    </div>
  )
}
