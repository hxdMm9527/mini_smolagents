import { useEffect, useState } from 'react'
import type { MemoryHit, SessionInfo } from '../types'

interface Props {
  hits: MemoryHit[]
  onResume: (sessionId: string) => void
}

const PAGE = 5

export default function MemoryPanel({ hits, onResume }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MemoryHit[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [showAll, setShowAll] = useState(false)

  const loadSessions = () => {
    fetch('/api/sessions')
      .then((r) => r.json())
      .then((data) => setSessions(data.sessions ?? []))
      .catch(() => {})
  }

  useEffect(loadSessions, [])

  const removeSession = async (sessionId: string) => {
    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' })
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
  }

  const search = async () => {
    if (!query.trim()) return
    setSearching(true)
    try {
      const resp = await fetch('/api/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })
      const data = await resp.json()
      setResults(data.hits)
    } finally {
      setSearching(false)
    }
  }

  const visible = showAll ? sessions : sessions.slice(0, PAGE)

  return (
    <aside className="memory-panel">
      <h2>🔍 记忆搜索</h2>
      <input
        placeholder="搜索历史记忆..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && search()}
      />
      {searching && <div className="memory-item">搜索中...</div>}

      {results && results.length === 0 && (
        <div className="memory-item">没有找到相关记忆</div>
      )}
      {results?.map((r, i) => (
        <div className="memory-item" key={i}>
          <div className="task">{r.task}</div>
          <div>{r.document.slice(0, 150)}</div>
          <div className="score">score: {r.score.toFixed(3)}</div>
        </div>
      ))}

      {hits.length > 0 && (
        <>
          <h2 style={{ marginTop: 14 }}>🧠 本次相关记忆</h2>
          {hits.map((h, i) => (
            <div className="memory-hit" key={i}>
              <div className="task">{h.task}</div>
              <div>{h.document.slice(0, 150)}</div>
            </div>
          ))}
        </>
      )}

      <h2 style={{ marginTop: 14 }}>🗂 历史会话</h2>
      {sessions.length === 0 && <div className="memory-item">暂无历史会话</div>}
      {visible.map((s) => (
        <div className="session-item" key={s.session_id}>
          <button className="session-main" onClick={() => onResume(s.session_id)}>
            <div className="task">{s.title}</div>
            <div className="session-time">{s.saved_at.slice(0, 19).replace('T', ' ')}</div>
          </button>
          <button
            className="session-del"
            title="删除"
            onClick={() => removeSession(s.session_id)}
          >
            ✕
          </button>
        </div>
      ))}
      {sessions.length > PAGE && (
        <button className="more-btn" onClick={() => setShowAll((v) => !v)}>
          {showAll ? '收起' : '更多'}
        </button>
      )}
    </aside>
  )
}
