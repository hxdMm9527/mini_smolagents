import { useState } from 'react'
import type { MemoryHit } from '../types'

interface Props {
  hits: MemoryHit[]
}

export default function MemoryPanel({ hits }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MemoryHit[] | null>(null)
  const [searching, setSearching] = useState(false)

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
    </aside>
  )
}
