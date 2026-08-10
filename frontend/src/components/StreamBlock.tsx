import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { groupEvents } from '../groupEvents'
import type { StreamEvent } from '../types'
import SubAgentWindow from './SubAgentWindow'

interface Props {
  events: StreamEvent[]
  agentName: string
}

function ActionCard({ tool, args }: { tool: string; args: Record<string, unknown> }) {
  return (
    <div className="action-card">
      <div className="action-head">🔧 调用 {tool}</div>
      <div className="action-args">{JSON.stringify(args, null, 2)}</div>
    </div>
  )
}

function Md({ text }: { text: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  )
}

function MainBlock({ ev }: { ev: StreamEvent }) {
  switch (ev.type) {
    case 'thought':
      return <div className="thought">💬 <Md text={ev.content} /></div>
    case 'action':
      return <ActionCard tool={ev.tool} args={ev.args} />
    case 'result':
      return <div className="result">✅ <Md text={ev.content} /></div>
    case 'note':
      return <div className="note">{ev.content}</div>
    case 'done':
      return (
        <div className="done">
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
            ✨ 最终结果{ev.stored ? '（已存入记忆）' : ''}
          </div>
          <Md text={ev.content} />
        </div>
      )
    case 'error':
      return <div className="error">❌ {ev.content}</div>
    default:
      return null
  }
}

export default function StreamBlock({ events, agentName }: Props) {
  const items = groupEvents(events)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const totalSteps = events.filter((e) => e.type === 'step' && !e.delegation_id).length

  const toggle = (did: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(did)) next.delete(did)
      else next.add(did)
      return next
    })
  }

  return (
    <div>
      <div className="stream-block">
        <div className="stream-header">
          <span>🤖 {agentName}</span>
          {totalSteps > 0 && <span className="step-badge">Step {totalSteps}/{Math.max(totalSteps, 10)}</span>}
          {events.some((e) => e.type === 'memory') && <span style={{ fontSize: 12, color: 'var(--green)' }}>🧠 已关联历史记忆</span>}
        </div>
        <div className="stream-body">
          {items.map((item, i) =>
            item.kind === 'main' ? (
              <MainBlock key={i} ev={item.ev} />
            ) : (
              <SubAgentWindow
                key={item.did}
                group={item}
                collapsed={collapsed.has(item.did)}
                onToggle={() => toggle(item.did)}
              />
            ),
          )}
        </div>
      </div>
    </div>
  )
}
