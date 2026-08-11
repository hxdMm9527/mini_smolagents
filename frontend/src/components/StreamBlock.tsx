import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { groupEvents, type DisplayItem } from '../groupEvents'
import type { StreamEvent } from '../types'
import SubAgentWindow from './SubAgentWindow'

interface Props {
  events: StreamEvent[]
  agentName: string
}

function Md({ text }: { text: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  )
}

interface ToolCardProps {
  tool: string
  args: Record<string, unknown>
  result?: string
  collapsed: boolean
  onToggle: () => void
}

function ToolCard({ tool, args, result, collapsed, onToggle }: ToolCardProps) {
  return (
    <div className="tool-card">
      <div className="tool-head" onClick={onToggle}>
        <span>{collapsed ? '▶' : '▼'}</span>
        <span>🔧 调用 {tool}</span>
        {collapsed && <span className="tool-preview">{args[Object.keys(args)[0]]?.toString().slice(0, 40) ?? ''}</span>}
      </div>
      {!collapsed && (
        <div className="tool-body">
          <div className="action-args">{JSON.stringify(args, null, 2)}</div>
          {result !== undefined && (
            <div className="result">
              ✅ <Md text={result} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface ThoughtCardProps {
  content: string
  collapsed: boolean
  onToggle: () => void
}

function ThoughtCard({ content, collapsed, onToggle }: ThoughtCardProps) {
  if (collapsed) {
    return (
      <div className="thought-head" onClick={onToggle}>
        <span>▶</span>
        <span>💬 思考</span>
      </div>
    )
  }
  return (
    <div>
      <div className="thought-head" onClick={onToggle}>
        <span>▼</span>
        <span>💬 思考</span>
      </div>
      <div className="thought">
        <Md text={content} />
      </div>
    </div>
  )
}

function MainBlock({ ev }: { ev: StreamEvent }) {
  switch (ev.type) {
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

function ItemList({ items }: { items: DisplayItem[] }) {
  // ponytail: thought 默认折叠，避免长思考撑长答案
  const [collapsed, setCollapsed] = useState<Set<string>>(
    () => new Set(items.map((item, i) => (item.kind === 'main' && item.ev.type === 'thought' ? `thought-${i}` : null)).filter(Boolean) as string[]),
  )

  const toggle = (key: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const nodes = []
  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    if (item.kind === 'sub') {
      nodes.push(
        <SubAgentWindow
          key={item.did}
          group={item}
          collapsed={collapsed.has(item.did)}
          onToggle={() => toggle(item.did)}
        />,
      )
      continue
    }
    const ev = item.ev
    if (ev.type === 'thought') {
      const key = `thought-${i}`
      nodes.push(
        <ThoughtCard
          key={key}
          content={ev.content}
          collapsed={collapsed.has(key)}
          onToggle={() => toggle(key)}
        />,
      )
      continue
    }
    // action + 紧随其后的 result 合并成一个可折叠工具卡片
    if (ev.type === 'action') {
      let result: string | undefined
      let consumed = 0
      const next = items[i + 1]
      if (next?.kind === 'main' && next.ev.type === 'result') {
        result = next.ev.content
        consumed = 1
      }
      const key = `tool-${i}`
      nodes.push(
        <ToolCard
          key={key}
          tool={ev.tool}
          args={ev.args}
          result={result}
          collapsed={collapsed.has(key)}
          onToggle={() => toggle(key)}
        />,
      )
      i += consumed
      continue
    }
    nodes.push(<MainBlock key={`m${i}`} ev={ev} />)
  }
  return <>{nodes}</>
}

export default function StreamBlock({ events, agentName }: Props) {
  const items = groupEvents(events)
  const totalSteps = events.filter((e) => e.type === 'step' && !e.delegation_id).length

  return (
    <div>
      <div className="stream-block">
        <div className="stream-header">
          <span>🤖 {agentName}</span>
          {totalSteps > 0 && <span className="step-badge">Step {totalSteps}/{Math.max(totalSteps, 10)}</span>}
          {events.some((e) => e.type === 'memory') && <span style={{ fontSize: 12, color: 'var(--green)' }}>🧠 已关联历史记忆</span>}
        </div>
        <div className="stream-body">
          <ItemList items={items} />
        </div>
      </div>
    </div>
  )
}
