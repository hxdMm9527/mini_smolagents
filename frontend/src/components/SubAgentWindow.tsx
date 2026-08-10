import ReactMarkdown from 'react-markdown'
import type { SubGroup } from '../groupEvents'

interface Props {
  group: SubGroup
  collapsed: boolean
  onToggle: () => void
}

function ChildEvent({ ev }: { ev: SubGroup['children'][number] }) {
  switch (ev.type) {
    case 'step':
      return (
        <div className="sub-step">
          Step {ev.step}/{ev.max_steps}
        </div>
      )
    case 'thought':
      return (
        <div className="sub-thought">
          <ReactMarkdown>{ev.content}</ReactMarkdown>
        </div>
      )
    case 'action':
      return <div className="sub-step">🔧 调用 {ev.tool}</div>
    case 'result':
      return (
        <div className="sub-result">
          <ReactMarkdown>{ev.content}</ReactMarkdown>
        </div>
      )
    case 'done':
      return (
        <div className="sub-done">
          <ReactMarkdown>{ev.content}</ReactMarkdown>
        </div>
      )
    case 'note':
      return <div className="sub-step">{ev.content}</div>
    default:
      return null
  }
}

export default function SubAgentWindow({ group, collapsed, onToggle }: Props) {
  const subName = group.action.tool
  const doneEvent = group.children.find((e) => e.type === 'done')
  const preview = doneEvent?.type === 'done' ? doneEvent.content.slice(0, 60) : '执行中...'

  return (
    <div className="sub-agent-window">
      <div className="sub-header" onClick={onToggle}>
        {collapsed ? '▶' : '▼'} 🧩 {subName}
        {collapsed && <span className="collapsed">{preview}</span>}
      </div>
      {!collapsed && (
        <div className="sub-body">
          {group.children.map((ev, i) => (
            <ChildEvent key={i} ev={ev} />
          ))}
          {group.finalResult && group.children.some((e) => e.type !== 'done') && (
            <div className="sub-result">
              <ReactMarkdown>{group.finalResult.content}</ReactMarkdown>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
