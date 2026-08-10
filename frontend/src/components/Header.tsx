import type { AgentInfo } from '../types'

interface Props {
  agents: AgentInfo[]
  selected: string
  onSelect: (name: string) => void
}

export default function Header({ agents, selected, onSelect }: Props) {
  return (
    <header className="header">
      <h1>mini_smolagents</h1>
      <select value={selected} onChange={(e) => onSelect(e.target.value)}>
        {agents.map((a) => (
          <option key={a.name} value={a.name}>
            {a.name}
          </option>
        ))}
      </select>
    </header>
  )
}
