import { useRef, useEffect, useState } from 'react'
import { useStreamChat } from '../useStreamChat'
import type { MemoryHit, SessionMessage } from '../types'
import StreamBlock from './StreamBlock'

interface Props {
  agentId: string
  initialSession: { session_id: string; messages: SessionMessage[] } | null
  onMemoryHits: (hits: MemoryHit[]) => void
  onSessionId: (sid: string) => void
}

export default function ChatPanel({ agentId, initialSession, onMemoryHits, onSessionId }: Props) {
  const { events, streaming, sessionId, send } = useStreamChat()
  const [messages, setMessages] = useState<SessionMessage[]>(initialSession?.messages ?? [])
  const inputRef = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const hit = events.find((e) => e.type === 'memory')
    if (hit) onMemoryHits((hit as { type: 'memory'; hits: MemoryHit[] }).hits)
    const sid = events.find((e) => e.type === 'session')
    if (sid) onSessionId((sid as { type: 'session'; session_id: string }).session_id)
  }, [events, onMemoryHits, onSessionId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [events, messages])

  const doneEvent = [...events].reverse().find((e) => e.type === 'done')

  const handleSend = async () => {
    const text = inputRef.current?.value.trim()
    if (!text || streaming) return
    if (inputRef.current) inputRef.current.value = ''
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    await send(agentId, text, initialSession?.session_id ?? sessionId ?? undefined)
  }

  return (
    <div className="chat-panel">
      <div className="scroll-area" ref={scrollRef}>
        {messages.length === 0 && events.length === 0 && (
          <div className="empty-hint">选择一个 Agent 并输入任务开始对话</div>
        )}

        {messages.map((m, i) => (
          <div className={`msg-bubble ${m.role}`} key={`m${i}`}>
            <div className="agent-title">{m.role === 'user' ? '💬 用户' : '🤖 ' + agentId}</div>
            <div className="bubble">{m.content}</div>
          </div>
        ))}

        {events.length > 0 && (
          <div className="msg-bubble">
            <div className="agent-title">
              🤖 {agentId} {streaming ? '执行中...' : '（完成）'}
            </div>
            <StreamBlock events={events} agentName={agentId} />
          </div>
        )}

        {!streaming && doneEvent && (
          <div className="msg-bubble">
            <div className="agent-title">✍️ {doneEvent.stored ? '已存入长期记忆' : '结果未存入记忆'}</div>
          </div>
        )}
      </div>

      <div className="input-area">
        <input
          ref={inputRef}
          placeholder={streaming ? 'Agent 执行中...' : '输入消息...'}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={streaming}
        />
        <button onClick={handleSend} disabled={streaming}>
          {streaming ? '执行中' : '发送'}
        </button>
      </div>
    </div>
  )
}
