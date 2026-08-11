import { useCallback, useEffect, useRef, useState } from 'react'
import type { StreamEvent } from './types'

export interface UseStreamChat {
  events: StreamEvent[]
  turns: StreamEvent[][]
  streaming: boolean
  sessionId: string | null
  send: (agentId: string, message: string, sessionId?: string) => Promise<void>
  stop: () => void
  reset: () => void
}

export function useStreamChat(): UseStreamChat {
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [turns, setTurns] = useState<StreamEvent[][]>([])
  const [streaming, setStreaming] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const eventsRef = useRef<StreamEvent[]>([])

  useEffect(() => {
    eventsRef.current = events
  }, [events])

  const send = useCallback(
    async (agentId: string, message: string, sessionIdArg?: string) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      // 归档上一轮已完成的事件到历史轮次，再清空开始新一轮
      const prev = eventsRef.current
      if (prev.length) {
        setTurns((t) => [...t, prev])
      }
      setEvents([])
      setStreaming(true)
      if (sessionIdArg) setSessionId(sessionIdArg)

      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent_id: agentId, message, session_id: sessionIdArg ?? null }),
          signal: controller.signal,
        })

        if (!response.body) throw new Error('No response body')
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let idx: number
          while ((idx = buffer.indexOf('\n\n')) !== -1) {
            const chunk = buffer.slice(0, idx)
            buffer = buffer.slice(idx + 2)
            const dataLine = chunk
              .split('\n')
              .find((l) => l.startsWith('data:'))
            if (!dataLine) continue
            const payload = dataLine.slice(5).trim()
            if (!payload) continue
            try {
              const ev = JSON.parse(payload) as StreamEvent
              // 逐条渲染：每次 setEvents 后让出微任务，避免 React 18 同 tick 批处理成一次
              setEvents((prev) => [...prev, ev])
              if (ev.type === 'session') setSessionId(ev.session_id)
              await new Promise<void>((resolve) => queueMicrotask(resolve))
            } catch {
              /* ignore malformed line */
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setEvents((prev) => [
            ...prev,
            { type: 'error', content: (err as Error).message },
          ])
        }
      } finally {
        setStreaming(false)
      }
    },
    [],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setStreaming(false)
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setEvents([])
    setTurns([])
    setSessionId(null)
    setStreaming(false)
  }, [])

  return { events, turns, streaming, sessionId, send, stop, reset }
}
