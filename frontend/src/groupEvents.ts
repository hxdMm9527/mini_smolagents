import type { ActionEvent, ResultEvent, StreamEvent } from './types'

export interface MainEvent {
  kind: 'main'
  ev: StreamEvent
}

export interface SubGroup {
  kind: 'sub'
  did: string
  action: ActionEvent
  children: StreamEvent[]
  finalResult?: ResultEvent
}

export type DisplayItem = MainEvent | SubGroup

/** 把连续的 token 事件合并成一条 thought 事件（打字机累积渲染）。 */
function mergeTokens(events: StreamEvent[]): StreamEvent[] {
  const out: StreamEvent[] = []
  let buf = ''
  let bufAgent: string | undefined

  const flush = () => {
    if (buf) {
      out.push({ type: 'thought', agent: bufAgent, content: buf } as StreamEvent)
      buf = ''
    }
  }

  for (const ev of events) {
    if (ev.type === 'token') {
      buf += ev.content
      bufAgent = ev.agent
      continue
    }
    flush()
    out.push(ev)
  }
  flush()
  return out
}

export function groupEvents(events: StreamEvent[]): DisplayItem[] {
  const items: DisplayItem[] = []
  const subs = new Map<string, StreamEvent[]>()
  const mainBuf: StreamEvent[] = []

  for (const ev of events) {
    if (ev.delegation_id) {
      const did = ev.delegation_id
      if (ev.type === 'action') {
        items.push({ kind: 'sub', did, action: ev as ActionEvent, children: [], finalResult: undefined })
        subs.set(did, [])
      } else {
        subs.get(did)?.push(ev)
        // attach final sub result to the matching group
        const group = items.find(
          (i) => i.kind === 'sub' && i.did === did && !i.finalResult,
        )
        if (group?.kind === 'sub' && ev.type === 'result') {
          group.finalResult = ev as ResultEvent
        }
      }
      continue
    }
    mainBuf.push(ev)
  }

  for (const ev of mergeTokens(mainBuf)) {
    items.push({ kind: 'main', ev })
  }

  // assign children to groups (merge tokens within each sub stream)
  for (const item of items) {
    if (item.kind === 'sub') {
      item.children = mergeTokens(subs.get(item.did) ?? [])
    }
  }
  return items
}
