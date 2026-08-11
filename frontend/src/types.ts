export interface AgentInfo {
  name: string
  description: string
  capabilities: string[]
  tools: string[]
}

export interface MemoryHit {
  document: string
  task: string
  score: number
}

export interface SessionInfo {
  session_id: string
  saved_at: string
  title: string
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Turn {
  user: string
  events: StreamEvent[]
}

export interface BaseEvent {
  type: string
  agent?: string
  delegation_id?: string
}

export interface StepEvent extends BaseEvent {
  type: 'step'
  step: number
  max_steps: number
}

export interface MemoryEvent extends BaseEvent {
  type: 'memory'
  hits: MemoryHit[]
}

export interface ThoughtEvent extends BaseEvent {
  type: 'thought'
  content: string
}

export interface TokenEvent extends BaseEvent {
  type: 'token'
  content: string
}

export interface ActionEvent extends BaseEvent {
  type: 'action'
  tool: string
  args: Record<string, unknown>
}

export interface ResultEvent extends BaseEvent {
  type: 'result'
  content: string
}

export interface NoteEvent extends BaseEvent {
  type: 'note'
  content: string
}

export interface DoneEvent extends BaseEvent {
  type: 'done'
  content: string
  stored?: boolean
  session_id?: string
}

export interface SessionEvent extends BaseEvent {
  type: 'session'
  session_id: string
}

export interface EndEvent extends BaseEvent {
  type: 'end'
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  content: string
}

export type StreamEvent =
  | StepEvent
  | MemoryEvent
  | ThoughtEvent
  | TokenEvent
  | ActionEvent
  | ResultEvent
  | NoteEvent
  | DoneEvent
  | SessionEvent
  | EndEvent
  | ErrorEvent
