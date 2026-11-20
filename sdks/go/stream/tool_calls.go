package stream

import (
	"encoding/json"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// listeners are the handlers On registered, by the kind of event each wants, and the last
// state of every tool call the session reported.
type listeners struct {
	mu        sync.Mutex
	next      int
	by        map[string]map[int]func(Event)
	toolCalls map[string]acceleration.ToolCallState
}

// On calls handle with every event of kind the session reports, such as "tool_call.finished"
// or "responded", until off is called. Handlers run one at a time on the goroutine reading
// the session, before the event reaches Events, so a slow one holds up the rest.
func (p *Pipeline) On(kind string, handle func(Event)) (off func()) {
	p.listeners.mu.Lock()
	defer p.listeners.mu.Unlock()
	if p.listeners.by == nil {
		p.listeners.by = map[string]map[int]func(Event){}
	}
	if p.listeners.by[kind] == nil {
		p.listeners.by[kind] = map[int]func(Event){}
	}
	id := p.listeners.next
	p.listeners.next++
	p.listeners.by[kind][id] = handle
	return func() {
		p.listeners.mu.Lock()
		defer p.listeners.mu.Unlock()
		delete(p.listeners.by[kind], id)
	}
}

// ToolCall is the last state the session reported for the tool call id, whichever process
// runs it, reporting false for a call it has reported nothing about.
func (p *Pipeline) ToolCall(id string) (acceleration.ToolCallState, bool) {
	p.listeners.mu.Lock()
	defer p.listeners.mu.Unlock()
	state, ok := p.listeners.toolCalls[id]
	return state, ok
}

// ToolCalls is the last state of every tool call the session reported, finished ones too.
func (p *Pipeline) ToolCalls() []acceleration.ToolCallState {
	p.listeners.mu.Lock()
	defer p.listeners.mu.Unlock()
	states := make([]acceleration.ToolCallState, 0, len(p.listeners.toolCalls))
	for _, state := range p.listeners.toolCalls {
		states = append(states, state)
	}
	return states
}

// observe keeps the state event carries and hands event to its kind's handlers.
func (p *Pipeline) observe(event Event) {
	p.listeners.mu.Lock()
	if event.ToolCall != nil {
		if p.listeners.toolCalls == nil {
			p.listeners.toolCalls = map[string]acceleration.ToolCallState{}
		}
		p.listeners.toolCalls[event.ToolCall.ToolCallId] = *event.ToolCall
	}
	handlers := make([]func(Event), 0, len(p.listeners.by[event.Kind]))
	for _, handle := range p.listeners.by[event.Kind] {
		handlers = append(handlers, handle)
	}
	p.listeners.mu.Unlock()
	for _, handle := range handlers {
		handle(event)
	}
}

// toolCallOf reads a tool_call.started, tool_call.progress or tool_call.finished frame,
// reporting false for any other.
func toolCallOf(frame Frame) (*acceleration.ToolCallState, bool) {
	if !strings.HasPrefix(frame.Type(), "tool_call.") {
		return nil, false
	}
	raw, err := json.Marshal(frame)
	if err != nil {
		return nil, false
	}
	var state acceleration.ToolCallState
	if json.Unmarshal(raw, &state) != nil || state.ToolCallId == "" {
		return nil, false
	}
	return &state, true
}
