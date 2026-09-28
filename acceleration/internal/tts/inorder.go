package tts

import (
	"slices"
	"sync"
)

// InOrder makes a provider that takes each sentence as its own request speak them in the
// order they were asked for.
//
// Such a provider runs its requests side by side, which is what has the next sentence
// ready the moment the last one ends, and sends each one's audio back as it arrives.
// Played as it arrives, two sentences are heard spliced into one another. This holds a
// sentence's audio until every sentence asked for before it has finished, so the requests
// still overlap and only what the listener hears is put in order.
//
// A barge-in drops everything held. Every sentence still settles, because a completion is
// what a caller counts speech in flight by and what it is billed by: a sentence nobody
// heard is completed all the same, marked interrupted.
func InOrder(provider TTS) TTS {
	ordered := &inOrder{
		TTS:     provider,
		emitter: NewEmitter(64),
		held:    map[string]*heldSynthesis{},
		dropped: map[string]struct{}{},
	}
	go ordered.relay()
	return ordered
}

type inOrder struct {
	TTS
	emitter *Emitter

	mu sync.Mutex
	// queue is every sentence not yet settled, in the order it was asked for. The head is
	// the one being heard.
	queue []string
	// held is what has arrived for a sentence behind the head.
	held map[string]*heldSynthesis
	// dropped is the sentences a barge-in gave up on whose completion has not come yet.
	dropped map[string]struct{}
}

// heldSynthesis is a sentence waiting for its turn to be heard.
type heldSynthesis struct {
	events []Event
	// complete is set once the provider has finished it, so its completion is the last
	// of events.
	complete bool
}

// Synthesize queues the sentence behind the ones already asked for and sends it to the
// provider straight away.
func (o *inOrder) Synthesize(request Request) error {
	// The order is kept by id, so the sentence needs one before the provider is asked.
	if request.ID == "" {
		request.ID = NewSynthesis("").ID
	}
	o.mu.Lock()
	o.queue = append(o.queue, request.ID)
	o.mu.Unlock()

	if err := o.TTS.Synthesize(request); err != nil {
		// Refused outright, it will send nothing, and waiting for it would silence
		// everything asked for after it.
		o.mu.Lock()
		released := o.forget(request.ID)
		o.mu.Unlock()
		o.sendLater(released)
		return err
	}
	return nil
}

// Interrupt drops every sentence held or being heard and stops the provider.
func (o *inOrder) Interrupt() error {
	o.mu.Lock()
	var settled []Event
	for _, id := range o.queue {
		held := o.held[id]
		if held == nil || !held.complete {
			o.dropped[id] = struct{}{}
			continue
		}
		// Finished while it waited, so its completion is already here and was never
		// passed on. Nobody heard it.
		complete := held.events[len(held.events)-1].(SynthesisComplete)
		complete.Interrupted = true
		settled = append(settled, complete)
	}
	o.queue = nil
	clear(o.held)
	o.mu.Unlock()

	o.sendLater(settled)
	return o.TTS.Interrupt()
}

// Events carries the provider's events with each sentence's audio in the order asked for.
func (o *inOrder) Events() <-chan Event { return o.emitter.Events() }

// Voice is the provider's voice, empty when it cannot say.
func (o *inOrder) Voice() string {
	if voiced, ok := o.TTS.(Voiced); ok {
		return voiced.Voice()
	}
	return ""
}

// relay puts the provider's events in order until it closes.
func (o *inOrder) relay() {
	defer o.emitter.Close()

	for event := range o.TTS.Events() {
		for _, ordered := range o.order(event) {
			o.emitter.Send(ordered)
		}
	}

	// Whatever is still held when the provider closes is passed on rather than lost, so
	// every sentence that finished is still settled.
	o.mu.Lock()
	var rest []Event
	for _, id := range o.queue {
		if held := o.held[id]; held != nil {
			rest = append(rest, held.events...)
		}
	}
	o.queue = nil
	clear(o.held)
	o.mu.Unlock()
	for _, event := range rest {
		o.emitter.Send(event)
	}
}

// order returns what can be passed on now that event has arrived.
func (o *inOrder) order(event Event) []Event {
	var id string
	switch typed := event.(type) {
	case AudioChunk:
		id = typed.SynthesisID
	case SynthesisComplete:
		id = typed.SynthesisID
	default:
		// Starts, errors and the connection's own events are never held: nothing is heard
		// from them, and a caller settling a sentence needs its error before its end.
		return []Event{event}
	}
	complete, completes := event.(SynthesisComplete)

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, gone := o.dropped[id]; gone {
		if !completes {
			return nil
		}
		delete(o.dropped, id)
		complete.Interrupted = true
		return []Event{complete}
	}

	position := slices.Index(o.queue, id)
	switch {
	case position < 0:
		// Not a sentence this asked for, so there is nothing to keep it in order with.
		return []Event{event}
	case position > 0:
		held := o.held[id]
		if held == nil {
			held = &heldSynthesis{}
			o.held[id] = held
		}
		held.events = append(held.events, event)
		held.complete = held.complete || completes
		return nil
	}

	if !completes {
		return []Event{event}
	}
	o.queue = o.queue[1:]
	return append([]Event{event}, o.release()...)
}

// forget takes a sentence out of the queue, returning what that lets through.
func (o *inOrder) forget(id string) []Event {
	position := slices.Index(o.queue, id)
	if position < 0 {
		return nil
	}
	o.queue = slices.Delete(o.queue, position, position+1)
	delete(o.held, id)
	if position > 0 {
		return nil
	}
	return o.release()
}

// release passes on what the new head has already been sent, and carries on past every
// head that had already finished.
func (o *inOrder) release() []Event {
	var released []Event
	for len(o.queue) > 0 {
		head := o.queue[0]
		held := o.held[head]
		delete(o.held, head)
		if held == nil {
			return released
		}
		released = append(released, held.events...)
		if !held.complete {
			return released
		}
		o.queue = o.queue[1:]
	}
	return released
}

// sendLater passes events on without blocking the caller. The caller may be the consumer
// itself, which would wait forever on a full channel only it can drain.
func (o *inOrder) sendLater(events []Event) {
	if len(events) == 0 {
		return
	}
	go func() {
		for _, event := range events {
			o.emitter.Send(event)
		}
	}()
}
