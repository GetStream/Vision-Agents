package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The native agent: one speech-to-speech model in place of a transcriber, a conversation
// model and a voice.
//
// The model owns endpointing, transcription, synthesis and barge-in, so the cascade's
// flow controller, cadence and chunker do not run. The harness still runs delegated work.
// What is left for the agent to
// do is carry audio in and out of the call, keep the history and the timings, report the
// same events the cascade reports so nothing downstream can tell the two apart, and run
// the tools the model asks for.

// native reports whether this agent is held by one speech-to-speech model.
func (a *Agent) native() bool { return a.options.STS != nil }

// speech returns the native session, or nil when the agent has not joined or is a cascade.
func (a *Agent) speech() *stsrouter.Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.sts
}

// joinNative opens the one session a native agent needs, then joins the call.
//
// Both transcripts are asked for, and asked for as terms: what the caller said and what
// the model said back are what the history, the chat log, the review and memory are all
// built on, so a model that cannot write them down is not a candidate.
func (a *Agent) joinNative() error {
	// Searching is routed before the tools are worked out, because whether the model is
	// offered one depends on whether a provider answered.
	a.startSearching(a.ctx)
	workers, err := a.workers()
	if err != nil {
		return err
	}
	if len(workers) > 0 {
		a.harness, err = harness.New(harness.Options{
			Workers: workers, Capture: a.captureVideo, Skills: a.options.Skills,
			Sandbox: a.options.Sandbox, Tasks: a.options.Tasks, Logger: a.logger,
		})
		if err != nil {
			return err
		}
		a.harnessDrained = make(chan struct{})
		a.running.Add(1)
		go a.consumeHarness()
	}
	tools, err := a.nativeTools()
	if err != nil {
		return err
	}

	// What earlier conversations established goes into the instructions the session
	// opens with, since a native model takes its prompt once rather than per turn.
	if a.memory != nil {
		a.recalled = memory.Prompt(a.memory.Recall(a.ctx, a.options.RecallLimit))
	}

	on := true
	session, err := a.options.STS.Start(a.ctx, stsrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        a.options.STSTarget,
		LanguageHints: a.options.LanguageHints,
		Tools:         tools,
		Options: options.STS{
			Instructions:     a.nativeInstructions(),
			Voice:            a.options.Voice,
			InputTranscript:  &on,
			OutputTranscript: &on,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: start sts: %w", err)
	}
	a.sts = session

	if err := a.options.Edge.Join(a.ctx); err != nil {
		return fmt.Errorf("agent: join edge: %w", err)
	}

	a.mu.Lock()
	a.arrivals = map[string]struct{}{}
	a.lastSpokeAt = time.Now()
	a.mu.Unlock()

	events := make(chan sts.Event, sts.EmitterBuffer)
	a.running.Add(4)
	go a.receiveSTS(a.sts.Events(), events)
	go a.consumeSTS(events)
	go a.consumeEdge()
	go a.consumeNativePresence()
	if roster, ok := a.options.Edge.(Roster); ok {
		a.running.Add(1)
		go a.consumeRoster(roster)
	}

	a.logger.Info("joined", "sts", session.Provider()+"/"+session.Model())
	a.emitter.Send(Joined{At: time.Now()})
	return nil
}

// Prompt asks a native model to speak now, guided by the text. It is what a greeting is
// on a native call: the model says what it makes of the words rather than the words.
func (a *Agent) Prompt(ctx context.Context, text string) error {
	session := a.speech()
	if session == nil {
		if a.native() {
			return errors.New("agent: not joined")
		}
		return errors.New("agent: only a speech-to-speech agent takes a prompt; use Say")
	}
	return session.Prompt(text)
}

// respondNative injects a typed turn. The model answers it as it would a spoken one, and
// the reply arrives on the same events. Images are refused: what a native model sees is
// its own business, and a typed turn is words.
func (a *Agent) respondNative(text string, images []llm.ImagePart) error {
	if len(images) > 0 {
		return errors.New("agent: a speech-to-speech agent takes no images on a typed turn")
	}
	session := a.speech()
	if session == nil {
		return errors.New("agent: not joined")
	}
	caller := stt.Participant{ID: "caller"}
	// A typed turn is not transcribed, so it is written into the history here, where a
	// spoken one is written when the model reports having heard it.
	a.mu.Lock()
	a.history = append(a.history, llm.Message{Role: llm.User, Content: text})
	a.heardText = text
	a.lastParticipant = caller
	a.mu.Unlock()
	return session.SendText(text, caller)
}

// hear feeds one participant's audio to the model.
//
// A speech-to-speech model takes one stream, so the first participant heard is bound to
// the session and everybody else is dropped until that participant leaves. Two people
// interleaved into one stream would be neither of them at half speed. A room of people
// needs a mixer, which this is not.
func (a *Agent) hear(inbound InboundAudio) {
	a.mu.Lock()
	if a.bound.ID == "" {
		a.bound = inbound.Participant
		a.lastParticipant = inbound.Participant
		a.logger.Debug("bound the speech-to-speech session to a participant",
			"participant", inbound.Participant.ID)
	}
	if _, known := a.arrivals[inbound.Participant.ID]; !known {
		a.arrivals[inbound.Participant.ID] = struct{}{}
		if inbound.Participant.ID != a.bound.ID {
			a.logger.Info("dropping a second participant's audio: the model hears one stream",
				"participant", inbound.Participant.ID, "bound", a.bound.ID)
		}
	}
	bound := a.bound.ID
	session := a.sts
	a.mu.Unlock()

	if inbound.Participant.ID != bound || session == nil {
		return
	}
	if err := session.ProcessAudio(inbound.Audio, inbound.Participant); err != nil {
		a.fail(err, "sts")
	}
}

// unbind frees the session for the next speaker when the bound participant leaves.
func (a *Agent) unbind(participant stt.Participant) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.bound.ID == participant.ID {
		a.bound = stt.Participant{}
	}
	delete(a.arrivals, participant.ID)
}

// receiveSTS stops interrupted playback before queuing the remaining ordered events.
func (a *Agent) receiveSTS(source <-chan sts.Event, events chan<- sts.Event) {
	defer a.running.Done()
	defer close(events)
	for event := range source {
		switch typed := event.(type) {
		case sts.ToolCall:
			a.mu.Lock()
			if a.nativeCalls == nil {
				a.nativeCalls = map[string]struct{}{}
			}
			_, duplicate := a.nativeCalls[typed.CallID]
			if a.closed || duplicate {
				a.mu.Unlock()
				continue
			}
			a.nativeCalls[typed.CallID] = struct{}{}
			a.pendingTools++
			a.running.Add(1)
			a.mu.Unlock()
			requested := harness.ToolRequested{
				TurnID: typed.ResponseID,
				Call:   llm.ToolCall{ID: typed.CallID, Name: typed.Name, Arguments: typed.Arguments},
			}
			ctx, cancel := a.prepareTool(requested)
			go func() {
				defer a.running.Done()
				a.executeTool(ctx, cancel, requested)
			}()
			continue
		case sts.ToolCancel:
			a.mu.Lock()
			for _, id := range typed.CallIDs {
				if cancel := a.toolCancels[id]; cancel != nil {
					cancel()
				}
			}
			a.mu.Unlock()
			continue
		case sts.SpeechStarted:
			a.mu.Lock()
			a.nativeListening = true
			a.mu.Unlock()
		case sts.SpeechStopped:
			a.mu.Lock()
			a.nativeListening = false
			a.mu.Unlock()
		}
		if complete, ok := event.(sts.ResponseComplete); ok && complete.Interrupted {
			a.mu.Lock()
			_, abandoned := a.abandoned[complete.ResponseID]
			a.abandoned[complete.ResponseID] = struct{}{}
			playing := a.speakingTurn == complete.ResponseID
			participant := a.lastParticipant
			a.mu.Unlock()
			if playing {
				a.dropSpeech()
			}
			if !abandoned {
				a.turns.interrupt(complete.ResponseID)
				a.emitter.Send(Interrupted{TurnID: complete.ResponseID, Participant: participant})
			}
		}
		select {
		case events <- event:
		case <-a.ctx.Done():
			return
		default:
			a.fail(errors.New("agent: speech-to-speech playback queue is full"), "sts")
			go a.Close()
			return
		}
	}
}

// consumeSTS keeps transcripts and playback in provider order. Interruption is handled
// separately by receiveSTS because PublishAudio can block until queued audio plays.
func (a *Agent) consumeSTS(events <-chan sts.Event) {
	defer a.running.Done()

	for event := range events {
		switch typed := event.(type) {
		case sts.Connected:
			a.logger.Info("conversing", "provider", typed.Provider, "model", typed.Model)

		case sts.SpeechStarted:
			a.mu.Lock()
			a.lastParticipant = a.participantOr(typed.Participant)
			a.lastHeardAt = typed.At
			a.mu.Unlock()

		case sts.SpeechStopped:
			// The moment the caller stopped is when their wait begins. The event's own time
			// rather than now: a model that had no stop event of its own reports the time
			// of the last words it heard, which is nearer the truth than when it said so.
			a.mu.Lock()
			a.waitingSince = typed.At
			a.lastHeardAt = typed.At
			a.mu.Unlock()

		case sts.InputTranscript:
			a.heardNative(typed)

		case sts.OutputTranscript:
			a.saidNative(typed)

		case sts.ResponseStarted:
			a.replyStarted(typed)

		case sts.AudioChunk:
			a.replyAudio(typed)

		case sts.ResponseComplete:
			a.replyComplete(typed)

		case sts.SessionExpiring:
			a.logger.Warn("the speech-to-speech session is about to be cut off",
				"time_left", typed.TimeLeft)

		case sts.Disconnected:
			if typed.Clean {
				a.logger.Debug("the speech-to-speech session closed",
					"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)
				continue
			}
			a.logger.Warn("the speech-to-speech session dropped",
				"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)

		case sts.Error:
			a.fail(typed.Err, "sts")
		}
	}

	// The model's session ending is the whole conversation ending: there is nothing
	// else on this call that can hear or speak. Closing waits for this goroutine, so it is
	// done from another.
	a.mu.Lock()
	closed := a.closed
	a.mu.Unlock()
	if !closed {
		a.logger.Warn("the speech-to-speech session ended, leaving the call")
		go func() {
			if err := a.Close(); err != nil {
				a.logger.Error("could not leave after the model went", "error", err)
			}
		}()
	}
}

// participantOr is the participant an event names, or whoever the session is bound to
// when it names nobody. The caller holds the lock.
func (a *Agent) participantOr(named stt.Participant) stt.Participant {
	if named.ID != "" {
		return named
	}
	if a.bound.ID != "" {
		return a.bound
	}
	return a.lastParticipant
}

// heardNative reports what the model wrote down of the caller. A settled turn goes into
// the history and is Heard; anything before that is Hearing, the words still changing.
func (a *Agent) heardNative(transcript sts.InputTranscript) {
	a.mu.Lock()
	participant := a.participantOr(transcript.Participant)
	if transcript.Mode != stt.ModeFinal {
		if transcript.Mode == stt.ModeReplacement {
			a.hearing.Reset()
		}
		a.hearing.WriteString(transcript.Text)
		text := a.hearing.String()
		a.mu.Unlock()
		a.emitter.Send(Hearing{Participant: participant, Text: text, Language: transcript.Language})
		return
	}

	text := strings.TrimSpace(transcript.Text)
	a.hearing.Reset()
	if text == "" {
		a.mu.Unlock()
		return
	}
	a.history = append(a.history, llm.Message{Role: llm.User, Content: text})
	a.heardText = text
	a.lastParticipant = participant
	a.lastHeardAt = time.Now()
	// SpeechStopped starts the wait. Transcription can finish after the reply, so it
	// must not open another wait for a turn the model has already answered.
	a.mu.Unlock()

	a.logger.Debug("heard", "participant", participant.ID, "text", text)
	a.emitter.Send(Heard{Participant: participant, Text: text, Language: transcript.Language})
}

// saidNative reports what the model wrote down of itself. Deltas are what a watcher reads
// as the reply streams; a settled transcript restates the whole and is kept for the
// history without being reported again.
func (a *Agent) saidNative(transcript sts.OutputTranscript) {
	switch transcript.Mode {
	case stt.ModeFinal, stt.ModeReplacement:
		a.spoken.Reset()
		a.spoken.WriteString(transcript.Text)
		if transcript.Mode == stt.ModeFinal {
			return
		}
	default:
		a.spoken.WriteString(transcript.Text)
	}
	a.mu.Lock()
	a.saying = a.spoken.String()
	a.mu.Unlock()
	a.emitter.Send(ResponseDelta{TurnID: transcript.ResponseID, Text: transcript.Text})
}

// replyStarted opens a turn for a reply the model began.
//
// The turn is only measured when somebody was waiting for it: a reply to a prompt, a
// typed turn or a tool result answers nobody's silence, so timing it from the last time
// the caller stopped would measure the wrong wait. The cascade skips the same replies.
func (a *Agent) replyStarted(started sts.ResponseStarted) {
	a.spoken.Reset()
	a.mu.Lock()
	a.nativeAwaiting = false
	a.speakingTurn = started.ResponseID
	a.generating = true
	a.utterances++
	participant := a.lastParticipant
	waiting := a.waitingSince
	a.waitingSince = time.Time{}
	prompt := a.heardText
	a.mu.Unlock()

	if !waiting.IsZero() {
		a.turns.begin(started.ResponseID, participant, waiting, 0)
	}
	a.emitter.Send(Responding{TurnID: started.ResponseID, Participant: participant, Prompt: prompt})
}

// replyAudio publishes a piece of the model's voice, unless the reply was abandoned.
func (a *Agent) replyAudio(chunk sts.AudioChunk) {
	if a.abandonedTurn(chunk.ResponseID) {
		a.turns.dropped(chunk.ResponseID, chunk.Audio.DurationMs())
		return
	}
	if err := a.options.Edge.PublishAudio(chunk.Audio); err != nil {
		a.fail(err, "edge")
	}
	if a.abandonedTurn(chunk.ResponseID) {
		a.dropSpeech()
		return
	}
	a.mu.Lock()
	a.lastSpokeAt = time.Now()
	a.mu.Unlock()
	// The wait ends when the participant can hear something, so this is timed after the
	// publish rather than before it.
	a.turns.firstAudio(chunk.ResponseID, time.Now())
}

// replyComplete settles a reply. Interrupted is the one signal that the caller cut in.
//
// A turn has three legs on a cascade and one here: the model heard, thought and spoke
// without a seam to time, so only the roundtrip is recorded and the tracker leaves the
// rest empty rather than zero.
func (a *Agent) replyComplete(complete sts.ResponseComplete) {
	said := strings.TrimSpace(a.spoken.String())
	a.spoken.Reset()

	a.mu.Lock()
	_, alreadyAbandoned := a.abandoned[complete.ResponseID]
	complete.Interrupted = complete.Interrupted || alreadyAbandoned
	if complete.Interrupted {
		a.abandoned[complete.ResponseID] = struct{}{}
	}
	if a.speakingTurn == complete.ResponseID {
		a.speakingTurn = ""
	}
	a.generating = false
	a.saying = ""
	participant := a.lastParticipant
	if said != "" {
		a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: said})
	}
	exchange := lastExchange(a.history)
	pending := a.pendingTools
	a.mu.Unlock()
	a.settle()

	if complete.Interrupted {
		// The model stopped when it heard the caller, and what it had already sent is
		// queued at the edge: leaving it there is the caller being talked over for as
		// long as the queue is deep. An interrupt the caller asked for by hand already
		// reported itself, so the model's own report of the same cut is not repeated.
		if !alreadyAbandoned {
			a.turns.interrupt(complete.ResponseID)
			a.emitter.Send(Interrupted{TurnID: complete.ResponseID, Participant: participant})
		}
	} else {
		a.turns.completed(complete.ResponseID, 0, 1)
		a.turns.spoke(complete.ResponseID, 0, complete.AudioDurationMs)
		// Remembering happens off the turn path: extraction takes longer than a turn and
		// the next thing the participant says must not wait for it.
		if a.memory != nil {
			a.memory.Remember(exchange)
		}
	}

	a.emitter.Send(Responded{TurnID: complete.ResponseID, Text: said, PendingWork: pending > 0 || a.Busy()})
	if !complete.Interrupted {
		a.emitter.Send(Spoke{
			TurnID:            complete.ResponseID,
			AudioDurationMs:   complete.AudioDurationMs,
			TimeToFirstByteMs: complete.TimeToFirstByteMs,
		})
	}
	a.followUp()
}

const (
	delegateSkill = "delegate_skill"
	cancelSkill   = "cancel_skill"
)

func (a *Agent) nativeInstructions() string {
	if a.harness == nil || len(a.options.Skills.Skills) == 0 {
		return a.instructions()
	}
	return a.instructions() + "\n\nUse delegate_skill for work that needs a subagent. " +
		"It returns a task ID immediately; the task is still running. Keep conversing while it runs. " +
		"The findings will arrive later. Never invent the result or delegate the same completed work again. " +
		"Use cancel_skill when the caller no longer needs that work. Interrupting speech alone does not cancel it."
}

func (a *Agent) nativeTools() ([]llm.Tool, error) {
	tools := a.availableTools()
	if a.harness == nil || len(a.options.Skills.Skills) == 0 {
		return tools.Requests(), nil
	}
	for _, name := range []string{delegateSkill, cancelSkill} {
		if _, exists := tools.Lookup(name); exists {
			return nil, fmt.Errorf("agent: %s is reserved for native delegation", name)
		}
	}
	names := make([]string, 0, len(a.options.Skills.Skills))
	var descriptions strings.Builder
	for _, skill := range a.options.Skills.Skills {
		names = append(names, skill.Name)
		fmt.Fprintf(&descriptions, "\n%s: %s", skill.Name, skill.Description)
	}
	return append(tools.Requests(), llm.Tool{
		Name: delegateSkill, Description: "Start background work and return its task ID without waiting for the answer." + descriptions.String(),
		Parameters: map[string]any{
			"type": "object", "required": []string{"skill", "prompt"},
			"properties": map[string]any{
				"skill":  map[string]any{"type": "string", "enum": names},
				"prompt": map[string]any{"type": "string", "description": "The question or work to hand over."},
			},
		},
	}, llm.Tool{
		Name: cancelSkill, Description: "Cancel a skill's background work when it is no longer relevant.",
		Parameters: map[string]any{
			"type": "object", "required": []string{"skill"},
			"properties": map[string]any{"skill": map[string]any{"type": "string", "enum": names}},
		},
	}), nil
}

func (a *Agent) nativeDelegate(call llm.ToolCall) ([]llm.ContentPart, bool, error) {
	var arguments struct {
		Skill  string `json:"skill"`
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return nil, false, err
	}
	if a.harness == nil {
		return nil, false, errors.New("agent: no subagents configured")
	}
	if call.Name == cancelSkill {
		err := a.harness.CancelSkill(arguments.Skill)
		return llm.TextParts("Cancelled the skill's pending work."), false, err
	}
	a.mu.Lock()
	history := make([]llm.Message, 0, len(a.history))
	for _, message := range a.history {
		if message.Role == llm.User || message.Role == llm.Assistant {
			history = append(history, message)
		}
	}
	a.mu.Unlock()
	id, err := a.harness.Delegate(arguments.Skill, arguments.Prompt, call.ID, nil, history)
	if err != nil {
		return nil, false, err
	}
	result, err := json.Marshal(map[string]string{"task_id": id, "status": "running"})
	return llm.TextParts(string(result)), false, err
}

func (a *Agent) followNative() error {
	a.following.Lock()
	defer a.following.Unlock()
	if a.waitingForPlayout() {
		return nil
	}
	a.mu.Lock()
	if a.closed || a.sts == nil || a.harness == nil || a.generating || a.utterances > 0 || a.pendingTools > 0 || a.nativeAwaiting || a.nativeListening || !a.waitingSince.IsZero() || !a.harness.Pending() {
		a.mu.Unlock()
		return nil
	}
	a.generating = true
	prompt := a.instructions() + "\n\nThis reply must deliver the following background task update to the caller. " +
		"The update supersedes the earlier running acknowledgement. Do not delegate again or say you are still waiting.\n" + a.harness.TakeNotes()
	model := a.sts
	a.mu.Unlock()
	if err := model.Prompt(prompt); err != nil {
		a.mu.Lock()
		a.generating = false
		a.mu.Unlock()
		return err
	}
	return nil
}

func (a *Agent) consumeNativePresence() {
	defer a.running.Done()
	ticker := time.NewTicker(presenceTick)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			a.followUp()
		case <-a.ctx.Done():
			return
		}
	}
}
