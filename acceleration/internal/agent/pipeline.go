package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// pipeline is what hears and answers on the call: the cascade's model, voice and
// transcribers, or one speech-to-speech model. The call, the history and the emitter
// belong to the agent and outlive it, which is what lets a session move onto other models
// without leaving the call.
type pipeline struct {
	native  bool
	ctx     context.Context
	cancel  context.CancelFunc
	running sync.WaitGroup
}

func newPipeline(parent context.Context, native bool) *pipeline {
	ctx, cancel := context.WithCancel(parent)
	return &pipeline{native: native, ctx: ctx, cancel: cancel}
}

// Settings is what a running session is moved onto. A speech-to-speech target makes it
// native; an empty one makes it a cascade on the other targets.
type Settings struct {
	LLMTarget        string
	ControllerTarget string
	STTTarget        string
	TTSTarget        string
	STSTarget        string
	SubagentTarget   string
	Voice            string
	Overwrites       options.LLM
}

// controller is the flow controller's target, which follows the conversation's model
// when none was named.
func (s Settings) controller() string {
	if s.ControllerTarget != "" {
		return s.ControllerTarget
	}
	return s.LLMTarget
}

// prepared is what a swap opened before touching the running pipeline, so one that could
// not open everything it needs changes nothing.
type prepared struct {
	llm *llmrouter.Session
	// controller is owned by harness when there is one.
	controller *llmrouter.Session
	harness    *harness.Harness
	tts        *ttsrouter.Session
	sts        *stsrouter.Session
	subagent   *llmrouter.Session
}

// close releases what was opened and never used.
func (p *prepared) close() {
	if p.harness != nil {
		_ = p.harness.Close()
	} else if p.controller != nil {
		_ = p.controller.Close()
	}
	for _, session := range []*llmrouter.Session{p.llm, p.subagent} {
		if session != nil {
			_ = session.Close()
		}
	}
	if p.tts != nil {
		_ = p.tts.Close()
	}
	if p.sts != nil {
		_ = p.sts.Close()
	}
}

// SetSettings moves the session onto other models, another voice or the other pipeline.
// The agent config it started from is not touched.
//
// Everything the change needs is opened first, so one that fails changes nothing. It is
// swapped in at the next turn boundary: a reply in flight finishes on the models it
// started on, the same rule SetInstructions keeps.
func (a *Agent) SetSettings(ctx context.Context, next Settings) error {
	a.swapping.Lock()
	defer a.swapping.Unlock()

	a.mu.Lock()
	if !a.joined || a.closed || a.pipe == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	current := a.settingsLocked()
	wasNative := a.pipe.native
	a.mu.Unlock()

	if err := a.validate(next); err != nil {
		return err
	}
	native := next.STSTarget != ""
	// Delegation changes the tools a native model is offered and whether a cascade has
	// anything to hand work to, so gaining or losing a subagent rebuilds the pipeline.
	restart := native != wasNative || (current.SubagentTarget == "") != (next.SubagentTarget == "")

	prep, err := a.prepare(current, next, native, restart)
	if err != nil {
		return err
	}
	if err := a.quiesce(ctx); err != nil {
		prep.close()
		return err
	}
	defer a.switching.Store(false)

	a.mu.Lock()
	a.options.LLMTarget = next.LLMTarget
	a.options.ControllerTarget = next.ControllerTarget
	a.options.STTTarget = next.STTTarget
	a.options.TTSTarget = next.TTSTarget
	a.options.STSTarget = next.STSTarget
	a.options.SubagentTarget = next.SubagentTarget
	a.options.Voice = next.Voice
	a.options.Overwrites = next.Overwrites
	a.mu.Unlock()

	var failures []error
	switch {
	case restart:
		// Work handed to the old pipeline's subagent is abandoned with its harness.
		failures = a.stopPipeline(false)
		if native {
			a.startNative(prep)
		} else {
			a.startCascade(prep)
		}
	case native && prep.sts != nil:
		failures = a.stopPipeline(true)
		a.startNative(prep)
	case !native:
		failures = a.swapCascade(current, next, prep)
	}
	if !restart {
		a.swapDelegation(current, next, prep)
	}
	for _, failure := range failures {
		a.logger.Warn("closing a replaced session failed", "error", failure)
	}

	changed := a.modelsChanged()
	a.logger.Info("models changed", "native", changed.Native, "llm", changed.LLM,
		"stt", changed.STT, "tts", changed.TTS, "sts", changed.STS, "voice", changed.Voice)
	a.emitter.Send(changed)
	return nil
}

// Native reports whether the session is held by one speech-to-speech model.
func (a *Agent) Native() bool { return a.native() }

// validate refuses a change this deployment or this kind of session cannot make.
func (a *Agent) validate(next Settings) error {
	if next.STSTarget != "" {
		if a.options.Text {
			return errors.New("agent: a text agent has no voice, so it cannot run a speech-to-speech model")
		}
		if a.options.STS == nil {
			return errors.New("agent: this deployment has no speech-to-speech models")
		}
	} else {
		if a.options.LLM == nil {
			return errors.New("agent: an llm router is required")
		}
		if !a.options.Text && (a.options.STT == nil || a.options.TTS == nil) {
			return errors.New("agent: this deployment cannot run a cascade")
		}
	}
	if next.SubagentTarget != "" && a.options.LLM == nil {
		return errors.New("agent: a subagent requires an llm router")
	}
	return nil
}

// prepare opens what the change needs, closing everything it opened if any of it fails.
func (a *Agent) prepare(current, next Settings, native, restart bool) (*prepared, error) {
	if restart {
		if native {
			return a.openNative(next)
		}
		rebuilt, err := a.openCascade(next)
		if err != nil {
			return nil, err
		}
		// Only on a swap: joining a call already pays for the first turn, and a model that
		// was chosen once and works does not need proving again on every call.
		if err := answers(a.ctx, rebuilt.llm); err != nil {
			rebuilt.close()
			return nil, fmt.Errorf("agent: %s cannot answer: %w", next.LLMTarget, err)
		}
		return rebuilt, nil
	}

	prep := &prepared{}
	var err error
	fail := func(failure error) (*prepared, error) {
		prep.close()
		return nil, failure
	}
	if native {
		if next.STSTarget != current.STSTarget || next.Voice != current.Voice {
			if prep.sts, err = a.openSpeech(next, current.SubagentTarget != ""); err != nil {
				return fail(err)
			}
		}
	} else {
		if next.LLMTarget != current.LLMTarget {
			if prep.llm, err = a.startLLM(next.LLMTarget); err != nil {
				return fail(fmt.Errorf("agent: start llm: %w", err))
			}
			if err = answers(a.ctx, prep.llm); err != nil {
				return fail(fmt.Errorf("agent: %s cannot answer: %w", next.LLMTarget, err))
			}
		}
		if next.controller() != current.controller() {
			if prep.controller, err = a.startLLM(next.controller()); err != nil {
				return fail(fmt.Errorf("agent: start flow controller: %w", err))
			}
		}
		if !a.options.Text && (next.TTSTarget != current.TTSTarget || next.Voice != current.Voice) {
			if prep.tts, err = a.startVoice(next); err != nil {
				return fail(fmt.Errorf("agent: start tts: %w", err))
			}
		}
		// A transcriber opens per participant on first hearing them, so the new target is
		// tried once here: a mistake is reported now rather than as a deaf agent later.
		if !a.options.Text && next.STTTarget != current.STTTarget {
			probe, err := a.startListener(next.STTTarget)
			if err != nil {
				return fail(fmt.Errorf("agent: start stt: %w", err))
			}
			_ = probe.Close()
		}
	}
	if next.SubagentTarget != "" && next.SubagentTarget != current.SubagentTarget {
		if prep.subagent, err = a.openSubagent(next.SubagentTarget)(a.ctx); err != nil {
			return fail(fmt.Errorf("agent: start subagent: %w", err))
		}
	}
	return prep, nil
}

// quiesce waits for a turn boundary: nothing being written, said or run on the caller's
// behalf. It then holds the floor, so no turn starts and no audio is taken in while the
// pipeline changes underneath.
func (a *Agent) quiesce(ctx context.Context) error {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		a.mu.Lock()
		if a.closed {
			a.mu.Unlock()
			return errors.New("agent: closed")
		}
		if !a.generating && a.utterances == 0 && a.pendingTools == 0 && !a.toolReply &&
			!a.nativeAwaiting && len(a.streams) == 0 && len(a.generatingCancel) == 0 {
			a.switching.Store(true)
			a.mu.Unlock()
			return nil
		}
		a.mu.Unlock()

		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// swapCascade moves a running cascade onto the sessions prepared for it, in place.
func (a *Agent) swapCascade(current, next Settings, prep *prepared) []error {
	a.mu.Lock()
	replacedLLM, replacedTTS := a.llm, a.tts
	if prep.llm != nil {
		a.llm = prep.llm
	}
	if prep.tts != nil {
		a.tts = prep.tts
		a.voicePrompt = prep.tts.Prompt()
		a.performs = prep.tts.Performs()
	}
	var listeners []*sttrouter.Session
	if next.STTTarget != current.STTTarget {
		for _, listener := range a.listeners {
			listeners = append(listeners, listener)
		}
		a.listeners = map[string]*sttrouter.Session{}
		a.voices = map[string]string{}
	}
	model, conversation, p := a.llm, a.harness, a.pipe
	a.mu.Unlock()

	if prep.llm != nil || prep.controller != nil {
		conversation.SetModel(model, prep.controller)
	}
	var failures []error
	if prep.tts != nil {
		p.running.Add(1)
		go a.consumeTTS(p, prep.tts)
		if err := replacedTTS.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close tts: %w", err))
		}
	}
	if prep.llm != nil {
		if err := replacedLLM.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close llm: %w", err))
		}
	}
	for _, listener := range listeners {
		if err := listener.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close stt: %w", err))
		}
	}
	return failures
}

// swapDelegation moves delegated work onto the prepared subagent and new overwrites.
// Tasks already running finish on the model they started on.
func (a *Agent) swapDelegation(current, next Settings, prep *prepared) {
	a.mu.Lock()
	conversation := a.harness
	a.mu.Unlock()
	if conversation == nil {
		return
	}
	if prep.subagent != nil {
		subagent := prep.subagent
		conversation.SetSubagent(func(context.Context) (*llmrouter.Session, error) { return subagent, nil })
	}
	conversation.SetOverwrites(next.Overwrites)
}

// startCascade runs a cascade on the sessions opened for it.
func (a *Agent) startCascade(prep *prepared) {
	p := newPipeline(a.ctx, false)
	replies := make(chan llm.Event, replyBuffer)
	drained := make(chan struct{})
	a.mu.Lock()
	a.pipe = p
	a.llm, a.harness, a.tts = prep.llm, prep.harness, prep.tts
	a.replies = replies
	a.harnessDrained = drained
	a.voicePrompt, a.performs = "", false
	if prep.tts != nil {
		// What the voice wants said about it is read once, here, rather than on every
		// turn: instructions() is called under the lock this session was opened outside of.
		a.voicePrompt = prep.tts.Prompt()
		a.performs = prep.tts.Performs()
	}
	a.mu.Unlock()
	a.nativeMode.Store(false)

	a.running.Add(1)
	go a.consumeHarness(prep.harness, drained)
	p.running.Add(1)
	go a.consumeLLM(p, replies)
	// The other three all begin at a microphone or end at a speaker, so a conversation
	// held in writing runs none of them.
	if !a.options.Text {
		p.running.Add(3)
		go a.consumeTTS(p, prep.tts)
		go a.consumeCadence(p)
		go a.consumePresence(p)
	}
}

// startNative runs a speech-to-speech model on the session opened for it. A prepared
// harness replaces the one the agent had; without one, the agent keeps its own.
func (a *Agent) startNative(prep *prepared) {
	p := newPipeline(a.ctx, true)
	var drained chan struct{}
	a.mu.Lock()
	a.pipe = p
	a.sts = prep.sts
	if prep.harness != nil {
		drained = make(chan struct{})
		a.harness, a.harnessDrained = prep.harness, drained
	}
	if a.arrivals == nil {
		a.arrivals = map[string]struct{}{}
	}
	a.voicePrompt, a.performs = "", false
	a.nativeListening, a.nativeAwaiting = false, false
	a.mu.Unlock()
	a.nativeMode.Store(true)

	if drained != nil {
		a.running.Add(1)
		go a.consumeHarness(prep.harness, drained)
	}
	events := make(chan sts.Event, sts.EmitterBuffer)
	p.running.Add(3)
	go a.receiveSTS(p, prep.sts.Events(), events)
	go a.consumeSTS(p, events)
	go a.consumeNativePresence(p)
}

// stopPipeline closes the running pipeline's sessions and waits for its goroutines. The
// harness survives when kept, and the call is never left.
func (a *Agent) stopPipeline(keepHarness bool) []error {
	p, failures := a.releasePipeline(keepHarness)
	a.mu.Lock()
	if !keepHarness {
		a.harness, a.harnessDrained = nil, nil
	}
	a.llm, a.tts, a.sts, a.replies, a.pipe = nil, nil, nil, nil, nil
	a.mu.Unlock()
	if p != nil {
		p.running.Wait()
	}
	return failures
}

// releasePipeline cancels the running pipeline and closes its sessions, in the order that
// lets each of its goroutines run out of work. It does not wait for them.
func (a *Agent) releasePipeline(keepHarness bool) (*pipeline, []error) {
	a.mu.Lock()
	p := a.pipe
	listeners := make([]*sttrouter.Session, 0, len(a.listeners))
	for _, listener := range a.listeners {
		listeners = append(listeners, listener)
	}
	a.listeners = map[string]*sttrouter.Session{}
	conversation, drained := a.harness, a.harnessDrained
	model, voice, speech, replies := a.llm, a.tts, a.sts, a.replies
	a.mu.Unlock()

	if p != nil {
		p.cancel()
	}
	var failures []error
	for _, listener := range listeners {
		if err := listener.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close stt: %w", err))
		}
	}
	// The harness goes before the model: it owns the subagent, and abandoning work
	// nobody will hear is the last useful thing either of them does. Its consumer is
	// waited on here rather than at the end, so what it abandoned is still reported: the
	// events channel is about to close.
	if !keepHarness && conversation != nil {
		if err := conversation.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close harness: %w", err))
		}
		if drained != nil {
			<-drained
		}
	}
	// Closing the model abandons every reply still being generated, which is what lets the
	// goroutine draining each one reach the end of its stream. Only once they have all
	// stopped can the channel they share close, and only then does the speaking goroutine
	// run out of work.
	if model != nil {
		if err := model.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close llm: %w", err))
		}
	}
	a.pumps.Wait()
	if replies != nil {
		close(replies)
	}
	if voice != nil {
		if err := voice.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close tts: %w", err))
		}
	}
	// Closing the native session ends its event stream, which is what lets its consumer
	// run out of work.
	if speech != nil {
		if err := speech.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close sts: %w", err))
		}
	}
	return p, failures
}

// openCascade opens the conversation model, the flow controller, the harness and the
// voice a cascade runs on.
func (a *Agent) openCascade(s Settings) (*prepared, error) {
	prep := &prepared{}
	var err error
	if prep.llm, err = a.startLLM(s.LLMTarget); err != nil {
		return nil, fmt.Errorf("agent: start llm: %w", err)
	}
	// Flow decisions use their own fast-model session so deciding whether speech is
	// complete never competes with the reply being streamed to the voice. It routes to a
	// non-thinking model of its own, since a decision this small has nothing to think about
	// and thinking would only add latency to every turn the caller waits through.
	if prep.controller, err = a.startLLM(s.controller()); err != nil {
		prep.close()
		return nil, fmt.Errorf("agent: start flow controller: %w", err)
	}
	prep.harness, err = harness.New(harness.Options{
		Text:         a.options.Text,
		Model:        prep.llm,
		Controller:   prep.controller,
		OpenSubagent: a.openSubagent(s.SubagentTarget),
		Capture:      a.captureVideo,
		Skills:       a.options.Skills,
		Tools:        a.availableTools(),
		Sandbox:      a.options.Sandbox,
		Tasks:        a.options.Tasks,
		MaxTokens:    a.options.MaxTokens,
		Overwrites:   s.Overwrites,
		CacheKey:     a.options.ConfigID,
		Logger:       a.logger,
	})
	if err != nil {
		prep.harness = nil
		prep.close()
		return nil, err
	}
	if !a.options.Text {
		if prep.tts, err = a.startVoice(s); err != nil {
			prep.close()
			return nil, fmt.Errorf("agent: start tts: %w", err)
		}
	}
	return prep, nil
}

// openNative opens the speech-to-speech model a native agent runs on, and the harness its
// delegated work runs in when it has a subagent.
func (a *Agent) openNative(s Settings) (*prepared, error) {
	prep := &prepared{}
	if subagent := a.openSubagent(s.SubagentTarget); subagent != nil {
		var err error
		prep.harness, err = harness.New(harness.Options{
			OpenSubagent: subagent, Capture: a.captureVideo, Skills: a.options.Skills,
			Sandbox: a.options.Sandbox, Tasks: a.options.Tasks, Logger: a.logger,
		})
		if err != nil {
			return nil, err
		}
	}
	var err error
	if prep.sts, err = a.openSpeech(s, prep.harness != nil); err != nil {
		prep.close()
		return nil, err
	}
	return prep, nil
}

// openSpeech opens a speech-to-speech session told everything it needs to carry on the
// conversation, the conversation so far included.
//
// Both transcripts are asked for, and asked for as terms: what the caller said and what
// the model said back are what the history, the chat log, the review and memory are all
// built on, so a model that cannot write them down is not a candidate.
func (a *Agent) openSpeech(s Settings, delegating bool) (*stsrouter.Session, error) {
	tools, err := a.nativeTools(delegating)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	instructions := a.nativeInstructions(delegating) + transcript(a.history)
	a.mu.Unlock()

	on := true
	session, err := a.options.STS.Start(a.ctx, stsrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        s.STSTarget,
		LanguageHints: a.options.LanguageHints,
		Tools:         tools,
		Options: options.STS{
			Instructions:     instructions,
			Voice:            s.Voice,
			InputTranscript:  &on,
			OutputTranscript: &on,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("agent: start sts: %w", err)
	}
	return session, nil
}

func (a *Agent) startLLM(target string) (*llmrouter.Session, error) {
	return a.options.LLM.Start(a.ctx, llmrouter.Request{
		CustomerID:    a.options.CustomerID,
		Caller:        a.options.Caller,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        target,
		LanguageHints: a.options.LanguageHints,
	})
}

// answers asks the model for one token, so a model that cannot answer is refused while the
// caller can still be told rather than on the next turn.
//
// Opening a session proves only that the target routes somewhere: nothing is sent, so a
// deployment whose key is rejected passes and the agent then goes quiet mid-call, which is
// the one failure nobody in the call can see. A word of one model's cheapest output is
// worth not finding out that way.
func answers(ctx context.Context, session *llmrouter.Session) error {
	stream, err := session.Create(ctx, llm.ResponseParams{
		Instructions:    "Reply with the word ok.",
		Input:           []llm.Message{{Role: llm.User, Content: "ok"}},
		MaxOutputTokens: 16,
	})
	if err != nil {
		return err
	}
	defer stream.Close()
	for stream.Next() {
	}
	return stream.Err()
}

func (a *Agent) startVoice(s Settings) (*ttsrouter.Session, error) {
	return a.options.TTS.Start(a.ctx, ttsrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        s.TTSTarget,
		LanguageHints: a.options.LanguageHints,
		Voice:         s.Voice,
	})
}

func (a *Agent) startListener(target string) (*sttrouter.Session, error) {
	return a.options.STT.Start(a.ctx, sttrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        target,
		LanguageHints: a.options.LanguageHints,
		Keyterms:      a.options.Keyterms,
	})
}

func (a *Agent) settingsLocked() Settings {
	return Settings{
		LLMTarget:        a.options.LLMTarget,
		ControllerTarget: a.options.ControllerTarget,
		STTTarget:        a.options.STTTarget,
		TTSTarget:        a.options.TTSTarget,
		STSTarget:        a.options.STSTarget,
		SubagentTarget:   a.options.SubagentTarget,
		Voice:            a.options.Voice,
		Overwrites:       a.options.Overwrites,
	}
}

// modelsChanged reports what the session runs on now.
func (a *Agent) modelsChanged() ModelsChanged {
	a.mu.Lock()
	defer a.mu.Unlock()
	changed := ModelsChanged{At: time.Now(), Native: a.pipe != nil && a.pipe.native, Voice: a.options.Voice}
	if a.llm != nil {
		changed.LLM = a.llm.Provider() + "/" + a.llm.Model()
	}
	if a.tts != nil {
		changed.TTS = a.tts.Provider() + "/" + a.tts.Model()
		if voice := a.tts.Voice(); voice != "" {
			changed.Voice = voice
		}
	}
	if a.sts != nil {
		changed.STS = a.sts.Provider() + "/" + a.sts.Model()
		if voice := a.sts.Voice(); voice != "" {
			changed.Voice = voice
		}
	}
	if !changed.Native && !a.options.Text {
		changed.STT = a.options.STTTarget
	}
	if a.harness != nil {
		if subagent := a.harness.Subagent(); subagent != nil {
			changed.Subagent = subagent.Provider() + "/" + subagent.Model()
		} else {
			changed.Subagent = a.options.SubagentTarget
		}
	}
	return changed
}

const (
	// transcriptMessages and transcriptChars cap the conversation a speech-to-speech model
	// is told about when it takes over, keeping the newest.
	transcriptMessages = 40
	transcriptChars    = 8000
)

// transcript renders the conversation so far for a speech-to-speech model taking over
// mid-call. Such a model takes no history, only instructions, so this is how it knows
// what it is continuing. Empty when nothing has been said yet.
func transcript(history []llm.Message) string {
	var lines []string
	length := 0
	for i := len(history) - 1; i >= 0 && len(lines) < transcriptMessages; i-- {
		message := history[i]
		text := strings.TrimSpace(llm.TextOf(message.Parts))
		if text == "" {
			text = strings.TrimSpace(message.Content)
		}
		var speaker string
		switch message.Role {
		case llm.User:
			speaker = "Caller"
		case llm.Assistant:
			speaker = "You"
		default:
			continue
		}
		if text == "" {
			continue
		}
		line := speaker + ": " + text
		if length+len(line) > transcriptChars {
			break
		}
		length += len(line)
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		return ""
	}
	for i, j := 0, len(lines)-1; i < j; i, j = i+1, j-1 {
		lines[i], lines[j] = lines[j], lines[i]
	}
	return "\n\nThe conversation so far, which you are continuing. Do not greet the caller again:\n" +
		strings.Join(lines, "\n")
}
