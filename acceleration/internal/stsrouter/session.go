package stsrouter

import (
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// Session is a live conversation attached to one customer. It forwards the provider's
// events and records a stat row per completed reply on the way past.
//
// It also keeps the one promise the contract makes about barge-in: once a reply has been
// reported interrupted, or the caller has asked for it to be, no more of its audio gets
// through. A provider learns of an interrupt one round trip after the caller, and the
// chunks in that gap would otherwise play as a tail on the words the caller cut off.
type Session struct {
	provider sts.STS
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder
	// backend records the text model a voice hands its reasoning to, which is billed as
	// that model.
	backend *routing.Recorder
	// id names the session on its own rows, as their operation, and on the rows of work it
	// set going, as their parent_session.
	id string

	events chan sts.Event

	mu sync.Mutex
	// inFlight tracks replies that have not been settled into a stat row yet.
	inFlight map[string]*reply
	settled  map[string]struct{}
	// billed is the backend responses and transcribed turns already recorded, so one reported
	// twice is a row once: the usage id keeps it out of the table, and this out of the live
	// spend, which adds every row it is handed.
	billed map[string]struct{}
	// heardMs is the caller's audio forwarded since the last reply settled, which is the
	// audio the next reply is billed against, and typed the typed turns sent since.
	heardMs float64
	typed   int64
	// reportedSeconds is the vendor's last running total of the seconds the session has been
	// open, and billedSeconds what the session's rows have billed of them.
	reportedSeconds, billedSeconds int64
	// current is the generation of the reply in flight, and staleBelow the highest one
	// whose audio is no longer wanted.
	current    int
	staleBelow int

	closeOnce sync.Once
}

// reply is what the session remembers about one response until it completes.
type reply struct {
	// startedAt is when the model began, so the stat row is stamped with when the customer
	// was answered rather than when the audio finished.
	startedAt time.Time
	// errorCode is set when the provider reported a failure for this reply. It turns the
	// completion into a failed row rather than adding a second one.
	errorCode string
}

func newSession(
	provider sts.STS,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder, backend *routing.Recorder,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		backend:  backend,
		id:       uuid.NewString(),
		events:   make(chan sts.Event, sts.EmitterBuffer),
		inFlight: map[string]*reply{},
		settled:  map[string]struct{}{},
		billed:   map[string]struct{}{},
	}
	go session.forward()
	return session
}

// ProcessAudio streams the caller's speech to the selected provider. Only audio the provider
// took is counted toward the next reply.
func (s *Session) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	if err := s.provider.ProcessAudio(pcm, participant); err != nil {
		return err
	}
	s.mu.Lock()
	s.heardMs += pcm.DurationMs()
	s.mu.Unlock()
	return nil
}

// SendText injects a typed turn, which the next row carries as a message: some vendors bill
// each one.
func (s *Session) SendText(text string, participant sts.Participant) error {
	if err := s.provider.SendText(text, participant); err != nil {
		return err
	}
	s.mu.Lock()
	s.typed++
	s.mu.Unlock()
	return nil
}

// SendFrame offers the model a still image.
func (s *Session) SendFrame(frame llm.ImagePart) error { return s.provider.SendFrame(frame) }

// SetInstructions changes the system prompt, where the model allows it.
func (s *Session) SetInstructions(text string) error { return s.provider.SetInstructions(text) }

// SetTools replaces what the model may call, where the model allows it.
func (s *Session) SetTools(tools []llm.Tool) error { return s.provider.SetTools(tools) }

// Answer returns what a tool produced.
func (s *Session) Answer(callID string, output string, err error) error {
	return s.provider.Answer(callID, output, err)
}

// Prompt asks the model to speak, guided by the text.
func (s *Session) Prompt(text string) error { return s.provider.Prompt(text) }

// Interrupt stops the reply in flight. Its audio stops being forwarded at once, before the
// provider has heard about it, since the caller has already stopped listening.
func (s *Session) Interrupt(playedMs int) error {
	s.mu.Lock()
	if s.current > s.staleBelow {
		s.staleBelow = s.current
	}
	s.mu.Unlock()
	return s.provider.Interrupt(playedMs)
}

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan sts.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Voice is the voice speaking. Empty when the vendor picked it and cannot say which.
func (s *Session) Voice() string {
	if voiced, ok := s.provider.(sts.Voiced); ok {
		return voiced.Voice()
	}
	return ""
}

// SampleRate is the rate the model speaks at.
func (s *Session) SampleRate() int { return s.provider.SampleRate() }

// Capabilities is what the model serving this session can be asked for.
func (s *Session) Capabilities() sts.Capabilities { return s.provider.Capabilities() }

// Quote is what that usage of this session's provider costs on its totals, as a reply whose
// vendor reports no split by modality is priced, so a caller can report a cost without
// reaching for the router's prices. A row with a split is priced by the split instead.
//
// A vendor that bills the session's seconds rather than its replies (SessionBilled) has its
// replies recorded free, so a reply quotes nothing: what such a session costs is its seconds,
// which only the vendor's reports say, as they arrive, and its rows bill.
func (s *Session) Quote(usage routing.Usage) pricing.Quote {
	if s.provider.Capabilities().SessionBilled {
		return pricing.Quote{}
	}
	return s.recorder.Quote(s.config, usage)
}

// Close ends the session. The event channel closes once the provider's events are drained.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.provider.Close() })
	return err
}

// forward relays provider events, recording statistics and dropping stale audio as they
// pass.
func (s *Session) forward() {
	defer close(s.events)

	for event := range s.provider.Events() {
		if s.observe(event) {
			s.events <- event
		}
	}
	s.recordTrailing()
}

// recordTrailing records what the session streamed after its last reply settled, which no
// event reports: the caller's audio and the typed turns since. It is recorded only where the
// price list bills it, the vendors priced on the caller's audio or on each message, and is
// marked as the router's own count.
func (s *Session) recordTrailing() {
	s.mu.Lock()
	heardMs, typed := s.heardMs, s.typed
	s.heardMs, s.typed = 0, 0
	s.mu.Unlock()

	stat := routing.Stat{
		Owner:       s.owner,
		StartedAt:   time.Now().UTC(),
		OperationID: s.id,
		Usage:       routing.Usage{AudioMs: int64(heardMs)},
		Success:     true,
		Unmeasured:  true,
	}
	stat.Use(pricing.Quantities{pricing.Messages: typed})
	if stat.AudioMs == 0 && typed == 0 || !s.recorder.QuoteStat(s.config, stat).Rated {
		return
	}
	stat.MeasuredByRouter()
	s.recorder.Record(s.config, stat)
}

// parentSession is the attribute naming the session whose rows a row of delegated or added
// work belongs with.
const parentSession = "parent_session"

// observe records what an event means for the books and reports whether it is forwarded.
func (s *Session) observe(event sts.Event) bool {
	switch typed := event.(type) {
	case sts.ResponseStarted:
		s.mu.Lock()
		if _, done := s.settled[typed.ResponseID]; done {
			s.mu.Unlock()
			return false
		}
		s.inFlight[typed.ResponseID] = &reply{startedAt: typed.At}
		s.current = typed.Generation
		s.mu.Unlock()

	case sts.AudioChunk:
		s.mu.Lock()
		stale := typed.Generation <= s.staleBelow
		s.mu.Unlock()
		return !stale

	case sts.ResponseComplete:
		// One reply is one unit of billable work, the way one utterance is for a voice.
		// An interrupted one still cost the model's time and still cost money.
		settled, heardMs, typedTurns, fresh := s.settle(typed)
		if !fresh {
			return false
		}
		audioMs := int64(heardMs)
		carried := false
		switch {
		case typed.Usage.AudioBilled:
			// The vendor counts the audio it bills, replies heard over a gap in the caller's
			// stream included, so its count replaces the caller's audio.
			audioMs = typed.Usage.BilledAudioMs
		case s.provider.Capabilities().BilledAudio:
			// A reply cut off before the vendor counted it: the vendor's next count includes
			// what the caller said meanwhile, and if none comes the session's last row bills
			// it, so it is carried rather than billed here.
			audioMs, carried = 0, true
			s.mu.Lock()
			s.heardMs += heardMs
			s.mu.Unlock()
		}
		stat := routing.Stat{
			Owner:       s.owner,
			StartedAt:   settled.startedAt,
			OperationID: s.id,
			Usage: routing.Usage{
				AudioMs:           audioMs,
				InputTokens:       typed.Usage.InputTokens,
				CachedInputTokens: typed.Usage.CachedInputTokens,
				OutputTokens:      typed.Usage.OutputTokens,
			},
			Meters:     modalityMeters(typed.Usage),
			Attributes: usageAttributes(typed.Usage),
			// Time to first byte is what the caller actually waited for; the rest of the
			// reply arrives while they are already hearing it.
			LatencyMs: typed.TimeToFirstByteMs,
			Success:   settled.errorCode == "",
			ErrorCode: settled.errorCode,
			// A vendor that bills the session's seconds bills them in SessionUsage, so the
			// reply is kept for its timing and not billed again.
			Free: s.provider.Capabilities().SessionBilled,
		}
		stat.Use(pricing.Quantities{pricing.Messages: typedTurns})
		if carried {
			// Its zero is the router's to give, not a vendor that said nothing.
			stat.MeasuredByRouter()
		}
		s.recorder.Record(s.config, stat)

	case sts.Error:
		// A failure that names a reply is settled by that reply's completion, so one
		// reply is still one row. Anything else is a session-level failure.
		if typed.ResponseID != "" && s.fail(typed.ResponseID, errorCode(typed)) {
			return true
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:       s.owner,
			StartedAt:   time.Now().UTC(),
			OperationID: s.id,
			Success:     false,
			ErrorCode:   errorCode(typed),
		})

	case sts.SessionUsage:
		s.billSeconds(typed.Seconds)
		return false

	case sts.BackendUsage:
		s.billBackend(typed)
		return false

	case sts.TranscriptionUsage:
		s.billTranscription(typed)
		return false
	}
	return true
}

// billSeconds records the seconds the vendor says the session has been open since its last
// report, so a session that ends without a last word loses at most the time since. Each row's
// usage id is the session and the seconds billed by the end of it, so a report repeated bills
// nothing more. A total lower than the last is a vendor session started over, counted from
// zero.
func (s *Session) billSeconds(reported int64) {
	s.mu.Lock()
	grown := reported - s.reportedSeconds
	if grown < 0 {
		grown = reported
	}
	s.reportedSeconds = reported
	s.billedSeconds += max(grown, 0)
	billed := s.billedSeconds
	s.mu.Unlock()
	if grown <= 0 {
		return
	}
	// Settle refuses only work with no usage id, and every row here has one.
	_ = routing.Settle(routing.Meter{
		Recorder:    s.recorder,
		Config:      s.config,
		Owner:       s.owner,
		UsageID:     routing.UsageID("sts_session_seconds", s.id, strconv.FormatInt(billed, 10)),
		OperationID: s.id,
		Unmeasured:  true,
	}, pricing.Quantities{pricing.SessionSeconds: grown})
}

// billBackend records a response of the text model the voice handed its reasoning to, as an
// LLM row of that model, priced as that model is, and billed once by the session and its
// response id.
func (s *Session) billBackend(used sts.BackendUsage) {
	if !s.first("backend:" + used.ResponseID) {
		return
	}
	attributes := map[string]string{parentSession: s.id}
	if used.ServiceTier != "" {
		attributes["service_tier"] = used.ServiceTier
	}
	if reasoning := used.Usage.OutputTokensDetails.ReasoningTokens; reasoning > 0 {
		attributes["reasoning_tokens"] = strconv.FormatInt(reasoning, 10)
	}
	_ = routing.Settle(routing.Meter{
		Recorder:   s.backend,
		Config:     routing.ProviderConfig{Provider: s.config.Provider, Model: used.Model},
		Owner:      s.owner,
		UsageID:    routing.UsageID("sts_backend", s.id, used.ResponseID),
		Attributes: attributes,
		// The voice relays the backend's responses without timing them.
		Unmeasured: true,
	}, llm.Response{ID: used.ResponseID, Usage: used.Usage, UsageReported: true})
}

// billTranscription records writing down a caller turn as a row of the transcriber, on this
// modality's price list, billed once by the turn's item id: by its tokens, or by the audio it
// heard for a transcriber billed by the second. A transcriber the price list does not rate
// for what it reported is still a row, counted unpriced or missing and alarmed, not dropped.
func (s *Session) billTranscription(used sts.TranscriptionUsage) {
	if !s.first("transcription:" + used.ItemID) {
		return
	}
	quantities := pricing.Quantities{
		pricing.InputTokens:  used.Usage.InputTokens,
		pricing.OutputTokens: used.Usage.OutputTokens,
		pricing.AudioMs:      used.AudioMs,
	}
	for meter, quantity := range modalityMeters(used.Usage) {
		quantities[meter] = quantity
	}
	_ = routing.Settle(routing.Meter{
		Recorder:   s.recorder,
		Config:     routing.ProviderConfig{Provider: s.config.Provider, Model: used.Model},
		Owner:      s.owner,
		UsageID:    routing.UsageID("sts_transcription", s.id, used.ItemID),
		Attributes: map[string]string{parentSession: s.id},
		Unmeasured: true,
	}, quantities)
}

// first reports whether this is the first time the session bills the work key names.
func (s *Session) first(key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, done := s.billed[key]; done {
		return false
	}
	s.billed[key] = struct{}{}
	return true
}

// fail marks a reply as failed, reporting whether it was still in flight.
func (s *Session) fail(responseID, code string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.inFlight[responseID]
	if !ok {
		return false
	}
	// The first failure is the one that explains the reply.
	if current.errorCode == "" {
		current.errorCode = code
	}
	return true
}

// settle takes a reply out of flight, so a late duplicate completion cannot bill twice, and
// claims the caller's audio heard and the turns typed since the last one. An interrupted reply
// also marks its generation stale, so audio still arriving for it is dropped. A reply the
// session never saw start is stamped with now.
func (s *Session) settle(complete sts.ResponseComplete) (reply, float64, int64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, done := s.settled[complete.ResponseID]; done {
		return reply{}, 0, 0, false
	}
	s.settled[complete.ResponseID] = struct{}{}
	heardMs, typed := s.heardMs, s.typed
	s.heardMs, s.typed = 0, 0
	if complete.Interrupted && complete.Generation > s.staleBelow {
		s.staleBelow = complete.Generation
	}

	current, ok := s.inFlight[complete.ResponseID]
	if !ok {
		return reply{startedAt: time.Now().UTC()}, heardMs, typed, true
	}
	delete(s.inFlight, complete.ResponseID)
	return reply{startedAt: current.startedAt.UTC(), errorCode: current.errorCode}, heardMs, typed, true
}

// modalityMeters is the reply's tokens by modality, which a vendor that bills each at its
// own rate reports and the price list prices in place of the totals. Each input modality is
// fresh, its cached part taken out, as the totals' input tokens are. Nil for a vendor that
// reports no split, which is priced on its totals.
func modalityMeters(usage sts.Usage) pricing.Quantities {
	meters := pricing.Quantities{}
	add := func(meter pricing.Meter, quantity int64) {
		if quantity > 0 {
			meters[meter] = quantity
		}
	}
	add(pricing.InputTextTokens, usage.InputTextTokens-usage.CachedInputTextTokens)
	add(pricing.InputAudioTokens, usage.InputAudioTokens-usage.CachedInputAudioTokens)
	add(pricing.InputImageTokens, usage.InputImageTokens-usage.CachedInputImageTokens)
	add(pricing.CachedInputTextTokens, usage.CachedInputTextTokens)
	add(pricing.CachedInputAudioTokens, usage.CachedInputAudioTokens)
	add(pricing.CachedInputImageTokens, usage.CachedInputImageTokens)
	add(pricing.OutputTextTokens, usage.OutputTextTokens)
	add(pricing.OutputAudioTokens, usage.OutputAudioTokens)
	if len(meters) == 0 {
		return nil
	}
	return meters
}

// usageAttributes explains the reply's tokens without pricing them: the thinking, which the
// output already includes, and the part of the prompt the vendor's split leaves out, which
// Gemini Live does not bill.
func usageAttributes(usage sts.Usage) map[string]string {
	attributes := map[string]string{}
	if usage.ReasoningTokens > 0 {
		attributes["reasoning_tokens"] = strconv.FormatInt(usage.ReasoningTokens, 10)
	}
	split := usage.InputTextTokens + usage.InputAudioTokens + usage.InputImageTokens
	if missing := usage.InputTokens - split; split > 0 && missing > 0 {
		attributes["prompt_tokens_not_in_details"] = strconv.FormatInt(missing, 10)
	}
	if len(attributes) == 0 {
		return nil
	}
	return attributes
}

func errorCode(failure sts.Error) string {
	if failure.Fatal {
		return "provider_fatal"
	}
	if failure.Context != "" {
		return failure.Context
	}
	return "provider_error"
}
