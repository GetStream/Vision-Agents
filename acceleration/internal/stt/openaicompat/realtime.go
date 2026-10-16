package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	realtimeQueueSize           = 64
	realtimeWriteTimeout        = 10 * time.Second
	realtimeHandshakeTimeout    = 10 * time.Second
	realtimeAppendBytes         = 9_600 // 200 ms at 24 kHz PCM16.
	realtimeMinTurnBytes        = 4_800 // The endpoint requires at least 100 ms per commit.
	realtimeMaxOutstandingTurns = 128
)

var (
	realtimeCloseTimeout = 10 * time.Second
)

// Realtime streams PCM to an OpenAI-compatible transcription WebSocket. The default
// router config keeps this path disabled until the model entry sets realtime: true.
type Realtime struct {
	endpoint  string
	apiKey    string
	model     string
	language  string
	languages []string
	settings  TranscriptionSettings

	emitter  *stt.Emitter
	commands chan realtimeCommand
	done     chan struct{}

	mu       sync.Mutex
	started  bool
	ready    bool
	conn     *websocket.Conn
	lifeCtx  context.Context
	cancel   context.CancelFunc
	closeErr error
	closeOne sync.Once
	closing  atomic.Bool

	sendMu            sync.Mutex
	endpointer        endpointer
	resampler         resampler
	appendBuffer      []byte
	appendTurnID      int64
	appendParticipant stt.Participant
}

// NewRealtime validates the configured endpoint and credentials for the streaming path.
func NewRealtime(options Options) (*Realtime, error) {
	baseURL, apiKey := endpointCredentials(options)
	options.BaseURL = strings.TrimSpace(baseURL)
	options.APIKey = strings.TrimSpace(apiKey)
	if options.BaseURL == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: base URL is required (set OPENAI_STT_BASE_URL)"))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if len(options.Model) > 128 || strings.ContainsAny(options.Model, "\r\n") || strings.TrimSpace(options.Model) == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: model must be a non-empty identifier of at most 128 bytes"))
	}
	if options.Settings.Language != "" && len(options.Settings.Languages) > 0 {
		return nil, stack.Wrap(errors.New("openai-compatible: language and languages settings cannot both be set"))
	}
	if hasBatchOnlyRealtimeSettings(options.Settings) {
		return nil, stack.Wrap(errors.New("openai-compatible: batch-only transcription settings are not supported for realtime"))
	}
	endpoint, err := realtimeEndpoint(options.BaseURL)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("openai-compatible: configure realtime endpoint: %w", err))
	}

	return &Realtime{
		endpoint:  endpoint,
		apiKey:    options.APIKey,
		model:     options.Model,
		language:  options.Language,
		languages: append([]string(nil), options.Languages...),
		settings: TranscriptionSettings{
			Prompt:    options.Settings.Prompt,
			Language:  options.Settings.Language,
			Languages: append([]string(nil), options.Settings.Languages...),
			Keywords:  append([]string(nil), options.Settings.Keywords...),
			Delay:     options.Settings.Delay,
			Include:   append([]string(nil), options.Settings.Include...),
		},
		emitter:      stt.NewEmitter(64),
		commands:     make(chan realtimeCommand, realtimeQueueSize),
		done:         make(chan struct{}),
		appendBuffer: make([]byte, 0, realtimeAppendBytes*2),
	}, nil
}

func hasBatchOnlyRealtimeSettings(settings TranscriptionSettings) bool {
	return settings.Temperature != nil || settings.ResponseFormat != "" ||
		len(settings.TimestampGranularities) > 0 || len(settings.ChunkingStrategy) > 0 ||
		len(settings.KnownSpeakerNames) > 0 || len(settings.KnownSpeakerReferences) > 0 ||
		settings.Stream != nil
}

// Start opens the socket and waits until the transcription session is configured.
func (r *Realtime) Start(ctx context.Context) error {
	r.mu.Lock()
	if r.started {
		r.mu.Unlock()
		return stack.Wrap(errors.New("openai-compatible: realtime session already started"))
	}
	if r.closing.Load() {
		r.mu.Unlock()
		return stack.Wrap(errors.New("openai-compatible: realtime session is closed"))
	}
	r.started = true
	r.lifeCtx, r.cancel = context.WithCancel(ctx)
	lifeCtx := r.lifeCtx
	r.mu.Unlock()

	dialer := websocket.Dialer{HandshakeTimeout: realtimeHandshakeTimeout}
	header := make(http.Header)
	if r.apiKey != "" {
		header.Set("Authorization", "Bearer "+r.apiKey)
	}
	conn, response, err := dialer.DialContext(lifeCtx, r.endpoint, header)
	if err != nil {
		r.cancel()
		if response != nil {
			return stack.Wrap(fmt.Errorf("openai-compatible: realtime dial: HTTP %d", response.StatusCode))
		}
		return stack.Wrap(fmt.Errorf("openai-compatible: realtime dial: %w", err))
	}
	r.mu.Lock()
	r.conn = conn
	r.mu.Unlock()
	stopClose := context.AfterFunc(lifeCtx, func() {
		_ = conn.Close()
		r.emitter.Close()
	})
	conn.SetReadLimit(1 << 20)

	if err := r.handshake(conn); err != nil {
		stopClose()
		r.cancel()
		_ = conn.Close()
		r.emitter.Close()
		return stack.Wrap(err)
	}
	r.mu.Lock()
	r.ready = true
	r.mu.Unlock()
	r.emitter.Send(stt.Connected{Provider: ProviderName, Model: r.model, At: time.Now()})
	go r.run(lifeCtx, conn, stopClose)
	return nil
}

func (r *Realtime) handshake(conn *websocket.Conn) error {
	deadline := time.Now().Add(realtimeHandshakeTimeout)
	if err := conn.SetReadDeadline(deadline); err != nil {
		return fmt.Errorf("openai-compatible: realtime handshake deadline: %w", err)
	}
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("openai-compatible: realtime handshake read: %w", err)
		}
		event, err := decodeRealtimeEvent(data)
		if err != nil {
			return err
		}
		switch event.Type {
		case "session.created":
			if err := writeRealtimeJSON(conn, sessionUpdate(r.model, r.language, r.languages, r.settings)); err != nil {
				return fmt.Errorf("openai-compatible: realtime session update: %w", err)
			}
		case "session.updated":
			_ = conn.SetReadDeadline(time.Time{})
			return nil
		case "error":
			return errors.New("openai-compatible: realtime session setup was refused")
		}
	}
}

// ProcessAudio adds one mono 16 kHz PCM chunk to the live stream.
func (r *Realtime) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return stack.Wrap(fmt.Errorf("openai-compatible: process audio: %w", err))
	}
	r.sendMu.Lock()
	defer r.sendMu.Unlock()
	if r.closing.Load() {
		return errors.New("openai-compatible: realtime session is closed")
	}
	r.mu.Lock()
	ready, lifeCtx := r.ready, r.lifeCtx
	r.mu.Unlock()
	if !ready || lifeCtx == nil {
		return errors.New("openai-compatible: realtime session is not started")
	}
	if err := lifeCtx.Err(); err != nil {
		return stack.Wrap(fmt.Errorf("openai-compatible: realtime session stopped: %w", err))
	}
	if r.endpointer.inTurn && r.endpointer.participant != participant {
		return errors.New("openai-compatible: participant changed during an active utterance")
	}
	if r.endpointer.pendingSamples > 0 && r.endpointer.pendingParticipant != participant {
		return errors.New("openai-compatible: participant changed within an audio frame")
	}

	var processErr error
	r.endpointer.Push(pcm.Samples, participant, func(frame endpointFrame) {
		if processErr == nil {
			processErr = r.acceptFrame(lifeCtx, frame)
		}
	})
	return processErr
}

func (r *Realtime) acceptFrame(ctx context.Context, frame endpointFrame) error {
	if frame.include {
		if err := r.appendSamples(ctx, frame.utterance, frame.participant, frame.samples); err != nil {
			return err
		}
	}
	if frame.commit {
		meta := turnMetadata{
			participant: frame.participant,
			utterance:   frame.utterance,
			samples:     frame.turnSamples,
		}
		r.appendBuffer = r.resampler.Flush(r.appendBuffer)
		if err := r.flushAppend(ctx); err != nil {
			return err
		}
		if err := r.sendCommand(ctx, realtimeCommand{kind: commandCommit, meta: meta}); err != nil {
			return err
		}
		r.resampler = resampler{}
	}
	return nil
}

func (r *Realtime) appendSamples(ctx context.Context, turnID int64, participant stt.Participant, samples []int16) error {
	if r.appendTurnID != 0 && r.appendTurnID != turnID {
		if err := r.flushAppend(ctx); err != nil {
			return err
		}
	}
	r.appendTurnID = turnID
	r.appendParticipant = participant
	r.appendBuffer = r.resampler.Append(r.appendBuffer, samples)
	for len(r.appendBuffer) >= realtimeAppendBytes {
		chunk := append([]byte(nil), r.appendBuffer[:realtimeAppendBytes]...)
		copy(r.appendBuffer, r.appendBuffer[realtimeAppendBytes:])
		r.appendBuffer = r.appendBuffer[:len(r.appendBuffer)-realtimeAppendBytes]
		if err := r.sendCommand(ctx, realtimeCommand{
			kind:   commandAudio,
			turnID: turnID,
			pcm:    chunk,
			meta: turnMetadata{
				participant: participant,
				utterance:   turnID,
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

func (r *Realtime) flushAppend(ctx context.Context) error {
	if len(r.appendBuffer) == 0 {
		return nil
	}
	chunk := append([]byte(nil), r.appendBuffer...)
	turnID := r.appendTurnID
	participant := r.appendParticipant
	r.appendBuffer = r.appendBuffer[:0]
	r.appendTurnID = 0
	r.appendParticipant = stt.Participant{}
	return r.sendCommand(ctx, realtimeCommand{
		kind:   commandAudio,
		turnID: turnID,
		pcm:    chunk,
		meta:   turnMetadata{participant: participant, utterance: turnID},
	})
}

func (r *Realtime) sendCommand(ctx context.Context, command realtimeCommand) error {
	select {
	case r.commands <- command:
		return nil
	case <-ctx.Done():
		return stack.Wrap(fmt.Errorf("openai-compatible: realtime session stopped: %w", ctx.Err()))
	case <-r.done:
		return errors.New("openai-compatible: realtime session stopped")
	}
}

// Events returns connection, transcript and error events.
func (r *Realtime) Events() <-chan stt.Event { return r.emitter.Events() }

// Close commits the final utterance and waits a bounded time for its transcript.
func (r *Realtime) Close() error {
	r.closeOne.Do(func() { r.closeErr = r.close() })
	return r.closeErr
}

func (r *Realtime) close() error {
	r.closing.Store(true)
	timedOut := make(chan struct{})
	timeout := time.AfterFunc(realtimeCloseTimeout, func() {
		close(timedOut)
		r.mu.Lock()
		cancel, conn := r.cancel, r.conn
		r.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		if conn != nil {
			_ = conn.Close()
		}
		r.emitter.Close()
	})
	defer timeout.Stop()
	stopOnTimeout := func() error {
		select {
		case <-timedOut:
			return stack.Wrap(errors.New("openai-compatible: realtime close timed out"))
		default:
			return nil
		}
	}
	r.sendMu.Lock()
	defer r.sendMu.Unlock()
	r.mu.Lock()
	ready, lifeCtx, cancel, conn := r.ready, r.lifeCtx, r.cancel, r.conn
	r.mu.Unlock()
	if !ready || lifeCtx == nil {
		if cancel != nil {
			cancel()
		}
		if conn != nil {
			_ = conn.Close()
		}
		r.emitter.Close()
		return stopOnTimeout()
	}

	var processErr error
	r.endpointer.Close(func(frame endpointFrame) {
		if processErr == nil {
			processErr = r.acceptFrame(lifeCtx, frame)
		}
	})
	if processErr == nil {
		processErr = r.sendCommand(lifeCtx, realtimeCommand{kind: commandClose})
	}
	if processErr != nil {
		if cancel != nil {
			cancel()
		}
		if conn != nil {
			_ = conn.Close()
		}
		r.emitter.Close()
		if err := stopOnTimeout(); err != nil {
			return err
		}
		return stack.Wrap(processErr)
	}

	select {
	case <-r.done:
		return stopOnTimeout()
	case <-timedOut:
		return stack.Wrap(errors.New("openai-compatible: realtime close timed out"))
	}
}

func (r *Realtime) Provider() string { return ProviderName }

func (r *Realtime) Model() string { return r.model }

func writeRealtimeJSON(conn *websocket.Conn, value any) error {
	if err := conn.SetWriteDeadline(time.Now().Add(realtimeWriteTimeout)); err != nil {
		return err
	}
	return conn.WriteJSON(value)
}
