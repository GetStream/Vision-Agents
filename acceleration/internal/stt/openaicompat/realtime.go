package openaicompat

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	realtimeQueueSize        = 64
	realtimeWriteTimeout     = 10 * time.Second
	realtimeHandshakeTimeout = 10 * time.Second
	realtimeInferenceTimeout = 2 * time.Minute
	realtimeAppendBytes      = 9_600 // 200 ms at 24 kHz PCM16.
	realtimeMinTurnBytes     = 4_800 // The endpoint requires at least 100 ms per commit.
	realtimeMaxBufferedBytes = 24_000 * 2 * 60
)

var realtimeCloseTimeout = 10 * time.Second

type commandKind uint8

const (
	commandAudio commandKind = iota
	commandCommit
	commandClose
)

type realtimeCommand struct {
	kind   commandKind
	turnID int64
	pcm    []byte
	meta   turnMetadata
}

type turnMetadata struct {
	participant stt.Participant
	utterance   int64
	samples     int
}

type realtimeTurn struct {
	turnMetadata
	audio       [][]byte
	audioBytes  int
	ended       bool
	itemID      string
	committedAt time.Time
}

type readResult struct {
	event realtimeEvent
	err   error
}

type realtimeSession struct {
	provider *Realtime
	conn     *websocket.Conn
	order    []*realtimeTurn
	inFlight *realtimeTurn
	buffered int
	closing  bool
	timer    *time.Timer
	timeout  <-chan time.Time
}

// Realtime streams PCM to an OpenAI-compatible transcription WebSocket. The default
// router config keeps this path disabled until the model entry sets realtime: true.
type Realtime struct {
	endpoint string
	apiKey   string
	model    string
	language string

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

	sendMu       sync.Mutex
	endpointer   endpointer
	resampler    resampler
	appendBuffer []byte
	appendTurnID int64
}

// NewRealtime validates the configured endpoint and credentials for the streaming path.
func NewRealtime(options Options) (*Realtime, error) {
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnv)
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnv)
	}
	options.BaseURL = strings.TrimSpace(options.BaseURL)
	options.APIKey = strings.TrimSpace(options.APIKey)
	if options.BaseURL == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: base URL is required (set OPENAI_STT_BASE_URL)"))
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: API key is required (set OPENAI_STT_API_KEY)"))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if len(options.Model) > 128 || strings.ContainsAny(options.Model, "\r\n") || strings.TrimSpace(options.Model) == "" {
		return nil, stack.Wrap(errors.New("openai-compatible: model must be a non-empty identifier of at most 128 bytes"))
	}
	if options.Language != "" && !validLanguage(options.Language) {
		return nil, stack.Wrap(errors.New("openai-compatible: language must be a two-letter lowercase ISO 639-1 code"))
	}
	endpoint, err := realtimeEndpoint(options.BaseURL, options.Model)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("openai-compatible: configure realtime endpoint: %w", err))
	}

	return &Realtime{
		endpoint:     endpoint,
		apiKey:       options.APIKey,
		model:        options.Model,
		language:     options.Language,
		emitter:      stt.NewEmitter(64),
		commands:     make(chan realtimeCommand, realtimeQueueSize),
		done:         make(chan struct{}),
		appendBuffer: make([]byte, 0, realtimeAppendBytes*2),
	}, nil
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
	header := http.Header{"Authorization": []string{"Bearer " + r.apiKey}}
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
			if err := writeRealtimeJSON(conn, sessionUpdate(r.model, r.language)); err != nil {
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
		if err := r.appendSamples(ctx, frame.utterance, frame.samples); err != nil {
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

func (r *Realtime) appendSamples(ctx context.Context, turnID int64, samples []int16) error {
	if r.appendTurnID != 0 && r.appendTurnID != turnID {
		if err := r.flushAppend(ctx); err != nil {
			return err
		}
	}
	r.appendTurnID = turnID
	r.appendBuffer = r.resampler.Append(r.appendBuffer, samples)
	for len(r.appendBuffer) >= realtimeAppendBytes {
		chunk := append([]byte(nil), r.appendBuffer[:realtimeAppendBytes]...)
		copy(r.appendBuffer, r.appendBuffer[realtimeAppendBytes:])
		r.appendBuffer = r.appendBuffer[:len(r.appendBuffer)-realtimeAppendBytes]
		if err := r.sendCommand(ctx, realtimeCommand{kind: commandAudio, turnID: turnID, pcm: chunk}); err != nil {
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
	r.appendBuffer = r.appendBuffer[:0]
	r.appendTurnID = 0
	return r.sendCommand(ctx, realtimeCommand{kind: commandAudio, turnID: turnID, pcm: chunk})
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

func writeAudioAppend(conn *websocket.Conn, pcm []byte) error {
	return writeRealtimeJSON(conn, map[string]string{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(pcm),
	})
}

func (r *Realtime) readLoop(ctx context.Context, conn *websocket.Conn, results chan<- readResult) {
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			select {
			case results <- readResult{err: err}:
			case <-ctx.Done():
			}
			return
		}
		event, err := decodeRealtimeEvent(data)
		if err != nil {
			select {
			case results <- readResult{err: err}:
			case <-ctx.Done():
			}
			return
		}
		select {
		case results <- readResult{event: event}:
		case <-ctx.Done():
			return
		}
	}
}

func (s *realtimeSession) stopTimer() {
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	s.timeout = nil
}

func (s *realtimeSession) startTimer() {
	s.stopTimer()
	s.timer = time.NewTimer(realtimeInferenceTimeout)
	s.timeout = s.timer.C
}

func (s *realtimeSession) fail(contextName string, err error) {
	r := s.provider
	r.emitter.Send(stt.Error{Provider: ProviderName, Model: r.model, Err: err, Context: contextName, Fatal: true})
	r.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: r.model, Reason: "upstream connection ended", Clean: false, At: time.Now()})
}

// drain sends queued audio in order, then commits at most one completed turn. The
// upstream session accepts only one in-flight transcription at a time.
func (s *realtimeSession) drain() (bool, error) {
	if s.inFlight != nil {
		return false, nil
	}
	if len(s.order) > 0 {
		turn := s.order[0]
		for _, chunk := range turn.audio {
			if err := writeAudioAppend(s.conn, chunk); err != nil {
				return false, fmt.Errorf("openai-compatible: realtime audio append: %w", err)
			}
			turn.audioBytes += len(chunk)
			s.buffered -= len(chunk)
		}
		turn.audio = nil
		if !turn.ended {
			return false, nil
		}
		if turn.audioBytes < realtimeMinTurnBytes {
			padding := make([]byte, realtimeMinTurnBytes-turn.audioBytes)
			if err := writeAudioAppend(s.conn, padding); err != nil {
				return false, fmt.Errorf("openai-compatible: realtime audio padding: %w", err)
			}
		}
		if err := writeRealtimeJSON(s.conn, map[string]string{"type": "input_audio_buffer.commit"}); err != nil {
			return false, fmt.Errorf("openai-compatible: realtime audio commit: %w", err)
		}
		turn.committedAt = time.Now()
		s.inFlight = turn
		s.order = s.order[1:]
		s.startTimer()
		return false, nil
	}
	if s.closing {
		_ = s.conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(realtimeWriteTimeout))
		return true, nil
	}
	return false, nil
}

func (s *realtimeSession) command(command realtimeCommand) error {
	switch command.kind {
	case commandAudio:
		turn := findRealtimeTurn(s.order, command.turnID)
		if turn == nil {
			turn = &realtimeTurn{turnMetadata: turnMetadata{utterance: command.turnID}}
			s.order = append(s.order, turn)
		}
		turn.audio = append(turn.audio, command.pcm)
		s.buffered += len(command.pcm)
		if s.buffered > realtimeMaxBufferedBytes {
			return errors.New("openai-compatible: realtime audio backlog exceeded 60 seconds")
		}
	case commandCommit:
		turn := findRealtimeTurn(s.order, command.meta.utterance)
		if turn == nil {
			turn = &realtimeTurn{turnMetadata: command.meta}
			s.order = append(s.order, turn)
		} else {
			turn.turnMetadata = command.meta
		}
		turn.ended = true
	case commandClose:
		s.closing = true
	}
	return nil
}

func (s *realtimeSession) event(event realtimeEvent) error {
	switch event.Type {
	case "session.updated", "input_audio_buffer.speech_started":
		return nil
	case "input_audio_buffer.committed":
		if s.inFlight == nil || event.ItemID == "" {
			return errors.New("openai-compatible: realtime commit acknowledgement had no active item")
		}
		s.inFlight.itemID = event.ItemID
		return nil
	case "conversation.item.input_audio_transcription.completed":
		turn := s.inFlight
		if turn == nil || (turn.itemID != "" && event.ItemID != turn.itemID) {
			return errors.New("openai-compatible: realtime transcript did not match the active audio item")
		}
		if text := strings.TrimSpace(event.Transcript); text != "" {
			s.provider.emitter.Send(stt.Transcript{
				Participant:      turn.participant,
				Mode:             stt.ModeFinal,
				Utterance:        turn.utterance,
				Text:             text,
				Provider:         ProviderName,
				Model:            s.provider.model,
				ProcessingTimeMs: float64(time.Since(turn.committedAt).Microseconds()) / 1000,
				AudioDurationMs:  float64(turn.samples) * 1000 / stt.SampleRate,
			})
		}
		s.inFlight = nil
		s.stopTimer()
		return nil
	case "error", "conversation.item.input_audio_transcription.failed":
		if eventErrorCode(event) == "server_busy" {
			return errors.New("openai-compatible: realtime server is busy")
		}
		return errors.New("openai-compatible: realtime server reported a transcription error")
	default:
		return nil
	}
}

func (r *Realtime) run(ctx context.Context, conn *websocket.Conn, stopClose func() bool) {
	defer close(r.done)
	defer stopClose()
	defer func() {
		_ = conn.Close()
		if r.cancel != nil {
			r.cancel()
		}
		r.emitter.Close()
	}()

	results := make(chan readResult, 16)
	go r.readLoop(ctx, conn, results)
	session := realtimeSession{provider: r, conn: conn}

	for {
		select {
		case <-ctx.Done():
			return
		case <-session.timeout:
			session.fail("transcribe", errors.New("openai-compatible: realtime inference timed out"))
			return
		case result := <-results:
			if result.err != nil {
				if ctx.Err() != nil {
					return
				}
				session.fail("read", errors.New("openai-compatible: realtime connection ended unexpectedly"))
				return
			}
			if err := session.event(result.event); err != nil {
				session.fail("transcribe", err)
				return
			}
		case command := <-r.commands:
			if err := session.command(command); err != nil {
				session.fail("buffer audio", err)
				return
			}
		}
		closed, err := session.drain()
		if err != nil {
			session.fail("transcribe", err)
			return
		}
		if closed {
			r.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: r.model, Clean: true, At: time.Now()})
			return
		}
	}
}

func findRealtimeTurn(turns []*realtimeTurn, utterance int64) *realtimeTurn {
	for _, turn := range turns {
		if turn.utterance == utterance {
			return turn
		}
	}
	return nil
}
