package openaicompat

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

var realtimeInferenceTimeout = 2 * time.Minute

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
	audio         [][]byte
	audioBytes    int
	ended         bool
	itemID        string
	commitEventID string
	hadDelta      bool
	committed     bool
	acknowledged  bool
	terminal      bool
	committedAt   time.Time
}

type readResult struct {
	event realtimeEvent
	err   error
}

type realtimeSession struct {
	provider        *Realtime
	conn            *websocket.Conn
	order           []*realtimeTurn
	wireActive      *realtimeTurn
	awaitingCommit  []*realtimeTurn
	pending         []*realtimeTurn
	byItemID        map[string]*realtimeTurn
	byClientEventID map[string]*realtimeTurn
	commitSequence  uint64
	closing         bool
	timer           *time.Timer
	timerDeadline   time.Time
	timeout         <-chan time.Time
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
	s.timerDeadline = time.Time{}
	s.timeout = nil
}

func (s *realtimeSession) scheduleTimer() {
	if len(s.pending) == 0 {
		s.stopTimer()
		return
	}
	deadline := s.pending[0].committedAt.Add(realtimeInferenceTimeout)
	if s.timer != nil && s.timerDeadline.Equal(deadline) {
		return
	}
	if s.timer != nil {
		s.timer.Stop()
	}
	delay := time.Until(deadline)
	if delay < 0 {
		delay = 0
	}
	s.timer = time.NewTimer(delay)
	s.timerDeadline = deadline
	s.timeout = s.timer.C
}

func (s *realtimeSession) fail(contextName string, err error) {
	r := s.provider
	r.emitter.Send(stt.Error{Provider: ProviderName, Model: r.model, Err: err, Context: contextName, Fatal: true})
	r.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: r.model, Reason: "upstream connection ended", Clean: false, At: time.Now()})
}

// drain sends each ready turn in wire order without waiting for its transcript.
func (s *realtimeSession) drain() (bool, error) {
	for len(s.order) > 0 {
		turn := s.order[0]
		for _, chunk := range turn.audio {
			if err := writeAudioAppend(s.conn, chunk); err != nil {
				return false, fmt.Errorf("openai-compatible: realtime audio append: %w", err)
			}
			turn.audioBytes += len(chunk)
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
		s.commitSequence++
		turn.commitEventID = fmt.Sprintf("realtime-commit-%d", s.commitSequence)
		if err := writeRealtimeJSON(s.conn, map[string]string{
			"type":     "input_audio_buffer.commit",
			"event_id": turn.commitEventID,
		}); err != nil {
			return false, fmt.Errorf("openai-compatible: realtime audio commit: %w", err)
		}
		turn.committedAt = time.Now()
		turn.committed = true
		s.awaitingCommit = append(s.awaitingCommit, turn)
		s.pending = append(s.pending, turn)
		if s.byClientEventID == nil {
			s.byClientEventID = make(map[string]*realtimeTurn)
		}
		s.byClientEventID[turn.commitEventID] = turn
		s.order = s.order[1:]
		if s.wireActive == turn {
			s.wireActive = nil
		}
		s.scheduleTimer()
	}
	if s.closing && len(s.pending) == 0 {
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
			if len(s.order)+len(s.pending) >= realtimeMaxOutstandingTurns {
				return errors.New("openai-compatible: realtime outstanding turn limit exceeded")
			}
			turn = &realtimeTurn{turnMetadata: command.meta}
			s.order = append(s.order, turn)
		} else {
			if turn.participant != (stt.Participant{}) && turn.participant != command.meta.participant {
				return errors.New("openai-compatible: realtime audio participant changed within an utterance")
			}
			turn.participant = command.meta.participant
		}
		turn.audio = append(turn.audio, command.pcm)
		s.wireActive = turn
	case commandCommit:
		turn := findRealtimeTurn(s.order, command.meta.utterance)
		if turn == nil {
			if len(s.order)+len(s.pending) >= realtimeMaxOutstandingTurns {
				return errors.New("openai-compatible: realtime outstanding turn limit exceeded")
			}
			turn = &realtimeTurn{turnMetadata: command.meta}
			s.order = append(s.order, turn)
		} else {
			if turn.participant != (stt.Participant{}) && turn.participant != command.meta.participant {
				return errors.New("openai-compatible: realtime commit participant did not match audio")
			}
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
	case "conversation.item.input_audio_transcription.delta":
		if event.ItemID == "" {
			return errors.New("openai-compatible: realtime transcript delta has no item")
		}
		turn, err := s.turnForDeltaItemID(event.ItemID)
		if err != nil {
			return err
		}
		if turn == nil {
			return nil
		}
		if event.ContentIndex != 0 {
			return errors.New("openai-compatible: realtime transcript delta has an unsupported content index")
		}
		if event.Delta == "" {
			return nil
		}
		turn.hadDelta = true
		s.provider.emitter.Send(stt.Transcript{
			Participant: turn.participant,
			Mode:        stt.ModeDelta,
			Utterance:   turn.utterance,
			Text:        event.Delta,
			Provider:    ProviderName,
			Model:       s.provider.model,
		})
		return nil
	case "input_audio_buffer.committed":
		if len(s.awaitingCommit) == 0 {
			return nil
		}
		if event.ItemID == "" {
			return errors.New("openai-compatible: realtime commit acknowledgement has no item")
		}
		turn := s.awaitingCommit[0]
		if err := s.bindItemID(turn, event.ItemID); err != nil {
			return fmt.Errorf("openai-compatible: realtime commit acknowledgement: %w", err)
		}
		s.awaitingCommit = s.awaitingCommit[1:]
		turn.acknowledged = true
		s.retireTurn(turn)
		return nil
	case "conversation.item.input_audio_transcription.completed":
		if event.ItemID == "" {
			return errors.New("openai-compatible: realtime transcript has no item")
		}
		turn, err := s.turnForCommittedItemID(event.ItemID)
		if err != nil {
			return err
		}
		if turn == nil {
			return nil
		}
		if event.ContentIndex != 0 {
			return errors.New("openai-compatible: realtime transcript has an unsupported content index")
		}
		if turn.terminal {
			return nil
		}
		if text := strings.TrimSpace(event.Transcript); text != "" || turn.hadDelta {
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
		turn.terminal = true
		s.retireTurn(turn)
		return nil
	case "conversation.item.input_audio_transcription.failed":
		if event.ItemID == "" {
			return errors.New("openai-compatible: realtime transcription failure has no item")
		}
		turn, err := s.turnForCommittedItemID(event.ItemID)
		if err != nil {
			return err
		}
		if turn == nil {
			return nil
		}
		if event.ContentIndex != 0 {
			return errors.New("openai-compatible: realtime transcription failure has an unsupported content index")
		}
		s.failTurn(turn, eventErrorCode(event))
		return nil
	case "error":
		if eventErrorCode(event) == "server_busy" {
			return errors.New("openai-compatible: realtime server is busy")
		}
		if s.byClientEventID[event.Error.EventID] != nil {
			message := "openai-compatible: realtime commit was rejected"
			if code := eventErrorCode(event); code != "" {
				message += " (" + code + ")"
			}
			return errors.New(message)
		}
		return errors.New("openai-compatible: realtime server reported a transcription error")
	default:
		return nil
	}
}

func (s *realtimeSession) turnForDeltaItemID(itemID string) (*realtimeTurn, error) {
	if turn := s.byItemID[itemID]; turn != nil {
		return turn, nil
	}
	if turn, err := s.turnForCommittedItemID(itemID); err != nil || turn != nil {
		return turn, err
	}
	if s.wireActive != nil && !s.wireActive.committed {
		if err := s.bindItemID(s.wireActive, itemID); err != nil {
			return nil, err
		}
		return s.wireActive, nil
	}
	return nil, nil
}

func (s *realtimeSession) turnForCommittedItemID(itemID string) (*realtimeTurn, error) {
	if turn := s.byItemID[itemID]; turn != nil {
		return turn, nil
	}
	for _, turn := range s.awaitingCommit {
		if turn.itemID == "" {
			if err := s.bindItemID(turn, itemID); err != nil {
				return nil, err
			}
			return turn, nil
		}
	}
	return nil, nil
}

func (s *realtimeSession) bindItemID(turn *realtimeTurn, itemID string) error {
	if itemID == "" {
		return errors.New("openai-compatible: realtime item id is empty")
	}
	if turn.itemID != "" && turn.itemID != itemID {
		return errors.New("openai-compatible: realtime item id changed for a turn")
	}
	if s.byItemID == nil {
		s.byItemID = make(map[string]*realtimeTurn)
	}
	if existing := s.byItemID[itemID]; existing != nil && existing != turn {
		return errors.New("openai-compatible: realtime item id matched multiple turns")
	}
	turn.itemID = itemID
	s.byItemID[itemID] = turn
	return nil
}

func (s *realtimeSession) failTurn(turn *realtimeTurn, code string) {
	if turn.terminal {
		return
	}
	if turn.hadDelta {
		s.provider.emitter.Send(stt.Transcript{
			Participant:      turn.participant,
			Mode:             stt.ModeFinal,
			Utterance:        turn.utterance,
			Provider:         ProviderName,
			Model:            s.provider.model,
			ProcessingTimeMs: float64(time.Since(turn.committedAt).Microseconds()) / 1000,
			AudioDurationMs:  float64(turn.samples) * 1000 / stt.SampleRate,
		})
	}
	message := "openai-compatible: realtime transcription failed"
	if code != "" {
		message += " (" + code + ")"
	}
	s.provider.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    s.provider.model,
		Err:      errors.New(message),
		Context:  "transcribe",
		Fatal:    false,
	})
	turn.terminal = true
	s.retireTurn(turn)
}

func (s *realtimeSession) retireTurn(turn *realtimeTurn) {
	if !turn.terminal || !turn.acknowledged {
		return
	}
	for i, pending := range s.pending {
		if pending == turn {
			s.pending = append(s.pending[:i], s.pending[i+1:]...)
			break
		}
	}
	if turn.itemID != "" {
		delete(s.byItemID, turn.itemID)
	}
	if turn.commitEventID != "" {
		delete(s.byClientEventID, turn.commitEventID)
	}
	s.scheduleTimer()
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
	session := realtimeSession{
		provider:        r,
		conn:            conn,
		byItemID:        make(map[string]*realtimeTurn),
		byClientEventID: make(map[string]*realtimeTurn),
	}

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
