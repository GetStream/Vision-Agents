package api

import (
	"context"
	"encoding/binary"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// noStreams is what the modality socket says on a deployment that inspects routing without
// serving it.
const noStreams = "this deployment does not stream this modality"

// startWait bounds how long a socket waits to be told what it is for. A caller that
// upgraded and then said nothing is holding a connection and a goroutine for no reason.
const startWait = 30 * time.Second

// Streams is the routing a caller running its own pipeline reaches directly, rather than
// by holding a conversation.
//
// It is the same routers a session uses. What differs is only who holds the conversation:
// here the caller does, and the router is one piece of their pipeline rather than the
// whole of it. Three of them are sockets, and search and the two recording jobs are plain
// requests, because nothing about them arrives in pieces.
type Streams struct {
	STT *sttrouter.Router
	TTS *ttsrouter.Router
	LLM *llmrouter.Router
	// Search answers a question at /v1/search.
	Search *searchrouter.Router
	// Transcriptions and Speech run the non-realtime jobs, against the batch half of each
	// vendor rather than the streaming one.
	Transcriptions *sttrouter.Recordings
	Speech         *ttsrouter.Recordings
}

// start is the first frame on every modality socket. It says what to route to and what to
// bill it against, which cannot be defaulted per frame without every frame carrying it.
//
// The options may be named here or kept in a router config and named by id, in which case
// anything sent as well overrides that one field of it. A caller with one set of options
// and many sockets stores them once; a caller with one socket sends them.
type start struct {
	Type string `json:"type"`
	// ConfigID names a stored router config to take the options from, by id or by name.
	ConfigID string `json:"config_id"`
	// Target is a "provider/model" name or a capability shortcut.
	Target string `json:"target"`
	// Voice selects the speaker, for text-to-speech.
	Voice string `json:"voice"`
	// Languages narrow the candidates.
	Languages []string `json:"languages"`
	// AgentID and CallID attribute the work to a conversation, when there is one.
	AgentID string `json:"agent_id"`
	CallID  string `json:"call_id"`
	// Tags are the caller's own cost labels.
	Tags map[string]string `json:"tags"`
	// SampleRate is the rate of the PCM that follows, for speech-to-text. Zero means
	// 16 kHz, which is what every provider here wants.
	SampleRate int `json:"sample_rate"`
	// Keyterms are the business-specific words the transcriber would otherwise get wrong.
	Keyterms []string `json:"keyterms"`
	// STT, TTS and LLM are the modality's own option block. Only the one belonging to the
	// socket's modality is read.
	STT options.STT `json:"stt"`
	TTS options.TTS `json:"tts"`
	LLM options.LLM `json:"llm"`
}

// options merges what the start frame said over what its config holds, filling in the
// three fields the frame has always carried at the top level so a caller written before
// the blocks existed keeps working.
func (s start) options(config store.RouterConfig) (options.STT, options.TTS, options.LLM) {
	speech := config.STT.Merge(s.STT)
	voice := config.TTS.Merge(s.TTS)
	model := config.LLM.Merge(s.LLM)

	if s.Target != "" {
		speech.Target, voice.Target, model.Target = s.Target, s.Target, s.Target
	}
	if len(s.Languages) > 0 {
		speech.Languages, voice.Languages = s.Languages, s.Languages
	}
	if s.Voice != "" {
		voice.Voice = s.Voice
	}
	if len(s.Keyterms) > 0 {
		speech.Keyterms = s.Keyterms
	}
	if s.SampleRate > 0 {
		rate := s.SampleRate
		speech.SampleRate = &rate
	}
	return speech, voice, model
}

// streamModality routes one modality for a caller holding its own pipeline.
func (s *Server) streamModality(w http.ResponseWriter, r *http.Request) {
	customerID, ok := CustomerFrom(r.Context())
	if !ok {
		writeError(w, http.StatusUnauthorized, "the "+CustomerHeader+" header is required")
		return
	}
	modality := routing.Modality(r.PathValue("modality"))
	if s.streams == nil || !s.serves(modality) {
		writeError(w, http.StatusNotFound, noStreams)
		return
	}

	connection, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Debug("could not upgrade the stream socket", "error", err)
		return
	}
	defer connection.Close()
	out := &socket{connection: connection}

	opening, err := readStart(connection)
	if err != nil {
		out.failed(err)
		return
	}

	// The socket's own lifetime bounds the provider session: a caller that hangs up has
	// stopped paying attention, and a session outliving it would go on being billed.
	ctx, cancel := context.WithCancel(context.WithoutCancel(r.Context()))
	defer cancel()

	config, err := s.routerOptions(ctx, customerID, opening.ConfigID)
	if err != nil {
		out.failed(err)
		return
	}
	speech, voice, model := opening.options(config)

	request := routing.Request{
		CustomerID: customerID,
		AgentID:    opening.AgentID,
		CallID:     opening.CallID,
		Tags:       tagsUnder(config, &opening.Tags),
	}
	if err := request.Tags.Validate(); err != nil {
		out.failed(err)
		return
	}

	switch modality {
	case routing.STT:
		err = s.streamSTT(ctx, out, request, targeted(speech))
	case routing.TTS:
		if voice.Target == "" {
			voice.Target = ttsDefaultTarget
		}
		err = s.streamTTS(ctx, out, request, voice)
	case routing.LLM:
		if model.Target == "" {
			model.Target = llmDefaultTarget
		}
		err = s.streamLLM(ctx, out, request, model)
	default:
		err = errors.New(noStreams)
	}
	if err != nil {
		out.failed(err)
	}
	out.frame(frame{"type": "closed"})
}

// serves reports whether this deployment routes a modality over a socket.
func (s *Server) serves(modality routing.Modality) bool {
	switch modality {
	case routing.STT:
		return s.streams.STT != nil
	case routing.TTS:
		return s.streams.TTS != nil
	case routing.LLM:
		return s.streams.LLM != nil
	default:
		return false
	}
}

// streamSTT transcribes binary PCM frames until the caller stops sending them.
func (s *Server) streamSTT(
	ctx context.Context,
	out *socket,
	request routing.Request,
	held options.STT,
) error {
	session, err := s.streams.STT.Start(ctx, sttrouter.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        held.Target,
		LanguageHints: held.Languages,
		Keyterms:      held.Keyterms,
		Options:       held,
	})
	if err != nil {
		return err
	}
	defer session.Close()

	out.frame(frame{
		"type":     "started",
		"provider": session.Provider(),
		"model":    session.Model(),
	})

	sampleRate := count(held.SampleRate)
	if sampleRate <= 0 {
		sampleRate = defaultSampleRate
	}

	// The two directions run at once: audio keeps arriving while transcripts come back,
	// and a reader that waited for each would transcribe at the speed of the answers.
	var writing sync.WaitGroup
	writing.Add(1)
	go func() {
		defer writing.Done()
		for event := range session.Events() {
			if encoded, ok := sttFrame(event); ok {
				if err := out.frame(encoded); err != nil {
					return
				}
			}
		}
	}()

	speaker := stt.Participant{ID: "caller"}
	for {
		kind, payload, err := out.connection.ReadMessage()
		if err != nil {
			break
		}
		if kind != websocket.BinaryMessage {
			// The only text frame that means anything mid-stream is a request to stop,
			// which is what closing the socket already says. Anything else is ignored
			// rather than treated as audio.
			continue
		}
		pcm := audio.FromBytes(payload, sampleRate, 1)
		if err := session.ProcessAudio(pcm, speaker); err != nil {
			out.failed(err)
		}
	}

	session.Close()
	writing.Wait()
	return nil
}

// streamTTS speaks what the caller sends and returns the audio as binary frames.
func (s *Server) streamTTS(
	ctx context.Context,
	out *socket,
	request routing.Request,
	held options.TTS,
) error {
	session, err := s.streams.TTS.Start(ctx, ttsrouter.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        held.Target,
		LanguageHints: held.Languages,
		Voice:         held.Voice,
		Options:       held,
	})
	if err != nil {
		return err
	}
	defer session.Close()

	out.frame(frame{
		"type":      "started",
		"provider":  session.Provider(),
		"model":     session.Model(),
		"streaming": session.Streaming(),
	})

	var writing sync.WaitGroup
	writing.Add(1)
	go func() {
		defer writing.Done()
		for event := range session.Events() {
			if err := writeTTS(out, event); err != nil {
				return
			}
		}
	}()

	for {
		var command struct {
			Type string `json:"type"`
			ID   string `json:"id"`
			Text string `json:"text"`
			// Final closes the utterance. A caller streaming a sentence a word at a time
			// sends false until the last piece.
			Final    *bool  `json:"final"`
			Voice    string `json:"voice"`
			Language string `json:"language"`
		}
		if err := out.connection.ReadJSON(&command); err != nil {
			break
		}

		switch command.Type {
		case "speak":
			final := true
			if command.Final != nil {
				final = *command.Final
			}
			if err := session.Synthesize(tts.Request{
				ID:       command.ID,
				Text:     command.Text,
				Voice:    command.Voice,
				Language: command.Language,
				Final:    final,
			}); err != nil {
				out.failed(err)
			}
		case "interrupt":
			if err := session.Interrupt(); err != nil {
				out.failed(err)
			}
		}
	}

	session.Close()
	writing.Wait()
	return nil
}

// streamLLM answers completions for a caller holding its own conversation.
func (s *Server) streamLLM(
	ctx context.Context,
	out *socket,
	request routing.Request,
	held options.LLM,
) error {
	session, err := s.streams.LLM.Start(ctx, llmrouter.Request{
		CustomerID:    request.CustomerID,
		AgentID:       request.AgentID,
		CallID:        request.CallID,
		Tags:          request.Tags,
		Target:        held.Target,
		LanguageHints: nil,
	})
	if err != nil {
		return err
	}
	defer session.Close()

	out.frame(frame{
		"type":     "started",
		"provider": session.Provider(),
		"model":    session.Model(),
	})

	// Each response is drained by a goroutine of its own, since a caller may have several
	// in flight and each one arrives on its own stream. The socket serialises the frames.
	var writing sync.WaitGroup
	var mu sync.Mutex
	inFlight := map[string]*llm.Stream{}

	for {
		var command respond
		if err := out.connection.ReadJSON(&command); err != nil {
			break
		}

		switch command.Type {
		case "respond":
			stream, err := session.Create(ctx, command.params(held))
			if err != nil {
				out.failed(err)
				continue
			}

			mu.Lock()
			inFlight[command.ID] = stream
			mu.Unlock()

			writing.Add(1)
			go func(id string, stream *llm.Stream) {
				defer writing.Done()
				defer stream.Close()

				for stream.Next() {
					encoded, ok := llmFrame(stream.Current())
					if !ok {
						continue
					}
					if err := out.frame(encoded); err != nil {
						break
					}
				}

				mu.Lock()
				delete(inFlight, id)
				mu.Unlock()
			}(command.ID, stream)

		case "interrupt":
			mu.Lock()
			for _, id := range command.ResponseIDs {
				if stream, running := inFlight[id]; running {
					stream.Close()
				}
			}
			mu.Unlock()
		}
	}

	// Closing the session abandons whatever is still being generated, which is what lets
	// each drainer reach the end of its stream and write its last frame.
	session.Close()
	writing.Wait()
	return nil
}

// respond is a frame on the language-model socket: either one response to generate or a
// list of responses to abandon.
//
// It carries the whole of what a response can be asked for rather than a corner of it,
// because a routed completion should be able to call a tool or cache a prompt prefix the
// way a session's can. Anything it leaves out falls back to the socket's options, which
// fall back to its config.
type respond struct {
	Type string `json:"type"`
	ID   string `json:"id"`
	// ResponseIDs are the responses an interrupt frame abandons.
	ResponseIDs []string `json:"response_ids"`

	Instructions string `json:"instructions"`
	Messages     []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Tools      []llm.Tool `json:"tools"`
	ToolChoice string     `json:"tool_choice"`
	// MaxTokens is the older name for the same cap, kept so a caller written before
	// max_output_tokens keeps working.
	MaxTokens          int               `json:"max_tokens"`
	MaxOutputTokens    int               `json:"max_output_tokens"`
	Temperature        *float64          `json:"temperature"`
	ReasoningEffort    string            `json:"reasoning_effort"`
	Format             string            `json:"format"`
	Verbosity          string            `json:"verbosity"`
	Store              *bool             `json:"store"`
	PreviousResponseID string            `json:"previous_response_id"`
	Conversation       string            `json:"conversation"`
	PromptCacheKey     string            `json:"prompt_cache_key"`
	Metadata           map[string]string `json:"metadata"`
}

// params turns the frame into a response to generate, with the socket's options behind
// anything it did not name.
func (r respond) params(held options.LLM) llm.ResponseParams {
	messages := make([]llm.Message, 0, len(r.Messages))
	for _, message := range r.Messages {
		messages = append(messages, llm.Message{
			Role:    llm.Role(message.Role),
			Content: message.Content,
		})
	}

	params := llm.ResponseParams{
		ID:                 r.ID,
		Instructions:       fallback(r.Instructions, held.Instructions),
		Input:              messages,
		Tools:              r.Tools,
		ToolChoice:         fallback(r.ToolChoice, held.ToolChoice),
		MaxOutputTokens:    firstSet(r.MaxOutputTokens, r.MaxTokens, count(held.MaxOutputTokens)),
		Temperature:        pick(r.Temperature, held.Temperature),
		Reasoning:          llm.ReasoningParams{Effort: fallback(r.ReasoningEffort, held.ReasoningEffort)},
		Store:              truthy(pick(r.Store, held.Store)),
		PreviousResponseID: r.PreviousResponseID,
		Conversation:       r.Conversation,
		PromptCacheKey:     fallback(r.PromptCacheKey, held.PromptCacheKey),
		Metadata:           r.Metadata,
	}
	params.Text = llm.TextParams{
		Format:    llm.TextFormat(fallback(r.Format, held.Format)),
		Verbosity: fallback(r.Verbosity, held.Verbosity),
	}
	if len(params.Metadata) == 0 {
		params.Metadata = held.Metadata
	}
	return params
}

// fallback, firstSet and pick are what "the frame wins, then the socket's options, then
// the provider's own default" looks like for the three kinds of field a response has.
func fallback(named, held string) string {
	if named != "" {
		return named
	}
	return held
}

func firstSet(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func pick[T any](named, held *T) *T {
	if named != nil {
		return named
	}
	return held
}

// defaultSampleRate is what every provider here transcribes at.
const defaultSampleRate = 16000

// readStart waits for the frame that says what the socket is for.
//
// A frame naming a config need not name a target, because the config it names may hold
// one. A frame naming neither is refused: nothing about it says what to route to.
func readStart(connection *websocket.Conn) (start, error) {
	connection.SetReadDeadline(time.Now().Add(startWait))
	defer connection.SetReadDeadline(time.Time{})

	var opening start
	if err := connection.ReadJSON(&opening); err != nil {
		return start{}, errors.New("the socket opens with a start frame naming the target")
	}
	if opening.Type != "" && opening.Type != "start" {
		return start{}, errors.New("the first frame must be a start frame")
	}
	if opening.Target == "" && opening.ConfigID == "" {
		return start{}, errors.New("routing needs a target, either sent or held in a config")
	}
	return opening, nil
}

// sttFrame renders a transcription event.
func sttFrame(event stt.Event) (frame, bool) {
	switch typed := event.(type) {
	case stt.Transcript:
		return frame{
			"type":               "transcript",
			"text":               typed.Text,
			"final":              typed.Final(),
			"confidence":         typed.Confidence,
			"language":           typed.Language,
			"provider":           typed.Provider,
			"model":              typed.Model,
			"processing_time_ms": typed.ProcessingTimeMs,
			"audio_duration_ms":  typed.AudioDurationMs,
		}, true
	case stt.Error:
		return frame{
			"type":    "error",
			"error":   typed.Err.Error(),
			"context": typed.Context,
			"fatal":   typed.Fatal,
		}, true
	default:
		return nil, false
	}
}

// audioHeader is the size of the header on every audio frame: a sample rate as a
// little-endian uint32, a channel count as a uint16, and two bytes held back so the
// samples that follow stay aligned.
const audioHeader = 8

// audioMessage frames one chunk of speech so that it describes itself.
//
// The rate is whatever the provider that spoke chose, and providers disagree, so a client
// told the rate once at the start would mis-play the first session that fell over to
// another voice. Eight bytes ahead of a chunk of audio is cheaper than the base64 it would
// take to say the same thing in JSON.
func audioMessage(pcm audio.PcmData) []byte {
	payload := pcm.Bytes()
	message := make([]byte, audioHeader+len(payload))
	binary.LittleEndian.PutUint32(message[0:4], uint32(pcm.SampleRate))
	binary.LittleEndian.PutUint16(message[4:6], uint16(pcm.Channels))
	copy(message[audioHeader:], payload)
	return message
}

// writeTTS sends one synthesis event, which is binary for audio and JSON for everything
// else. Audio goes out as raw little-endian PCM16 behind a small header rather than base64
// in a JSON field, because a voice is the one thing here worth not doubling in size.
func writeTTS(out *socket, event tts.Event) error {
	switch typed := event.(type) {
	case tts.AudioChunk:
		return out.binary(audioMessage(typed.Audio))
	case tts.SynthesisComplete:
		return out.frame(frame{
			"type":                  "synthesis_complete",
			"id":                    typed.SynthesisID,
			"provider":              typed.Provider,
			"model":                 typed.Model,
			"characters":            typed.Characters,
			"audio_duration_ms":     typed.AudioDurationMs,
			"time_to_first_byte_ms": typed.TimeToFirstByteMs,
			"synthesis_time_ms":     typed.SynthesisTimeMs,
			"interrupted":           typed.Interrupted,
		})
	case tts.Error:
		return out.frame(frame{
			"type":    "error",
			"id":      typed.SynthesisID,
			"error":   typed.Err.Error(),
			"context": typed.Context,
			"fatal":   typed.Fatal,
		})
	default:
		return nil
	}
}

// llmFrame renders a response event.
func llmFrame(event llm.Event) (frame, bool) {
	switch typed := event.(type) {
	case llm.OutputTextDelta:
		return frame{"type": "delta", "id": typed.ResponseID, "text": typed.Delta}, true
	case llm.ReasoningTextDelta:
		return frame{"type": "reasoning_delta", "id": typed.ResponseID, "text": typed.Delta}, true
	case llm.ResponseCompleted:
		response := typed.Response
		calls := make([]frame, 0, len(response.ToolCalls))
		for _, call := range response.ToolCalls {
			calls = append(calls, frame{
				"id":        call.ID,
				"name":      call.Name,
				"arguments": call.Arguments,
			})
		}
		return frame{
			"type":                   "complete",
			"id":                     response.ID,
			"provider":               response.Provider,
			"model":                  response.Model,
			"status":                 string(response.Status),
			"text":                   response.OutputText,
			"tool_calls":             calls,
			"input_tokens":           response.Usage.InputTokens,
			"cached_input_tokens":    response.Usage.InputTokensDetails.CachedTokens,
			"cache_write_tokens":     response.Usage.InputTokensDetails.CacheWriteTokens,
			"output_tokens":          response.Usage.OutputTokens,
			"reasoning_tokens":       response.Usage.OutputTokensDetails.ReasoningTokens,
			"time_to_first_token_ms": response.TimeToFirstTokenMs,
		}, true
	case llm.ResponseFailed:
		return frame{
			"type":    "error",
			"id":      typed.ResponseID,
			"error":   typed.Err.Error(),
			"context": typed.Context,
			"fatal":   typed.Fatal,
		}, true
	default:
		return nil, false
	}
}

// socket serialises writes to one connection.
//
// Both directions of a modality stream produce output: the provider's events are written
// by one goroutine while the frame reader writes failures of its own. Gorilla allows one
// writer at a time, and two would interleave halves of two frames into one unreadable
// message.
type socket struct {
	connection *websocket.Conn
	mu         sync.Mutex
}

// frame sends one JSON message under the write deadline.
func (s *socket) frame(encoded frame) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.connection.SetWriteDeadline(time.Now().Add(writeWait))
	return s.connection.WriteJSON(encoded)
}

// binary sends one binary message under the write deadline.
func (s *socket) binary(payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.connection.SetWriteDeadline(time.Now().Add(writeWait))
	return s.connection.WriteMessage(websocket.BinaryMessage, payload)
}

// failed reports a failure to the caller, which is all that can be done about one here.
func (s *socket) failed(err error) {
	s.frame(frame{"type": "error", "error": err.Error()})
}
