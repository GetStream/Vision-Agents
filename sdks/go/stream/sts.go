package stream

import (
	"context"
	"encoding/binary"
	"log/slog"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// STS is a whole conversation with one native audio model: the caller's audio in, the
// model's voice, what it heard, what it said and what it wants run out.
func (r Router) STS() Conversing { return Conversing{router: r} }

// Conversing routes speech-to-speech conversations.
type Conversing struct{ router Router }

// Tool is a function a speech-to-speech model may call.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// Parameters is a JSON Schema object describing the arguments.
	Parameters map[string]any `json:"parameters,omitempty"`
}

// Realtime opens a conversation socket, configured and ready for the caller's audio.
//
// Tools are given here rather than afterwards because some models take them only as the
// session opens; naming any routes the conversation to a model that calls them.
func (c Conversing) Realtime(
	ctx context.Context,
	options *acceleration.StsOptions,
	tools []Tool,
) (*Conversation, error) {
	opening := Frame{"sts": block(options)}
	if len(tools) > 0 {
		opening["tools"] = tools
	}
	socket, err := c.router.open(ctx, "sts", opening)
	if err != nil {
		return nil, err
	}

	conversation := &Conversation{
		socket: socket,
		events: make(chan Turn, audio),
	}
	go conversation.read(c.router.logger())
	return conversation, nil
}

// Turn is one thing the model did: a piece of its voice, or a frame saying what it heard,
// what it said, that a reply began or ended, or that it wants a tool run.
type Turn struct {
	// Samples are signed 16-bit little-endian PCM of the model's voice. Empty for a frame.
	Samples    []byte
	SampleRate int
	Channels   int
	// Generation numbers the reply the audio belongs to and Index the chunk within it. A
	// client drops audio whose generation a response_complete frame has already reported
	// interrupted: the model learns of a barge-in a round trip after the caller, and the
	// chunks in that gap are the tail of the words the caller talked over.
	Generation int
	Index      int
	// Frame is the JSON frame, for everything that is not audio.
	Frame Frame
}

// Conversation is one open speech-to-speech socket.
type Conversation struct {
	socket *Socket
	events chan Turn
}

// Send hands over 16 kHz mono PCM of the caller's speech.
func (c *Conversation) Send(pcm []byte) error { return c.socket.SendAudio(pcm) }

// Say injects a typed turn, which the model answers as it would a spoken one. A model that
// takes none refuses it with an error frame.
func (c *Conversation) Say(text string) error {
	return c.socket.Send(Frame{"type": "text", "text": text})
}

// Instruct changes the system prompt, on the models that allow it mid-session.
func (c *Conversation) Instruct(instructions string) error {
	return c.socket.Send(Frame{"type": "instructions", "instructions": instructions})
}

// Answer returns what a tool produced against the call that asked for it.
func (c *Conversation) Answer(callID, output string) error {
	return c.socket.Send(Frame{"type": "tool_result", "tool_call_id": callID, "output": output})
}

// Fail tells the model a tool did not work, and why.
func (c *Conversation) Fail(callID, reason string) error {
	return c.socket.Send(Frame{"type": "tool_result", "tool_call_id": callID, "error": reason})
}

// Interrupt stops the reply in flight. playedMs is how much of it the listener heard, so
// the model's own record ends where the listener's does; zero leaves it to the router.
func (c *Conversation) Interrupt(playedMs int) error {
	frame := Frame{"type": "interrupt"}
	if playedMs > 0 {
		frame["played_ms"] = playedMs
	}
	return c.socket.Send(frame)
}

// Events yields what the model does until the socket closes, when the channel closes.
func (c *Conversation) Events() <-chan Turn { return c.events }

// Close shuts the socket. Safe to call twice.
func (c *Conversation) Close() error { return c.socket.Close() }

func (c *Conversation) read(logger *slog.Logger) {
	defer close(c.events)

	for {
		frame, payload, err := c.socket.Read()
		if err != nil {
			ended(logger, "conversation", err)
			return
		}
		if payload != nil {
			if turn, ok := spoken(payload); ok {
				c.events <- turn
			}
			continue
		}
		if frame == nil || frame.Type() == "" {
			continue
		}
		c.events <- Turn{Frame: frame}
	}
}

// conversationHeader is what opens every audio frame on this socket: a uint32 sample rate,
// a uint16 channel count, a uint16 header version, a uint32 generation and a uint32 index,
// little-endian. It is longer than the voice socket's because it says which reply the
// audio belongs to.
const conversationHeader = 16

// conversationHeaderVersion is the version of that header this SDK reads.
const conversationHeaderVersion = 1

// spoken reads one audio frame, reporting false for a header this SDK does not know.
func spoken(payload []byte) (Turn, bool) {
	if len(payload) < conversationHeader {
		return Turn{}, false
	}
	if binary.LittleEndian.Uint16(payload[6:8]) != conversationHeaderVersion {
		return Turn{}, false
	}
	return Turn{
		Samples:    payload[conversationHeader:],
		SampleRate: int(binary.LittleEndian.Uint32(payload[0:4])),
		Channels:   int(binary.LittleEndian.Uint16(payload[4:6])),
		Generation: int(binary.LittleEndian.Uint32(payload[8:12])),
		Index:      int(binary.LittleEndian.Uint32(payload[12:16])),
	}, true
}
