// Package streamedge puts an agent in a Stream call.
//
// It is the transport half of internal/agent: everything about credentials, tracks,
// subscriptions and codecs lives here, so the agent itself only ever sees 16 kHz mono PCM
// in and out. Inbound Opus is decoded straight to 16 kHz; outbound PCM is encoded back to
// 48 kHz Opus for the track the agent publishes.
package streamedge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/getstream-go-webrtc/coordinator"
	"github.com/GetStream/getstream-go-webrtc/jointrace"
	"github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/google/uuid"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
	userTokenEnvVar = "STREAM_USER_TOKEN"
	regionEnvVar    = "STREAM_REGION"
	baseURLEnvVar   = "STREAM_BASE_URL"
	wsURLEnvVar     = "STREAM_WS_URL"
)

// defaultCallType is the call type an agent joins under.
const defaultCallType = "agent"

// audioBuffer is how many chunks of inbound speech may queue before the decoder is made to
// wait. A chunk is 20 ms, so this is a fifth of a second of slack.
const audioBuffer = 10

// attendanceBuffer is how many arrivals and departures may queue. Generous relative to what
// a call sees, because these are delivered on the signalling goroutine: a full channel there
// would hold up every other event the SFU is trying to report.
const attendanceBuffer = 32

// Options configures an Edge. The credentials fall back to the environment, which is how
// every other provider in this service is configured.
type Options struct {
	// CallID is the call to join.
	CallID string
	// CallType defaults to "agent".
	CallType string
	// User is the identity the agent joins as.
	User User

	// APIKey defaults to STREAM_API_KEY.
	APIKey string
	// APISecret defaults to STREAM_API_SECRET. It mints the agent's token, which is why a
	// server-side agent needs no token of its own.
	APISecret string
	// UserToken defaults to STREAM_USER_TOKEN and is used in preference to a secret.
	UserToken string

	// Region is where the agent runs: a GCP or AWS region ("us-east1", "eu-west-1") or an
	// airport code. It defaults to STREAM_REGION. The coordinator puts the agent on an SFU
	// near it; without one, near the address the agent connects from.
	Region string

	// BaseURL is the coordinator to join through. It defaults to STREAM_BASE_URL, which the
	// rest of the service's Stream clients also read, and then to Stream's production API.
	BaseURL string
	// WSURL is the coordinator's websocket. It defaults to STREAM_WS_URL, and then to
	// BaseURL's /api/v2/connect.
	WSURL string

	Logger *slog.Logger

	// clientOptions are passed to the SDK client as they are, so a test can add a network
	// delay.
	clientOptions []rtc.Option
	// coordinatorOptions are added to BaseURL's coordinator options, so a test can pin the
	// SFU. The SDK keeps only the last rtc.WithCoordinatorOptions, so clientOptions cannot.
	coordinatorOptions []coordinator.Option
	// joinOptions are added to the call's join options, so a test can pick the join flow.
	joinOptions []rtc.JoinOption
}

// User is who the agent is in the call.
type User struct {
	ID   string
	Name string
}

// Edge is an agent's place in a Stream call. It satisfies agent.Edge.
type Edge struct {
	options Options
	logger  *slog.Logger
	// location is what the agent tells the coordinator about where it is (Options.Region).
	location string

	// inbound carries every participant's speech, already decoded to what the
	// speech-to-text providers accept.
	inbound *emit.Emitter[agent.InboundAudio]
	// attending carries who comes and goes, which is how an agent that did not start the
	// call knows somebody is there to talk to.
	attending *emit.Emitter[agent.Attendance]
	// tracing carries how the call was joined, once.
	tracing *emit.Emitter[agent.JoinTrace]
	speaker *speaker

	client *rtc.Client
	call   *rtc.Call

	mu sync.Mutex
	// listening holds what stops the decoding of each subscribed track, so a track that
	// goes away stops being decoded.
	listening   map[string]chan struct{}
	unregisters []func()
	left        bool

	leaveOnce sync.Once
	leftDone  chan struct{}
}

// New validates the options and returns an Edge. It connects nothing; Join does that.
func New(options Options) (*Edge, error) {
	if options.CallID == "" {
		return nil, errors.New("streamedge: a call id is required")
	}
	if options.User.ID == "" {
		return nil, errors.New("streamedge: a user id is required")
	}
	if options.CallType == "" {
		options.CallType = defaultCallType
	}
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.UserToken == "" {
		options.UserToken = os.Getenv(userTokenEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("streamedge: %s is not set", apiKeyEnvVar)
	}
	if options.APISecret == "" && options.UserToken == "" {
		return nil, fmt.Errorf("streamedge: set %s or %s", userTokenEnvVar, apiSecretEnvVar)
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Region == "" {
		options.Region = os.Getenv(regionEnvVar)
	}
	location, known := locationFor(options.Region)
	if !known {
		options.Logger.Warn("streamedge: unknown region, placing the agent by its address", "region", options.Region)
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.WSURL == "" {
		options.WSURL = os.Getenv(wsURLEnvVar)
	}
	if options.WSURL == "" && options.BaseURL != "" {
		wsURL, err := websocketURL(options.BaseURL)
		if err != nil {
			return nil, err
		}
		options.WSURL = wsURL
	}

	return &Edge{
		options:   options,
		logger:    options.Logger.With("call", options.CallType+":"+options.CallID),
		location:  location,
		inbound:   emit.New[agent.InboundAudio](audioBuffer),
		attending: emit.New[agent.Attendance](attendanceBuffer),
		tracing:   emit.New[agent.JoinTrace](1),
		speaker:   newSpeaker(options.Logger),
		listening: map[string]chan struct{}{},
		leftDone:  make(chan struct{}),
	}, nil
}

// Join connects and joins the call, publishing the agent's own audio track from the join.
//
// There is nothing to subscribe to: the SFU sends every participant everyone else's audio,
// what is already being published at the join and whatever is published later.
func (e *Edge) Join(ctx context.Context) error {
	started := time.Now()
	info, voice, err := e.voice()
	if err != nil {
		return err
	}
	client, err := e.connect()
	if err != nil {
		return err
	}
	e.client = client

	// Kept off the edge until the join succeeds: leaving a call that never connected panics
	// in the SDK, which has no signaling client yet to report its stats through.
	call := client.Call(e.options.CallType, e.options.CallID)
	// Set before Join, which starts the trace.
	call.OnJoinTrace(func(trace jointrace.Trace) { e.onJoinTrace(trace, call.JoinFlow()) })
	joinOptions := append([]rtc.JoinOption{rtc.WithLocation(e.location), rtc.WithTrack(info, voice),
		rtc.WithOnTrack(rtc.SubscriberFunc(func(remote rtc.OnTrackReceived) {
			e.listen(remote)
		}))}, e.options.joinOptions...)
	joined, err := call.Join(ctx, joinOptions...)
	if err != nil {
		return fmt.Errorf("streamedge: join %s:%s: %w", e.options.CallType, e.options.CallID, err)
	}
	e.call = call

	// Registered before the people already here are reported, so somebody arriving during
	// this is reported once rather than not at all.
	e.watchAttendance()
	e.reportPresent(joined.GetCallState())

	flow := call.JoinFlow()
	e.logger.Info("joined the call", "user", e.options.User.ID, "session", e.call.SessionID.Load(),
		"flow", flow, "join_ms", float64(time.Since(started).Microseconds())/1000)
	if flow != rtc.JoinFlowFast {
		// The SDK falls back without failing the join, so this is the only word of it.
		e.logger.Warn("joined without the fast join: the deployment does not offer it", "flow", flow)
	}
	return nil
}

// Audio carries what the participants said, as 16 kHz mono PCM.
func (e *Edge) Audio() <-chan agent.InboundAudio { return e.inbound.Events() }

// Attendance reports who comes and goes, satisfying agent.Roster.
func (e *Edge) Attendance() <-chan agent.Attendance { return e.attending.Events() }

// PublishAudio sends a chunk of the agent's speech to the call.
func (e *Edge) PublishAudio(pcm audio.PcmData) error { return e.speaker.Write(pcm) }

// SpeechPending reports whether published speech is still waiting to go out, satisfying
// agent.Playout.
func (e *Edge) SpeechPending() bool { return e.speaker.pending() }

// DropSpeech throws away published speech that has not been heard yet, satisfying
// agent.Playout.
func (e *Edge) DropSpeech() { e.speaker.drop() }

// Leave releases the call. It is safe to call more than once.
func (e *Edge) Leave() error {
	var err error
	e.leaveOnce.Do(func() { err = e.leave() })
	return err
}

// Call exposes the underlying call, so a caller can reach the SDK's own features.
func (e *Edge) Call() *rtc.Call { return e.call }

func (e *Edge) leave() error {
	close(e.leftDone)
	e.mu.Lock()
	e.left = true
	unregisters := e.unregisters
	e.unregisters = nil
	for _, stop := range e.listening {
		close(stop)
	}
	e.listening = map[string]chan struct{}{}
	e.mu.Unlock()

	for _, unregister := range unregisters {
		unregister()
	}
	e.inbound.Close()
	e.attending.Close()
	e.tracing.Close()

	var failures []error
	if err := e.speaker.Close(); err != nil {
		failures = append(failures, err)
	}
	if e.call != nil {
		if err := e.call.Leave("the agent finished"); err != nil {
			failures = append(failures, fmt.Errorf("streamedge: leave: %w", err))
		}
	}
	if e.client != nil {
		e.client.Close()
	}
	return errors.Join(failures...)
}

// JoinTraces reports how the call was joined, satisfying agent.Connector.
func (e *Edge) JoinTraces() <-chan agent.JoinTrace { return e.tracing.Events() }

// onJoinTrace hands the SDK's record of the join to the agent, unchanged. The SDK calls it
// once, on its own goroutine, when media flows both ways or after rtc.JoinTraceTimeout.
func (e *Edge) onJoinTrace(trace jointrace.Trace, flow rtc.JoinFlow) {
	joined, err := joinTrace(trace, flow)
	if err != nil {
		e.logger.Warn("join trace not reported", "error", err)
		return
	}
	e.logger.Info("join trace", "flow", joined.Flow, "critical_ms", joined.CriticalMs,
		"critical_rtts", joined.CriticalRTTs, "critical_path", joined.CriticalPath)
	e.logger.Debug("join DAG\n" + trace.String())
	e.tracing.Send(joined)
}

// joinTrace is the SDK's trace as the agent carries it: the SDK's own JSON report, the flow
// the join took, and the critical path pulled out of it for logs.
func joinTrace(trace jointrace.Trace, flow rtc.JoinFlow) (agent.JoinTrace, error) {
	report := trace.Report()
	encoded, err := json.Marshal(report)
	if err != nil {
		return agent.JoinTrace{}, fmt.Errorf("streamedge: encode the join trace: %w", err)
	}
	return agent.JoinTrace{
		Trace:        encoded,
		Flow:         string(flow),
		CriticalPath: strings.Join(report.CriticalPath, " > "),
		CriticalMs:   report.CriticalMs,
		CriticalRTTs: report.CriticalRTTs,
	}, nil
}

// connect builds the SDK client, preferring a token over a secret.
//
// The coordinator websocket stays on even though the agent reads none of its events: its
// connect is what registers the agent as a user, and the coordinator refuses to let a user it
// has never seen join a call. The SDK connects it in the background and the join does not wait
// for it, except the very first join of a new agent user.
func (e *Edge) connect() (*rtc.Client, error) {
	user := rtc.User{ID: e.options.User.ID, Name: e.options.User.Name}
	if user.Name == "" {
		user.Name = user.ID
	}
	options := slices.Clone(e.options.clientOptions)
	if e.options.BaseURL != "" {
		coordinatorOptions := append([]coordinator.Option{
			coordinator.ApiURL(e.options.BaseURL), coordinator.WithWsURL(e.options.WSURL)}, e.options.coordinatorOptions...)
		options = append(options, rtc.WithCoordinatorOptions(coordinatorOptions...))
	}

	if e.options.UserToken != "" {
		client, err := rtc.NewClient(e.options.APIKey, user, rtc.StaticToken(e.options.UserToken), options...)
		if err != nil {
			return nil, fmt.Errorf("streamedge: connect: %w", err)
		}
		return client, nil
	}

	clientOptions := []rtc.ClientOption{rtc.WithUser(user)}
	for _, option := range options {
		clientOptions = append(clientOptions, option)
	}
	client, err := rtc.NewRTCClient(e.options.APIKey, e.options.APISecret, clientOptions...)
	if err != nil {
		return nil, fmt.Errorf("streamedge: connect: %w", err)
	}
	return client, nil
}

// websocketURL is the coordinator websocket that goes with a coordinator base URL.
func websocketURL(base string) (string, error) {
	parsed, err := url.Parse(base)
	if err != nil {
		return "", fmt.Errorf("streamedge: %s %q: %w", baseURLEnvVar, base, err)
	}
	switch parsed.Scheme {
	case "https":
		parsed.Scheme = "wss"
	case "http":
		parsed.Scheme = "ws"
	default:
		return "", fmt.Errorf("streamedge: %s %q: want http or https", baseURLEnvVar, base)
	}
	parsed.Path = strings.TrimSuffix(parsed.Path, "/") + "/api/v2/connect"
	return parsed.String(), nil
}

// voice is the agent's Opus track, which the join publishes. The track starts pulling frames
// from the speaker as soon as the transceiver is bound.
func (e *Edge) voice() (*sfu_models.TrackInfo, webrtc.TrackLocal, error) {
	info := &sfu_models.TrackInfo{
		TrackId:   uuid.NewString(),
		TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO,
	}
	voice, err := track.NewAudioTrack(info, e.speaker, webrtc.RTPCodecCapability{
		MimeType:  webrtc.MimeTypeOpus,
		ClockRate: opusSampleRate,
		Channels:  opusNegotiatedChannels,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("streamedge: build audio track: %w", err)
	}
	return info, voice, nil
}

// listen decodes one participant's track into the audio the agent listens to. Reading the
// track is also what pulls media through the receiver, so a track nobody reads is a track
// that never arrives.
func (e *Edge) listen(remote rtc.OnTrackReceived) {
	if remote.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
		return
	}
	// The SFU forwards every other session's audio. A second instance of the same agent
	// left behind in the call arrives here and is heard as a caller, and the two then
	// answer each other until nobody else can get a word in.
	if string(remote.ParticipantID.UserID) == e.options.User.ID {
		e.logger.Warn("another instance of this agent is in the call, ignoring it",
			"user", e.options.User.ID, "session", string(remote.ParticipantID.SessionID))
		return
	}

	participant := stt.Participant{
		ID:     string(remote.ParticipantID.SessionID),
		UserID: string(remote.ParticipantID.UserID),
	}
	if remote.Participant != nil {
		participant.Name = remote.Participant.Name
	}

	reader, err := audiortc.NewTrackReader(remote.Track,
		audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
	if err != nil {
		e.logger.Error("could not decode a participant's audio",
			"participant", participant.UserID, "error", err)
		return
	}

	stop := make(chan struct{})
	e.mu.Lock()
	if e.left {
		e.mu.Unlock()
		return
	}
	if previous, ok := e.listening[remote.Track.ID()]; ok {
		close(previous)
	}
	e.listening[remote.Track.ID()] = stop
	e.mu.Unlock()

	e.logger.Debug("listening to a participant", "participant", participant.UserID)
	go e.hear(reader, participant, stop)
}

// hear hands one participant's decoded audio to the agent until the track ends or stop is
// closed.
func (e *Edge) hear(reader *audiortc.TrackReader, participant stt.Participant, stop <-chan struct{}) {
	defer reader.Close()
	for pcm, err := range reader.Frames() {
		if err != nil {
			e.logger.Debug("stopped hearing a participant", "participant", participant.UserID, "error", err)
			return
		}
		select {
		case <-stop:
			return
		default:
		}
		e.inbound.Send(agent.InboundAudio{
			Participant: participant,
			Audio: audio.PcmData{
				Samples:    pcm.ToInt16().Int16(),
				SampleRate: stt.SampleRate,
				Channels:   1,
			},
		})
	}
}

// watchAttendance reports arrivals and departures.
//
// The SFU says who is there whether or not they have published anything, which is the point:
// a caller who has not spoken yet is still somebody to say hello to, and a track is the only
// other evidence there would be.
func (e *Edge) watchAttendance() {
	joined := rtc.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_ParticipantJoined) {
		e.report(event.ParticipantJoined.GetParticipant(), true)
	})
	left := rtc.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_ParticipantLeft) {
		e.report(event.ParticipantLeft.GetParticipant(), false)
	})

	e.mu.Lock()
	e.unregisters = append(e.unregisters, joined, left)
	e.mu.Unlock()
}

// reportPresent reports the people who were already in the call when the agent joined. An
// agent that answers a call the caller reached first would otherwise be told about nobody.
func (e *Edge) reportPresent(state *sfu_models.CallState) {
	for _, participant := range state.GetParticipants() {
		e.report(participant, true)
	}
}

// report puts one arrival or departure on the channel, leaving out the agent itself.
func (e *Edge) report(participant *sfu_models.Participant, joined bool) {
	if participant == nil || participant.GetUserId() == e.options.User.ID {
		return
	}

	e.mu.Lock()
	left := e.left
	e.mu.Unlock()
	if left {
		return
	}

	e.attending.Send(agent.Attendance{
		Participant: stt.Participant{
			ID:     participant.GetSessionId(),
			UserID: participant.GetUserId(),
			Name:   participant.GetName(),
		},
		Joined: joined,
	})
}
