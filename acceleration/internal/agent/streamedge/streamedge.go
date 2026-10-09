// Package streamedge puts an agent in a Stream call.
//
// It is the transport half of internal/agent: everything about credentials, tracks,
// subscriptions and codecs lives here, so the agent itself only ever sees 16 kHz mono PCM
// in and out. Inbound Opus is decoded straight to 16 kHz; outbound PCM is encoded back to
// 48 kHz Opus for the track the agent publishes.
package streamedge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/GetStream/getstream-go-webrtc/track"
	getstream "github.com/GetStream/getstream-go/v5"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/google/uuid"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
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

// Options configures an Edge. The credentials are whoever builds the edge's to give: the
// router gives the identity of the Stream app the session is pinned to, and nothing is read
// from the environment, which would make every call the deployment's.
type Options struct {
	// CallID is the call to join.
	CallID string
	// CallType defaults to "agent".
	CallType string
	// User is the identity the agent joins as.
	User User

	// APIKey is the app's key.
	APIKey string
	// APISecret mints the agent's token, which is why a server-side agent needs no token
	// of its own.
	APISecret string
	// UserToken is a fixed token used in preference to a secret.
	UserToken string
	// BaseURL is the Stream API the app is reached at, and HTTPClient what reaches it.
	// Empty leaves both to the SDK.
	BaseURL    string
	HTTPClient *http.Client

	Logger *slog.Logger
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

	// inbound carries every participant's speech, already decoded to what the
	// speech-to-text providers accept.
	inbound *emit.Emitter[agent.InboundAudio]
	// attending carries who comes and goes, which is how an agent that did not start the
	// call knows somebody is there to talk to.
	attending *emit.Emitter[agent.Attendance]
	speaker   *speaker

	client *rtc.Client
	call   *rtc.Call

	mu sync.Mutex
	// listening holds what stops the decoding of each subscribed track, so a track that
	// goes away stops being decoded.
	listening map[string]chan struct{}
	// subscribed is the whole subscription list, because the SFU replaces it wholesale on
	// every update rather than adding to it.
	subscribed  []*signal_rpc.TrackSubscriptionDetails
	unregisters []func()
	left        bool

	leaveOnce sync.Once
	leftDone  chan struct{}
}

// New validates the options and returns an Edge. It connects nothing; Join does that.
func New(options Options) (*Edge, error) {
	if options.CallID == "" {
		return nil, stack.Wrap(errors.New("streamedge: a call id is required"))
	}
	if options.User.ID == "" {
		return nil, stack.Wrap(errors.New("streamedge: a user id is required"))
	}
	if options.CallType == "" {
		options.CallType = defaultCallType
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("streamedge: an api key is required"))
	}
	if options.APISecret == "" && options.UserToken == "" {
		return nil, stack.Wrap(errors.New("streamedge: a secret or a user token is required"))
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Edge{
		options:   options,
		logger:    options.Logger.With("call", options.CallType+":"+options.CallID),
		inbound:   emit.New[agent.InboundAudio](audioBuffer),
		attending: emit.New[agent.Attendance](attendanceBuffer),
		speaker:   newSpeaker(options.Logger),
		listening: map[string]chan struct{}{},
		leftDone:  make(chan struct{}),
	}, nil
}

// Join connects, subscribes to what the participants are already saying, and publishes the
// agent's own audio track.
func (e *Edge) Join(ctx context.Context) error {
	started := time.Now()
	client, err := e.connect()
	if err != nil {
		return err
	}
	e.client = client

	// Kept off the edge until the join succeeds: leaving a call that never connected panics
	// in the SDK, which has no signaling client yet to report its stats through.
	call := client.Call(e.options.CallType, e.options.CallID)
	signalingStarted := time.Now()
	joined, err := call.Join(ctx, rtc.WithOnTrack(rtc.SubscriberFunc(func(remote rtc.OnTrackReceived) {
		e.listen(remote)
	})))
	if err != nil {
		return stack.Wrap(fmt.Errorf("streamedge: join %s:%s: %w", e.options.CallType, e.options.CallID, err))
	}
	e.call = call
	signalingMs := float64(time.Since(signalingStarted).Microseconds()) / 1000

	// Joining subscribes to nothing, so the SFU has to be told what to forward: whatever is
	// already being published, and then whatever is published later.
	e.mu.Lock()
	e.subscribed = audioSubscriptions(joined.GetCallState(), e.options.User.ID)
	subscriptions := slices.Clone(e.subscribed)
	e.mu.Unlock()

	subscribeStarted := time.Now()
	if err := e.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
		return stack.Wrap(fmt.Errorf("streamedge: subscribe: %w", err))
	}
	subscribeMs := float64(time.Since(subscribeStarted).Microseconds()) / 1000
	e.watchForNewTracks(ctx)
	// Registered before the people already here are reported, so somebody arriving during
	// this is reported once rather than not at all.
	e.watchAttendance()
	e.reportPresent(joined.GetCallState())

	publishStarted := time.Now()
	if err := e.publish(); err != nil {
		return err
	}
	e.logger.Info("call connection timing", "signaling_ms", signalingMs,
		"subscribe_ms", subscribeMs,
		"publish_ms", float64(time.Since(publishStarted).Microseconds())/1000,
		"join_ms", float64(time.Since(started).Microseconds())/1000)
	go e.reportICE(ctx, started)

	e.logger.Info("joined the call",
		"user", e.options.User.ID, "session", e.call.SessionID.Load(), "tracks", len(subscriptions))
	return nil
}

// Audio carries what the participants said, as 16 kHz mono PCM.
func (e *Edge) Audio() <-chan agent.InboundAudio { return e.inbound.Events() }

// Attendance reports who comes and goes, satisfying agent.Roster.
func (e *Edge) Attendance() <-chan agent.Attendance { return e.attending.Events() }

// PublishAudio sends a chunk of the agent's speech to the call.
func (e *Edge) PublishAudio(pcm audio.PcmData) error { return e.speaker.Write(pcm) }

// PublishAudioContext queues a chunk of speech, stopping when ctx is cancelled.
func (e *Edge) PublishAudioContext(ctx context.Context, pcm audio.PcmData) error {
	return e.speaker.WriteContext(ctx, pcm)
}

// PublishAudioMarked queues a chunk of speech like PublishAudioContext, and tells marks when
// its first frame was queued and when the call took the first one that was not silence,
// satisfying agent.MarkedPlayout.
func (e *Edge) PublishAudioMarked(ctx context.Context, pcm audio.PcmData, marks agent.PlayoutMarks) error {
	return e.speaker.WriteMarked(ctx, pcm, marks)
}

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

// reportICE observes the two peer connections without replacing the SDK's own ICE
// callbacks. Sampling adds at most 20 ms to the reported connection time.
func (e *Edge) reportICE(ctx context.Context, started time.Time) {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	timeout := time.NewTimer(30 * time.Second)
	defer timeout.Stop()
	timeoutCh := timeout.C
	var publisherMs, subscriberMs float64
	for {
		if pc := e.call.PublisherPC(); pc != nil && publisherMs == 0 && iceConnected(pc.ICEConnectionState()) {
			publisherMs = float64(time.Since(started).Microseconds()) / 1000
			e.logger.Info("ice connection timing", "peer", "publisher", "connected_ms", publisherMs)
			timeout.Stop()
			timeoutCh = nil
		}
		if pc := e.call.SubscriberPC(); pc != nil && subscriberMs == 0 && iceConnected(pc.ICEConnectionState()) {
			subscriberMs = float64(time.Since(started).Microseconds()) / 1000
			e.logger.Info("ice connection timing", "peer", "subscriber", "connected_ms", subscriberMs)
		}
		if publisherMs > 0 && subscriberMs > 0 {
			return
		}
		select {
		case <-ticker.C:
		case <-timeoutCh:
			e.logger.Warn("publisher ice connection did not complete")
			return
		case <-ctx.Done():
			return
		case <-e.leftDone:
			return
		}
	}
}

func iceConnected(state webrtc.ICEConnectionState) bool {
	return state == webrtc.ICEConnectionStateConnected || state == webrtc.ICEConnectionStateCompleted
}

// connect builds the SDK client, preferring a token over a secret.
//
// The coordinator websocket stays on even though the agent reads none of its events: it is
// what registers the agent as a user, and the coordinator refuses to let a user it has never
// seen join a call.
func (e *Edge) connect() (*rtc.Client, error) {
	user := rtc.User{ID: e.options.User.ID, Name: e.options.User.Name}
	if user.Name == "" {
		user.Name = user.ID
	}

	if e.options.UserToken != "" {
		client, err := rtc.NewClient(e.options.APIKey, user, rtc.StaticToken(e.options.UserToken))
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("streamedge: connect: %w", err))
		}
		return client, nil
	}

	options := []rtc.ClientOption{rtc.WithUser(user)}
	if e.options.BaseURL != "" {
		options = append(options, getstream.WithBaseUrl(e.options.BaseURL))
	}
	if e.options.HTTPClient != nil {
		options = append(options, getstream.WithHTTPClient(e.options.HTTPClient))
	}
	client, err := rtc.NewRTCClient(e.options.APIKey, e.options.APISecret, options...)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("streamedge: connect: %w", err))
	}
	return client, nil
}

// publish adds the agent's Opus track. The track starts pulling frames from the speaker as
// soon as the transceiver is bound.
func (e *Edge) publish() error {
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
		return stack.Wrap(fmt.Errorf("streamedge: build audio track: %w", err))
	}
	if _, err := e.call.AddTrack(info, voice); err != nil {
		return stack.Wrap(fmt.Errorf("streamedge: publish audio track: %w", err))
	}
	return nil
}

// listen decodes one participant's track into the audio the agent listens to. Reading the
// track is also what pulls media through the receiver, so a track nobody reads is a track
// that never arrives.
func (e *Edge) listen(remote rtc.OnTrackReceived) {
	if remote.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
		return
	}
	// Subscribing skips the agent's own user, but the SFU forwards what it forwards. A
	// second instance of the same agent left behind in the call arrives here and is heard
	// as a caller, and the two then answer each other until nobody else can get a word in.
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

	codec := remote.Track.Codec().RTPCodecCapability
	reader, err := audiortc.NewRTPReader(
		remote.Track, codec,
		audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}},
	)
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

// watchForNewTracks subscribes to whatever is published after the agent joined, which is
// how someone who joins later gets heard.
func (e *Edge) watchForNewTracks(ctx context.Context) {
	unregister := rtc.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_TrackPublished) {
		published := event.TrackPublished
		if published.GetUserId() == e.options.User.ID {
			return
		}
		if published.GetType() != sfu_models.TrackType_TRACK_TYPE_AUDIO {
			return
		}

		e.mu.Lock()
		if e.left {
			e.mu.Unlock()
			return
		}
		e.subscribed = append(e.subscribed, &signal_rpc.TrackSubscriptionDetails{
			UserId:    published.GetUserId(),
			SessionId: published.GetSessionId(),
			TrackType: published.GetType(),
		})
		subscriptions := slices.Clone(e.subscribed)
		e.mu.Unlock()

		if err := e.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
			e.logger.Error("could not subscribe to a new track",
				"participant", published.GetUserId(), "error", err)
		}
	})

	e.mu.Lock()
	e.unregisters = append(e.unregisters, unregister)
	e.mu.Unlock()
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

// audioSubscriptions asks for the audio every other participant is already publishing.
// Video is deliberately left alone: the agent listens and talks.
func audioSubscriptions(state *sfu_models.CallState, selfUserID string) []*signal_rpc.TrackSubscriptionDetails {
	var subscriptions []*signal_rpc.TrackSubscriptionDetails
	for _, participant := range state.GetParticipants() {
		if participant.GetUserId() == selfUserID {
			continue
		}
		for _, trackType := range participant.GetPublishedTracks() {
			if trackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
				continue
			}
			subscriptions = append(subscriptions, &signal_rpc.TrackSubscriptionDetails{
				UserId:    participant.GetUserId(),
				SessionId: participant.GetSessionId(),
				TrackType: trackType,
			})
		}
	}
	return subscriptions
}
