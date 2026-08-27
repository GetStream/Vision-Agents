// Package streamedge puts an agent in a Stream call.
//
// It is the transport half of internal/agent: everything about credentials, tracks,
// subscriptions and codecs lives here, so the agent itself only ever sees 16 kHz mono PCM
// in and out. Inbound Opus is decoded and resampled by media-sdk, which is the same path
// cmd/transcribe already uses for LiveKit; outbound PCM is encoded back to 48 kHz Opus for
// the track the agent publishes.
package streamedge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"sync"

	videosdk "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/google/uuid"
	"github.com/livekit/media-sdk"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
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
)

// defaultCallType is the call type a Stream app has out of the box.
const defaultCallType = "default"

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
	// CallType defaults to "default".
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

	client *videosdk.Client
	call   *videosdk.Call

	mu sync.Mutex
	// listening holds the decoder per subscribed track, so a track that goes away stops
	// being decoded.
	listening map[string]*lkmedia.PCMRemoteTrack
	// subscribed is the whole subscription list, because the SFU replaces it wholesale on
	// every update rather than adding to it.
	subscribed  []*signal_rpc.TrackSubscriptionDetails
	unregisters []func()
	left        bool

	leaveOnce sync.Once
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

	return &Edge{
		options:   options,
		logger:    options.Logger.With("call", options.CallType+":"+options.CallID),
		inbound:   emit.New[agent.InboundAudio](audioBuffer),
		attending: emit.New[agent.Attendance](attendanceBuffer),
		speaker:   newSpeaker(options.Logger),
		listening: map[string]*lkmedia.PCMRemoteTrack{},
	}, nil
}

// Join connects, subscribes to what the participants are already saying, and publishes the
// agent's own audio track.
func (e *Edge) Join(ctx context.Context) error {
	client, err := e.connect()
	if err != nil {
		return err
	}
	e.client = client

	e.call = client.Call(e.options.CallType, e.options.CallID)
	joined, err := e.call.Join(ctx, videosdk.WithOnTrack(videosdk.SubscriberFunc(func(remote videosdk.OnTrackReceived) {
		e.listen(remote)
	})))
	if err != nil {
		return fmt.Errorf("streamedge: join %s:%s: %w", e.options.CallType, e.options.CallID, err)
	}

	// Joining subscribes to nothing, so the SFU has to be told what to forward: whatever is
	// already being published, and then whatever is published later.
	e.mu.Lock()
	e.subscribed = audioSubscriptions(joined.GetCallState(), e.options.User.ID)
	subscriptions := slices.Clone(e.subscribed)
	e.mu.Unlock()

	if err := e.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
		return fmt.Errorf("streamedge: subscribe: %w", err)
	}
	e.watchForNewTracks(ctx)
	// Registered before the people already here are reported, so somebody arriving during
	// this is reported once rather than not at all.
	e.watchAttendance()
	e.reportPresent(joined.GetCallState())

	if err := e.publish(); err != nil {
		return err
	}

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

// Leave releases the call. It is safe to call more than once.
func (e *Edge) Leave() error {
	var err error
	e.leaveOnce.Do(func() { err = e.leave() })
	return err
}

// Call exposes the underlying call, so a caller can reach the SDK's own features.
func (e *Edge) Call() *videosdk.Call { return e.call }

func (e *Edge) leave() error {
	e.mu.Lock()
	e.left = true
	unregisters := e.unregisters
	e.unregisters = nil
	decoders := make([]*lkmedia.PCMRemoteTrack, 0, len(e.listening))
	for _, decoder := range e.listening {
		decoders = append(decoders, decoder)
	}
	e.listening = map[string]*lkmedia.PCMRemoteTrack{}
	e.mu.Unlock()

	for _, unregister := range unregisters {
		unregister()
	}
	// The channel closes before the decoders do, because closing a decoder waits for its
	// decode goroutine: one blocked handing over its last chunk would never be let go of.
	e.inbound.Close()
	e.attending.Close()
	for _, decoder := range decoders {
		decoder.Close()
	}

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

// connect builds the SDK client, preferring a token over a secret.
//
// The coordinator websocket stays on even though the agent reads none of its events: it is
// what registers the agent as a user, and the coordinator refuses to let a user it has never
// seen join a call.
func (e *Edge) connect() (*videosdk.Client, error) {
	user := videosdk.User{ID: e.options.User.ID, Name: e.options.User.Name}
	if user.Name == "" {
		user.Name = user.ID
	}

	if e.options.UserToken != "" {
		client, err := videosdk.NewClient(e.options.APIKey, user, videosdk.StaticToken(e.options.UserToken))
		if err != nil {
			return nil, fmt.Errorf("streamedge: connect: %w", err)
		}
		return client, nil
	}

	client, err := videosdk.NewClientWithSecret(e.options.APIKey, e.options.APISecret, user)
	if err != nil {
		return nil, fmt.Errorf("streamedge: connect: %w", err)
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
		return fmt.Errorf("streamedge: build audio track: %w", err)
	}
	if _, err := e.call.AddTrack(info, voice); err != nil {
		return fmt.Errorf("streamedge: publish audio track: %w", err)
	}
	return nil
}

// listen decodes one participant's track into the audio the agent listens to. Reading the
// track is also what pulls media through the receiver, so a track nobody reads is a track
// that never arrives.
func (e *Edge) listen(remote videosdk.OnTrackReceived) {
	if remote.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
		return
	}

	participant := stt.Participant{
		ID:     string(remote.ParticipantID.SessionID),
		UserID: string(remote.ParticipantID.UserID),
	}
	if remote.Participant != nil {
		participant.Name = remote.Participant.Name
	}

	decoder, err := lkmedia.NewPCMRemoteTrack(remote.Track,
		&listener{inbound: e.inbound, participant: participant},
		lkmedia.WithTargetSampleRate(stt.SampleRate),
		lkmedia.WithTargetChannels(1),
	)
	if err != nil {
		e.logger.Error("could not decode a participant's audio",
			"participant", participant.UserID, "error", err)
		return
	}

	e.mu.Lock()
	if e.left {
		e.mu.Unlock()
		decoder.Close()
		return
	}
	if previous, ok := e.listening[remote.Track.ID()]; ok {
		previous.Close()
	}
	e.listening[remote.Track.ID()] = decoder
	e.mu.Unlock()

	e.logger.Debug("listening to a participant", "participant", participant.UserID)
}

// watchForNewTracks subscribes to whatever is published after the agent joined, which is
// how someone who joins later gets heard.
func (e *Edge) watchForNewTracks(ctx context.Context) {
	unregister := videosdk.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_TrackPublished) {
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
	joined := videosdk.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_ParticipantJoined) {
		e.report(event.ParticipantJoined.GetParticipant(), true)
	})
	left := videosdk.HandleCallEvent(e.call, func(event *sfu_events.SfuEvent_ParticipantLeft) {
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

// listener hands one participant's decoded audio to the agent.
type listener struct {
	inbound     *emit.Emitter[agent.InboundAudio]
	participant stt.Participant
}

func (l *listener) WriteSample(sample media.PCM16Sample) error {
	// The decoder reuses its buffer, so what crosses to the agent has to be a copy.
	l.inbound.Send(agent.InboundAudio{
		Participant: l.participant,
		Audio: audio.PcmData{
			Samples:    slices.Clone(sample),
			SampleRate: stt.SampleRate,
			Channels:   1,
		},
	})
	return nil
}

func (l *listener) Close() error { return nil }
