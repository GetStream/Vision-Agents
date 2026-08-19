//go:build cgo && webrtc

package streamrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"sync"

	videosdk "github.com/GetStream/getstream-go-webrtc"
	sdktrack "github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/google/uuid"
	"github.com/livekit/media-sdk"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

type rtcCall struct {
	options Options
	logger  *slog.Logger
	recv    chan transport.Frame
	speaker *speaker

	client *videosdk.Client
	call   *videosdk.Call

	mu         sync.Mutex
	listening  map[string]*lkmedia.PCMRemoteTrack
	subscribed []*signal_rpc.TrackSubscriptionDetails
	unregister func()
	closed     bool
	pending    []int16
}

func join(ctx context.Context, options Options) (transport.Media, error) {
	if options.CallID == "" {
		return nil, errors.New("streamrtc: a call id is required")
	}
	if options.CallType == "" {
		options.CallType = defaultCallType
	}
	if options.UserID == "" {
		options.UserID = defaultUserID
	}
	if options.UserName == "" {
		options.UserName = options.UserID
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
		return nil, fmt.Errorf("streamrtc: %s is not set", apiKeyEnvVar)
	}
	if options.APISecret == "" && options.UserToken == "" {
		return nil, fmt.Errorf("streamrtc: set %s or %s", userTokenEnvVar, apiSecretEnvVar)
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	c := &rtcCall{
		options:   options,
		logger:    options.Logger.With("call", options.CallType+":"+options.CallID),
		recv:      make(chan transport.Frame, 64),
		speaker:   newSpeaker(),
		listening: map[string]*lkmedia.PCMRemoteTrack{},
	}
	if err := c.connect(ctx); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *rtcCall) connect(ctx context.Context) error {
	user := videosdk.User{ID: c.options.UserID, Name: c.options.UserName}
	var client *videosdk.Client
	var err error
	if c.options.UserToken != "" {
		client, err = videosdk.NewClient(c.options.APIKey, user, videosdk.StaticToken(c.options.UserToken))
	} else {
		client, err = videosdk.NewClientWithSecret(c.options.APIKey, c.options.APISecret, user)
	}
	if err != nil {
		return fmt.Errorf("streamrtc: connect: %w", err)
	}
	c.client = client
	c.call = client.Call(c.options.CallType, c.options.CallID)

	joined, err := c.call.Join(ctx, videosdk.WithOnTrack(videosdk.SubscriberFunc(func(remote videosdk.OnTrackReceived) {
		c.listen(remote)
	})))
	if err != nil {
		return fmt.Errorf("streamrtc: join %s:%s: %w", c.options.CallType, c.options.CallID, err)
	}

	c.mu.Lock()
	c.subscribed = audioSubscriptions(joined.GetCallState(), c.options.UserID)
	subscriptions := slices.Clone(c.subscribed)
	c.mu.Unlock()
	if err := c.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
		return fmt.Errorf("streamrtc: subscribe: %w", err)
	}
	c.watchForNewTracks(ctx)
	if err := c.publish(); err != nil {
		return err
	}
	c.logger.Info("joined the call", "user", c.options.UserID, "tracks", len(subscriptions))
	return nil
}

func (c *rtcCall) publish() error {
	info := &sfu_models.TrackInfo{
		TrackId:   uuid.NewString(),
		TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO,
	}
	voice, err := sdktrack.NewAudioTrack(info, c.speaker, webrtc.RTPCodecCapability{
		MimeType:  webrtc.MimeTypeOpus,
		ClockRate: opusSampleRate,
		Channels:  opusNegotiatedChannels,
	})
	if err != nil {
		return fmt.Errorf("streamrtc: build audio track: %w", err)
	}
	if _, err := c.call.AddTrack(info, voice); err != nil {
		return fmt.Errorf("streamrtc: publish audio track: %w", err)
	}
	return nil
}

func (c *rtcCall) listen(remote videosdk.OnTrackReceived) {
	if remote.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
		return
	}
	if string(remote.ParticipantID.UserID) == c.options.UserID {
		return
	}
	decoder, err := lkmedia.NewPCMRemoteTrack(remote.Track, &listener{call: c},
		lkmedia.WithTargetSampleRate(audio.Rate),
		lkmedia.WithTargetChannels(1),
	)
	if err != nil {
		c.logger.Error("could not decode agent audio", "error", err)
		return
	}
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		decoder.Close()
		return
	}
	if previous, ok := c.listening[remote.Track.ID()]; ok {
		previous.Close()
	}
	c.listening[remote.Track.ID()] = decoder
	c.mu.Unlock()
	c.logger.Info("listening to agent audio")
}

func (c *rtcCall) watchForNewTracks(ctx context.Context) {
	unregister := videosdk.HandleCallEvent(c.call, func(event *sfu_events.SfuEvent_TrackPublished) {
		published := event.TrackPublished
		if published.GetUserId() == c.options.UserID {
			return
		}
		if published.GetType() != sfu_models.TrackType_TRACK_TYPE_AUDIO {
			return
		}
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return
		}
		c.subscribed = append(c.subscribed, &signal_rpc.TrackSubscriptionDetails{
			UserId:    published.GetUserId(),
			SessionId: published.GetSessionId(),
			TrackType: published.GetType(),
		})
		subscriptions := slices.Clone(c.subscribed)
		c.mu.Unlock()
		if err := c.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
			c.logger.Error("could not subscribe to a new track", "error", err)
		}
	})
	c.mu.Lock()
	c.unregister = unregister
	c.mu.Unlock()
}

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

type listener struct {
	call *rtcCall
}

func (l *listener) WriteSample(sample media.PCM16Sample) error {
	return l.call.pushInbound(sample)
}

func (l *listener) Close() error { return nil }

func (c *rtcCall) pushInbound(sample media.PCM16Sample) error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.pending = append(c.pending, sample...)
	for len(c.pending) >= audio.FrameSamples {
		chunk := append([]int16(nil), c.pending[:audio.FrameSamples]...)
		c.pending = c.pending[audio.FrameSamples:]
		frame := transport.Frame{PCM: chunk}
		select {
		case c.recv <- frame:
		default:
		}
	}
	c.mu.Unlock()
	return nil
}

func (c *rtcCall) Send(pcm []int16) error {
	return c.speaker.Write(audio.PCM{Rate: audio.Rate, Samples: pcm})
}

func (c *rtcCall) Recv() <-chan transport.Frame { return c.recv }

func (c *rtcCall) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	unregister := c.unregister
	decoders := make([]*lkmedia.PCMRemoteTrack, 0, len(c.listening))
	for _, decoder := range c.listening {
		decoders = append(decoders, decoder)
	}
	c.listening = map[string]*lkmedia.PCMRemoteTrack{}
	c.mu.Unlock()

	if unregister != nil {
		unregister()
	}
	close(c.recv)
	for _, decoder := range decoders {
		decoder.Close()
	}
	var failures []error
	if err := c.speaker.Close(); err != nil {
		failures = append(failures, err)
	}
	if c.call != nil {
		if err := c.call.Leave("voicebench finished"); err != nil {
			failures = append(failures, err)
		}
	}
	if c.client != nil {
		c.client.Close()
	}
	return errors.Join(failures...)
}
