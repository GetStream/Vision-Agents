package session

import (
	"context"
	"log/slog"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// callCardTimeout bounds writing one call's episode card: a call read, three Postgres
// statements and three Stream Chat writes, each of which chatlog gives 10 s (writeTimeout).
const callCardTimeout = 30 * time.Second

// callCards writes the episode card of each phone call a session joins into the caller's
// omni-channel (T41, AI-883): source call, the call id, and the call channel the transcript
// is in. It runs off the session's start, so a caller never waits on it, and a card that
// cannot be written is logged.
//
// The session's own conversation is left as it is: its transcript stays in the call channel.
// Holding the call on the omni-channel (Spec.ConversationID) would write the raw transcript
// there, which the card is there to keep out (channels.md, «Why the omni-channel gets
// summaries, not the raw text»).
type callCards struct {
	cards  *omnichannel.Cards
	logger *slog.Logger

	ctx     context.Context
	cancel  context.CancelFunc
	running sync.WaitGroup
}

// newCallCards is nil without a store or Stream clients, where no card can be kept.
func newCallCards(pgStore *store.Store, stream *streamapp.Clients, logger *slog.Logger) *callCards {
	cards, err := omnichannel.New(omnichannel.Options{Store: pgStore, Stream: stream})
	if err != nil {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &callCards{cards: cards, logger: logger, ctx: ctx, cancel: cancel}
}

// started writes the card of a call session that has just joined, when its agent config
// turned the cards on (store.AgentConfig.EpisodeCards). Off, which every config is unless it
// says otherwise, nothing runs: no call read, no contact map row, no card, as before the
// cards existed. A session in writing, an incognito one, which records nothing, and one with
// no agent config, whose omni-channel the contact map cannot key, get none either.
func (c *callCards) started(created *Session, stream streamapp.Bound) {
	spec := created.spec
	if c == nil || !spec.EpisodeCards || spec.Text || spec.CallID == "" || spec.Incognito || spec.ConfigID == "" || stream.Client == nil {
		return
	}
	c.running.Add(1)
	go func() {
		defer c.running.Done()
		ctx, cancel := context.WithTimeout(c.ctx, callCardTimeout)
		defer cancel()
		if err := c.write(ctx, created, stream); err != nil {
			c.logger.Error("could not write a call's episode card", "session", created.id, "call", spec.CallID, "error", err)
		}
	}()
}

// Close abandons the cards still being written and waits for them to stop.
func (c *callCards) Close() {
	if c == nil {
		return
	}
	c.cancel()
	c.running.Wait()
}

func (c *callCards) write(ctx context.Context, created *Session, stream streamapp.Bound) error {
	spec := created.spec
	number, err := calledParty(ctx, spec, stream)
	if err != nil || number == "" {
		return err
	}
	person, err := omnichannel.Phone(number)
	if err != nil {
		c.logger.Info("no episode card for a call whose number is not in E.164", "session", created.id)
		return nil
	}
	// The transcript's channel: the conversation the call was held on, else the agent id's
	// (chatlog: «Without it, the channel id is spec.AgentID»).
	thread := spec.ConversationID
	if thread == "" {
		thread = streamapp.AgentChannelType + ":" + spec.AgentID
	}
	opened, err := c.cards.Open(ctx, omnichannel.Episode{
		CustomerID:    spec.CustomerID,
		AgentConfigID: spec.ConfigID,
		AgentName:     spec.AgentName,
		Person:        person,
		Source:        store.EpisodeCall,
		ThreadChannel: thread,
		CallID:        spec.CallID,
		SessionID:     created.id,
		StartedAt:     created.created,
		StreamAppPK:   spec.StreamApp,
	})
	if err != nil {
		return err
	}
	return c.cards.Write(ctx, opened)
}

// calledParty is the number of the person on the call with the agent: who the agent rang,
// for a call it placed (PhoneSpec.To), otherwise the SIP caller the inbound routing rule named
// (phone.CallerNumber), read off the call's session. Empty for a call no phone is on.
func calledParty(ctx context.Context, spec Spec, stream streamapp.Bound) (string, error) {
	if spec.Phone != nil && spec.Phone.To != "" {
		return spec.Phone.To, nil
	}
	call, err := stream.Client.Video().GetCall(ctx, spec.CallType, spec.CallID, &getstream.GetCallRequest{})
	if err != nil {
		return "", err
	}
	if call.Data.Call.Session == nil {
		return "", nil
	}
	for _, participant := range call.Data.Call.Session.Participants {
		if number, ok := phone.CallerNumber(participant.User.ID); ok {
			return number, nil
		}
	}
	return "", nil
}
