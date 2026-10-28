package channelbridge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"regexp"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// inboundLease is how far a worker pushes a queued message's next attempt when it takes it, so
// no other router takes it while it is written: longer than one attempt's writeTimeout, so a
// live worker never loses one, and short enough that one a stopped router held is written
// again within a minute. eventforward's lease, a choice, not a measurement.
//
// It is also the longest a worker waits between looks for queued messages (writeDue).
const inboundLease = time.Minute

// defaultInboundPoll is the shortest wait before a worker looks again while messages are
// queued, besides the wake-ups of a finished write. The shortest wait in defaultInboundRetries
// is 2 s, so a retry goes out at most a second late. eventforward's, a choice.
const defaultInboundPoll = time.Second

// inboundInFlight is how many queued messages one router writes at once. A thread's messages
// are written one at a time whatever it is (store.ClaimChannelInbound). eventforward's inFlight,
// a choice, not a measurement.
const inboundInFlight = 16

// pruneEvery is how often a worker deletes the rows past their keep: a day for a written
// message, 30 days for a given-up one (store.PruneChannelInbound), so an hour late is nothing.
// A choice.
const pruneEvery = time.Hour

// defaultInboundRetries are the waits before each write again of a queued message whose write
// failed: five more within about 13 minutes, spaced so a Stream Chat that answered 5xx or 429
// has time to recover. Then the message is given up, except an opt-out (optsOut), which is
// written again after the last wait until it is recorded. A choice, not a vendor's figure.
var defaultInboundRetries = []time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second, 2 * time.Minute, 10 * time.Minute}

// maxMessageID is the longest message id Stream Chat takes: «message.id must be a maximum of
// 255 characters in length», its answer to a longer one (staging app, 2026-10-27).
const maxMessageID = 255

// readableID is a provider message id written into a Stream message id as it is: the
// characters every provider id seen so far keeps to besides base64's + and /, which Stream
// Chat was not tried with. Stream Chat took «.» and «-» in an id (staging app, 2026-10-27),
// and «_»: probe-dupid-a900e9ce-under_score_a_b was written, then refused with code 4 as
// already there (staging app 1257545, 2026-10-28 05:20:35 UTC).
var readableID = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// Start runs the worker that writes the queued inbound messages until Close, and hands each one
// it writes to answer, the one way a channel message reaches the agent; nil answers none. It
// looks at start, for the ones a router queued before it stopped, then when a write finishes or
// wake says a message was queued, at the first one due, and at least once a lease: with nothing
// queued it sends Postgres 2 queries a minute.
func (b *Bridge) Start(answer Answer) {
	b.answer = answer
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		timer := time.NewTimer(0)
		defer timer.Stop()
		for {
			wait, again := b.writeDue()
			timer.Stop()
			if again {
				timer.Reset(wait)
			}
			select {
			case <-b.ctx.Done():
				return
			case <-timer.C:
			case <-b.kick:
			}
		}
	}()
}

// wake tells the worker there may be a message to write, without waiting.
func (b *Bridge) wake() {
	select {
	case b.kick <- struct{}{}:
	default:
	}
}

// writeDue takes as many due messages as there are free slots and writes each on its own
// goroutine, until none is due or no slot is free. It returns how long to wait before looking
// again, with again false when only a finished write needs to wake the worker: every slot is
// busy. While messages are queued it waits until the first is due, its retry or another
// router's lease running out, and at least the poll; a lease at most, also when none is
// queued: a router that queued or leased one and then stopped wakes nobody else.
func (b *Bridge) writeDue() (wait time.Duration, again bool) {
	b.prune()
	for {
		if b.ctx.Err() != nil {
			return 0, false
		}
		free := cap(b.slots) - len(b.slots)
		if free == 0 {
			return 0, false
		}
		now := time.Now()
		claimed, err := b.store.ClaimChannelInbound(b.ctx, now, free, now.Add(inboundLease))
		if err != nil {
			if b.ctx.Err() == nil {
				b.logger.Error("could not take the inbound messages due", "error", err)
			}
			return b.inboundPoll, true
		}
		for _, message := range claimed {
			b.slots <- struct{}{}
			b.working.Add(1)
			go func() {
				defer b.working.Done()
				defer func() {
					<-b.slots
					b.wake()
				}()
				b.attempt(message)
			}()
		}
		if len(claimed) < free {
			break
		}
	}
	next, queued, err := b.store.NextChannelInboundAt(b.ctx)
	switch {
	case err != nil:
		if b.ctx.Err() == nil {
			b.logger.Error("could not find when the next inbound message is due", "error", err)
		}
		return b.inboundPoll, true
	case !queued:
		return inboundLease, true
	}
	return min(max(time.Until(next), b.inboundPoll), inboundLease), true
}

// prune deletes the rows past their keep, once every pruneEvery. Only the worker's goroutine
// calls it.
func (b *Bridge) prune() {
	if time.Since(b.pruned) < pruneEvery {
		return
	}
	b.pruned = time.Now()
	if _, err := b.store.PruneChannelInbound(b.ctx, b.pruned); err != nil && b.ctx.Err() == nil {
		b.logger.Error("could not delete the inbound messages past their keep", "error", err)
	}
}

// attempt writes a claimed message once and records what came of it: done, due again after
// the next wait, or given up when the waits ran out or Stream Chat refused it for good. An
// opt-out is never given up: once the waits ran out it is due again after the last one. Close
// does not cut it short, so a message is never left half written by a router that stops; its
// lease covers writeTimeout.
func (b *Bridge) attempt(message store.ChannelInbound) {
	writing, cancel := context.WithTimeout(context.WithoutCancel(b.ctx), writeTimeout)
	err := b.writeQueued(writing, message)
	cancel()
	recording, done := context.WithTimeout(context.WithoutCancel(b.ctx), writeTimeout)
	defer done()
	log := b.logger.With("connector", message.ConnectorID, "customer", message.CustomerID, "channel", message.ChannelID,
		"seq", message.Seq)
	attempts := message.Attempts + 1
	var changed bool
	switch {
	case err == nil:
		changed, err = b.store.FinishChannelInbound(recording, message, time.Now())
		if err == nil && changed {
			log.Info("an inbound message is done", "attempts", attempts)
		}
	case (attempts <= len(b.inboundRetries) || optsOut(message.Message())) && !refusedForGood(err):
		log.Info("an inbound message was not written into its thread channel and is written again", "attempts", attempts, "error", err)
		if attempts > len(b.inboundRetries) {
			log.Warn("an opt-out is still not recorded and is tried again", "attempts", attempts, "error", err)
		}
		wait := b.inboundRetries[min(attempts, len(b.inboundRetries))-1]
		changed, err = b.store.RetryChannelInbound(recording, message, time.Now().Add(wait))
	default:
		cause := err
		changed, err = b.store.FailChannelInbound(recording, message, time.Now(), b.forwardUnanswered(message))
		if changed {
			log.Warn("gave up an inbound message: it was never written into its thread channel, so no agent answers it",
				"attempts", attempts, "error", cause)
		}
	}
	switch {
	case err != nil:
		log.Error("could not record an inbound message's attempt", "error", err)
	case !changed:
		// Its lease ran out while it was written and another worker took it: that one records it.
		log.Info("an inbound message's lease ran out before its attempt was recorded", "attempts", attempts)
	}
}

// writeQueued admits a queued message (admit), then writes it into its thread channel under its
// own Stream id, and the card of the episode it opens, and hands it to the agent (answer).
// Either write already done by an attempt whose answer was lost counts as written; the
// message is then rebuilt from its row. A message admit holds back, such as a keyword or one
// the gate refuses, is done with nothing written. The message is done only once its turn is
// claimed, so a router that stops between the write and the claim leaves it queued, and the
// next attempt finds it written and claims it.
func (b *Bridge) writeQueued(ctx context.Context, message store.ChannelInbound) error {
	thread, err := b.store.ChannelThread(ctx, message.ChannelID)
	if err != nil {
		return err
	}
	admitted, err := b.admit(ctx, thread, message)
	if err != nil || !admitted {
		return err
	}
	config := store.AgentConfig{ID: message.ConfigID, Name: message.ConfigName}
	inbound := message.Message()
	opened, err := b.episode(ctx, thread, config, inbound, message.CreatedAt)
	if err != nil {
		return err
	}
	// An earlier attempt may have opened the episode and stopped before its card was written:
	// it failed, its router stopped or its lease ran out. This Open then finds it open. The
	// worker opens a message's episode at the time the message was queued, so an episode that
	// started then is one this message opened; an earlier message's episode started at that
	// message's time, and has its card. Both times are Postgres's, so no two clocks are
	// compared. The card's id is the episode's, so writing it again is one card.
	if opened.Episode.ID != "" && opened.Episode.StartedAt.Equal(message.CreatedAt) {
		opened.New = true
	}
	id := inboundMessageID(thread.ChannelID, inbound.ProviderMessageID)
	written, err := b.writeInto(ctx, thread, config, inbound, id)
	switch {
	case alreadyWritten(err):
		written = rewritten(thread, inbound, id)
	case err != nil:
		return err
	}
	log := b.logger.With("connector", message.ConnectorID, "customer", message.CustomerID, "channel", message.ChannelID,
		"seq", message.Seq)
	log.Info("an inbound message is written into its thread channel", "message", id)
	if err := b.cards.Write(ctx, opened); err != nil && !alreadyWritten(err) {
		return err
	}
	if b.answer == nil {
		return nil
	}
	turn, err := b.answer(ctx, thread, written)
	if err != nil || turn == nil {
		return err
	}
	log.Info("an inbound message is handed to the agent", "message", id)
	// Off the worker's slot: a turn waits for the thread's earlier one and for the reply, up to
	// minutes, and the thread's next message waits for this row only. Close waits for it.
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		turn()
	}()
	return nil
}

// forwardUnanswered hands a given-up message's delivery to the customer's event destinations of
// unhandled events (Options.Unanswered), for store.FailChannelInbound to call: only the first
// given-up message of a delivery whose forward is queued does, so two of them are one forward.
// Nil without Options.Unanswered.
func (b *Bridge) forwardUnanswered(message store.ChannelInbound) func(context.Context) error {
	if b.unanswered == nil || message.Forward == nil {
		return nil
	}
	event := eventforward.Event{
		CustomerID: message.CustomerID, ConnectorID: message.ConnectorID, ID: message.Forward.ID,
		Headers: message.Forward.Headers, Body: message.Raw,
	}
	if message.Forward.HeadersUntil != nil {
		event.HeadersUntil = *message.Forward.HeadersUntil
	}
	return func(ctx context.Context) error {
		err := b.unanswered(ctx, event)
		if err != nil {
			b.logger.Error("could not forward a delivery no agent got", "connector", message.ConnectorID,
				"customer", message.CustomerID, "error", err)
		}
		return err
	}
}

// inboundMessageID is the Stream message id of a queued message: the thread channel's id and
// the provider's id for the message, so the provider's id is one message in its thread and a
// second write of it is refused (alreadyWritten). Stream Chat keeps a message id unique in the
// whole app, so the thread channel is part of it. A provider id longer than maxMessageID
// leaves room for, or with a character Stream Chat was not tried with, is its SHA-256 in hex
// after «=», which no readable id holds: «thread-<uuid>=<64 hex>» is 108 characters.
func inboundMessageID(channelID, providerMessageID string) string {
	if readableID.MatchString(providerMessageID) && len(channelID)+1+len(providerMessageID) <= maxMessageID {
		return channelID + "-" + providerMessageID
	}
	sum := sha256.Sum256([]byte(providerMessageID))
	return channelID + "=" + hex.EncodeToString(sum[:])
}

// refusedForGood is whether Stream Chat answered a write with a refusal a later attempt gets
// too: a 4xx other than 429, as the bridge counts a reply's (retryable). No answer, a 5xx or a
// 429 may pass later (RFC 9110 section 15.6, RFC 6585 section 4).
func refusedForGood(err error) bool {
	var refused *getstream.StreamError
	return errors.As(err, &refused) && refused.StatusCode >= http.StatusBadRequest &&
		refused.StatusCode < http.StatusInternalServerError && refused.StatusCode != http.StatusTooManyRequests
}

// alreadyWritten is whether Stream Chat refused a message because it already holds one with its
// id: a write repeated after an attempt whose answer was lost. Stream Chat answers it with HTTP
// 400, code 4 and «a message with ID <id> already exists» (a second SendMessage with the same id
// on the staging app, 2026-10-27 23:49 UTC). Code 4 alone is any input Stream refuses, so the
// text decides.
func alreadyWritten(err error) bool {
	var refused *getstream.StreamError
	return errors.As(err, &refused) && refused.StatusCode == http.StatusBadRequest && refused.Code == 4 &&
		strings.Contains(refused.Message, "a message with ID ") && strings.Contains(refused.Message, " already exists")
}
