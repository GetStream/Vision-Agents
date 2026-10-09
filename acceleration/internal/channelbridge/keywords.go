package channelbridge

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The words a texting program obeys whatever the agent would have said (T53, AI-881; wave 3c
// Q9: STOP, START and HELP are handled on the bridge, before the agent; the bridge answers
// only the ones the provider did not, episodeSource.answered). WhatsApp has them too (T51,
// AI-879): Meta's policy asks a business to «respect all requests (either on or off
// WhatsApp) by a person to block, discontinue, or otherwise opt out of communications from
// you via WhatsApp» and names no keyword Meta answers itself
// (https://whatsappbusiness.com/policy/, opened October 8, 2026),
// and internal/channels answers these words on its WhatsApp lines. iMessage through Linq has
// them too (T62a, AI-921), as internal/channels' iMessage lines do. Linq answers none of them
// itself: it refuses every later send to a person who texted «STOP» with 403, «including a
// final courtesy message», unless that one send sets override_optout, and lifts the block on
// the person's next message that is no stop keyword
// (https://docs.linqapp.com/channel/imessage/error/codes/2xxx/2024/index.md, opened October 8,
// 2026). A message is one of them when it is nothing else, read without its case or
// punctuation (keywordOf).
//
//   - stopWords: the CTIA Messaging Principles and Best Practices (May 2023), 5.1.3, «stop,
//     end, unsubscribe, cancel, quit»; the FCC's per se revocation by reply text, «'stop,'
//     'quit,' 'end,' 'revoke,' 'opt out,' 'cancel,' or 'unsubscribe'» (FCC 24-24, 89 FR 15756,
//     47 CFR 64.1200(a)(10),
//     https://www.govinfo.gov/content/pkg/FR-2024-03-05/html/2024-04587.htm); and Telnyx's
//     default opt-out words, which add STOPALL and STOP ALL.
//   - startWords and helpWords: Telnyx's default opt-in and help words, START, UNSTOP and HELP.
//
// Telnyx's page: https://developers.telnyx.com/docs/messaging/messages/advanced-opt-in-out
// (opened October 8, 2026). internal/channels/keywords.go keeps its own list until T62 moves
// its lines here.
var (
	stopWords  = []string{"STOP", "STOPALL", "STOP ALL", "END", "UNSUBSCRIBE", "CANCEL", "QUIT", "REVOKE", "OPT OUT"}
	startWords = []string{"START", "UNSTOP"}
	helpWords  = []string{"HELP"}
)

// What a keyword is answered with when the line's 10DLC use case says nothing
// (store.UseCaseForNumber: the use case its number is assigned to, else the customer's
// default): internal/channels/keywords.go's texts, so a person reads the same answer on either
// line. The CTIA's 5.1.3 asks for «one final opt-out confirmation message»; no other message
// follows it (Bridge.reply).
const (
	stoppedReply = "You are unsubscribed and will receive no more messages. Reply START to resubscribe."
	startedReply = "You are subscribed again. Reply STOP to unsubscribe."
	helpReply    = "Reply STOP to unsubscribe."
)

// optOutSource is opt_outs.source for a texted keyword (20261006120200_opt_outs.sql: «keyword
// (they texted STOP)»), as internal/channels records it.
const optOutSource = "keyword"

// keyword answers a carrier keyword before any agent sees the message, on a connector whose
// episode source names an opt-out channel, and keeps the messages of a person who opted out
// from the agent. handled is whether the message stops here. STOP and START are recorded in
// the customer's opt-outs (store.OptOut), the record dlc.Gate and the opt-out API read. The
// answer is the use case's text when the line's use case has one (useCase), as
// internal/channels answers, and it goes out past the sandbox gate: the confirmation is the
// one message a person is still owed. A keyword the provider answered itself is not answered
// again. A store that fails releases the message's claim, so the provider's next delivery of
// it is taken again: an opt-out is never lost to a retry.
func (b *Bridge) keyword(ctx context.Context, thread store.ChannelThread, message core.InboundMessage) (bool, error) {
	source := episodeSources[message.ConnectorID]
	channel := source.optOuts
	if channel == "" {
		return false, nil
	}
	word := keywordOf(message.Text)
	stop, start, help := slices.Contains(stopWords, word), slices.Contains(startWords, word), slices.Contains(helpWords, word)
	if !stop && !start && !help {
		optedOut, err := b.store.OptedOut(ctx, thread.CustomerID, source.recipientOf(message.AuthorID), channel)
		if err != nil {
			return false, b.unclaim(ctx, thread, message, err)
		}
		if optedOut {
			b.logger.Info("dropped an inbound message from a person who opted out",
				"connector", message.ConnectorID, "channel", thread.ChannelID)
		}
		return optedOut, nil
	}
	useCase := b.useCase(ctx, thread)
	var reply string
	var err error
	switch {
	case stop:
		err = b.store.OptOut(ctx, &store.OptOut{
			CustomerID: thread.CustomerID, Recipient: source.recipientOf(message.AuthorID), Channel: channel, Source: optOutSource,
		})
		reply = cmp.Or(useCase.OptOutMessage, stoppedReply)
	case start:
		err = b.store.RevokeOptOuts(ctx, thread.CustomerID, source.recipientOf(message.AuthorID), channel)
		reply = cmp.Or(useCase.OptInMessage, startedReply)
	default:
		reply = cmp.Or(useCase.HelpMessage, helpReply)
	}
	if err != nil {
		return false, b.unclaim(ctx, thread, message, err)
	}
	if source.answered != nil && source.answered(message.Raw) {
		return true, nil
	}
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		sending, cancel := context.WithTimeout(context.Background(), sendTimeout)
		defer cancel()
		if err := b.send(sending, thread, reply); err != nil {
			b.logger.Error("could not answer a keyword", "connector", thread.ConnectorID, "channel", thread.ChannelID, "error", err)
		}
	}()
	return true, nil
}

// telnyxAnswered is whether Telnyx answered a keyword itself: «When a user sends an opt-in,
// opt-out, or help keyword, the inbound message webhook includes an autoresponse_type field»,
// «STOP», «START» or «HELP» in the page's examples. Telnyx answers STOP, START and HELP
// whatever the customer configures («the defaults always remain active») and refuses a send
// to a number that texted STOP (40300, «Blocked due to STOP message»), so the bridge answers
// only the words Telnyx does not, such as REVOKE and OPT OUT. Page:
// https://developers.telnyx.com/docs/messaging/messages/advanced-opt-in-out (opened October 8,
// 2026). # unverified: whether the field is absent or null on a message Telnyx did not
// answer; either reads as not answered.
func telnyxAnswered(raw []byte) bool {
	var event struct {
		Data struct {
			Payload struct {
				AutoresponseType string `json:"autoresponse_type"`
			} `json:"payload"`
		} `json:"data"`
	}
	return json.Unmarshal(raw, &event) == nil && event.Data.Payload.AutoresponseType != ""
}

// unclaim releases a message's inbound claim after err, so its next delivery is taken again.
func (b *Bridge) unclaim(ctx context.Context, thread store.ChannelThread, message core.InboundMessage, err error) error {
	return errors.Join(err, b.store.ReleaseChannelThreadMessage(ctx, thread.ChannelID, store.ClaimInbound, message.ProviderMessageID))
}

// useCase is the 10DLC use case a thread's line sends as, whose texts answer a keyword
// (internal/channels/keywords.go reads it the same way). The line is the provider unit: the
// customer's number in E.164 on Telnyx and Linq (telnyx.yaml, linq.yaml). WhatsApp's is the
// business number's phone_number_id (whatsapp.yaml), which no number is stored under, so a
// WhatsApp line has the customer's default use case. None, or a store that fails, is the zero
// use case, whose empty texts leave today's; the keyword is answered either way.
func (b *Bridge) useCase(ctx context.Context, thread store.ChannelThread) store.UseCase {
	useCase, err := b.store.UseCaseForNumber(ctx, thread.CustomerID, thread.ProviderUnitID)
	if err != nil && !errors.Is(err, store.ErrUnknownUseCase) {
		b.logger.Error("could not find the use case a line sends as", "connector", thread.ConnectorID,
			"channel", thread.ChannelID, "error", err)
	}
	return useCase
}

// keywordOf is a message as a keyword: its words in upper case, without the punctuation
// around them. The CTIA's 5.1.3: an opt-out «should not be impacted by any de minimis
// variances ... such as capitalization, punctuation, or any letter-case sensitivities». So
// "Stop." and "opt-out" are STOP and OPT OUT, and "stop texting me" is no keyword.
func keywordOf(text string) string {
	words := strings.FieldsFunc(strings.ToUpper(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	return strings.Join(words, " ")
}
