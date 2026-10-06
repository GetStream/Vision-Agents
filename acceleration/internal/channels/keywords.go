package channels

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The words carriers expect a texting program to obey, whatever the agent would have said.
var (
	stopWords  = []string{"STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT", "OPTOUT", "REVOKE"}
	startWords = []string{"START", "UNSTOP"}
	helpWords  = []string{"HELP", "INFO"}
)

// What a keyword is answered with when the number's use case says nothing.
const (
	stoppedReply = "You are unsubscribed and will receive no more messages. Reply START to resubscribe."
	startedReply = "You are subscribed again. Reply STOP to unsubscribe."
	helpReply    = "Reply STOP to unsubscribe."
)

// keyword obeys STOP, START and HELP before any agent sees the message, and reports whether
// the message was one. The confirmation goes out past the gate: telling somebody they are
// unsubscribed is the one message they are still owed.
func (s *Service) keyword(ctx context.Context, account store.ChannelAccount, line Account, provider Provider, message Message) bool {
	word := strings.ToUpper(strings.TrimSpace(message.Text))
	stop, start, help := slices.Contains(stopWords, word), slices.Contains(startWords, word), slices.Contains(helpWords, word)
	if !stop && !start && !help {
		return false
	}

	useCase, err := s.store.UseCaseForNumber(ctx, account.CustomerID, account.E164)
	if err != nil && !errors.Is(err, store.ErrUnknownUseCase) {
		s.logger.Error("could not find what a line sends as", "number", account.E164, "error", err)
	}
	var reply string
	var recorded error
	switch {
	case stop:
		recorded = s.store.OptOut(ctx, &store.OptOut{
			CustomerID: account.CustomerID, Recipient: message.From, Channel: account.Kind, Source: "keyword",
		})
		reply = cmp.Or(useCase.OptOutMessage, stoppedReply)
	case start:
		recorded = s.store.RevokeOptOuts(ctx, account.CustomerID, message.From, account.Kind)
		reply = cmp.Or(useCase.OptInMessage, startedReply)
	default:
		reply = cmp.Or(useCase.HelpMessage, helpReply)
	}
	if recorded != nil {
		s.logger.Error("could not record a keyword", "keyword", word, "channel", account.Kind, "error", recorded)
		return true
	}
	s.say(ctx, line, provider, message, Reply{Text: reply})
	return true
}
