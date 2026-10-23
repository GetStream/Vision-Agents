package channels

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AttributeCategory is the WhatsApp message category a text is priced by. The agent writes
// only inside the 24 hours after somebody wrote, never a template, so all it sends is
// CategoryService.
const (
	AttributeCategory = "category"
	CategoryService   = "service"
)

// OriginNotice is the origin of a message sent first, rather than as an answer, when nothing
// says who asked for it.
var OriginNotice = routing.JobOrigin("channel_notice")

// Vendor is who carries a channel, which is who bills its messages.
func Vendor(kind Kind) string {
	switch kind {
	case WhatsApp:
		return "meta"
	case SMS:
		return "telnyx"
	case IMessage:
		return "linq"
	}
	return string(kind)
}

// sendFailed is the error code of a provider that would not send.
func sendFailed(error) string { return "send_failed" }

// textMeter records one message on a channel, to or from other, priced by the channel, its
// direction and where other is. An app the 10DLC sandbox holds has it priced and given away.
func (s *Service) textMeter(ctx context.Context, kind Kind, direction, other string, owner routing.Owner) routing.Meter {
	attributes := map[string]string{
		phone.AttributeDirection: direction,
		phone.AttributeCountry:   phone.Country(other),
	}
	if kind == WhatsApp && direction == store.Outbound {
		attributes[AttributeCategory] = CategoryService
	}
	meter := routing.Meter{
		Recorder:   s.recorder,
		Config:     routing.ProviderConfig{Provider: Vendor(kind), Model: string(kind)},
		Owner:      owner,
		Code:       sendFailed,
		Attributes: attributes,
	}
	held, err := s.gate.Sandboxed(ctx, owner.CustomerID)
	if err != nil {
		s.logger.Warn("could not tell whether the 10DLC sandbox holds an app; its texts are billed",
			"customer", owner.CustomerID, "error", err)
	}
	if held {
		meter.Free = true
		meter.Attributes[phone.AttributeDLCSandbox] = "true"
	}
	return meter
}

// received bills a message a provider delivered: it billed it whether or not the agent
// answers, and a delivery repeated names the same message, so it is billed once.
func (s *Service) received(ctx context.Context, account store.ChannelAccount, message Message) {
	kind := Kind(account.Kind)
	meter := s.textMeter(ctx, kind, store.Inbound, message.From, routing.Owner{
		CustomerID:  account.CustomerID,
		Attribution: routing.Attribution{APIKey: routing.OriginPhoneInbound},
	})
	meter.UsageID = routing.UsageID("text_in", Vendor(kind), message.ID)
	meter.Unmeasured = true
	got := Sent{Messages: 1}
	if kind == SMS {
		got.Segments = Segments(message.Text)
	}
	if err := routing.Settle(meter, got); err != nil {
		s.logger.Error("could not bill a message that arrived", "channel", kind, "error", err)
	}
}

// deliver sends a reply over a provider and records it, a failure included, for owner.
func (s *Service) deliver(ctx context.Context, line Account, provider Provider, to, thread string, reply Reply,
	owner routing.Owner) error {
	meter := s.textMeter(ctx, line.Kind, store.Outbound, to, owner)
	_, err := routing.Call(ctx, meter, func(ctx context.Context) (Sent, error) {
		return provider.Send(ctx, line, thread, reply)
	})
	return err
}
