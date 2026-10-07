package sipbridge

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// cleanupTimeout bounds the BYEs sent while giving up on a call.
const cleanupTimeout = 5 * time.Second

// cleanupContext outlives ctx, because the usual reason to clean up is that ctx was
// cancelled (Ctrl+C or the ring timeout) and the BYEs still have to go out.
func cleanupContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
}

// hangUp sends BYE on the legs that are not nil.
func hangUp(ctx context.Context, log *slog.Logger, customer, stream leg) {
	cctx, cancel := cleanupContext(ctx)
	defer cancel()
	for s, l := range [2]leg{customer, stream} {
		if l == nil {
			continue
		}
		if err := l.Bye(cctx); err != nil {
			log.Warn("cleanup BYE failed", "leg", side(s), "error", err)
		}
	}
}

// flowA is for a customer trunk that accepts an INVITE without SDP. The customer makes the
// offer, so nothing happens in Stream until the phone is answered.
func flowA(ctx context.Context, cfg Config, customer, stream leg) error {
	log := cfg.Logger

	ringCtx, cancelRing := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelRing()
	log.Info("ringing the customer without an offer")
	offer, err := customer.Invite(ringCtx, nil)
	if err != nil {
		return stack.Wrap(fmt.Errorf("customer leg: %w", err))
	}
	log.Info("customer answered", "offer", summary(offer))

	streamCtx, cancelStream := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelStream()
	answer, err := stream.Invite(streamCtx, offer)
	if err != nil {
		rejectCustomer(ctx, log, customer, offer)
		return stack.Wrap(fmt.Errorf("stream leg: %w", err))
	}
	log.Info("stream answered", "answer", summary(answer))

	if err := stream.Ack(ctx, nil); err != nil {
		rejectCustomer(ctx, log, customer, offer)
		hangUp(ctx, log, nil, stream)
		return stack.Wrap(fmt.Errorf("stream leg ack: %w", err))
	}
	if err := customer.Ack(ctx, answer); err != nil {
		hangUp(ctx, log, customer, stream)
		return stack.Wrap(fmt.Errorf("customer leg ack: %w", err))
	}
	return nil
}

// rejectCustomer ends a customer leg that answered with an offer we cannot connect. The
// customer is waiting for an answer in the ACK, so it gets one that rejects the stream.
func rejectCustomer(ctx context.Context, log *slog.Logger, customer leg, offer []byte) {
	cctx, cancel := cleanupContext(ctx)
	defer cancel()
	if err := customer.Ack(cctx, rejectingAnswer(offer)); err != nil {
		log.Warn("cleanup ACK failed", "leg", customerSide, "error", err)
	}
	if err := customer.Bye(cctx); err != nil {
		log.Warn("cleanup BYE failed", "leg", customerSide, "error", err)
	}
}

// rejectingAnswer is an SDP answer to offer that rejects its audio.
func rejectingAnswer(offer []byte) []byte {
	m, err := ParseMedia(offer)
	if err != nil {
		m = Media{}
	}
	return buildSDP(rejectAnswer(m), 1, 1)
}

// flowB opens the session in Stream first, because the customer trunk wants an offer in the
// INVITE and only Stream can give us an RTP address to offer.
func flowB(ctx context.Context, cfg Config, customer, stream leg) error {
	log := cfg.Logger
	codecs, err := codecsFromNames(cfg.CustomerTrunk.Codecs)
	if err != nil {
		return err
	}
	streamSDP, customerSDP := newSDPSession(), newSDPSession()
	placeholder := placeholderOffer(cfg.FlowB.PlaceholderAddr, cfg.FlowB.PlaceholderPort, codecs)

	streamCtx, cancelStream := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelStream()
	log.Info("opening the session in stream with a placeholder address")
	answer, err := stream.Invite(streamCtx, streamSDP.build(placeholder))
	if err != nil {
		return stack.Wrap(fmt.Errorf("stream leg: %w", err))
	}
	if err := stream.Ack(ctx, nil); err != nil {
		hangUp(ctx, log, nil, stream)
		return stack.Wrap(fmt.Errorf("stream leg ack: %w", err))
	}
	streamMedia, err := ParseMedia(answer)
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return stack.Wrap(fmt.Errorf("stream answer: %w", err))
	}
	log.Info("stream answered", "answer", summary(answer))

	if _, err := stream.Reinvite(ctx, streamSDP.build(holdOffer(placeholder))); err != nil {
		hangUp(ctx, log, nil, stream)
		return stack.Wrap(fmt.Errorf("stream hold: %w", err))
	}
	log.Info("stream on hold while the phone rings")

	toCustomer, err := customerOffer(streamMedia)
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return err
	}
	ringCtx, cancelRing := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelRing()
	customerAnswer, err := customer.Invite(ringCtx, customerSDP.build(toCustomer))
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return stack.Wrap(fmt.Errorf("customer leg: %w", err))
	}
	if err := customer.Ack(ctx, nil); err != nil {
		hangUp(ctx, log, customer, stream)
		return stack.Wrap(fmt.Errorf("customer leg ack: %w", err))
	}
	log.Info("customer answered", "answer", summary(customerAnswer))

	answered, err := ParseMedia(customerAnswer)
	if err == nil {
		var final Media
		if final, err = retargetOffer(answered, toCustomer); err == nil {
			_, err = stream.Reinvite(ctx, streamSDP.build(final))
		}
	}
	if err != nil {
		hangUp(ctx, log, customer, stream)
		return stack.Wrap(fmt.Errorf("pointing stream at the customer: %w", err))
	}
	log.Info("stream now sends RTP to the customer")
	return nil
}

// summary is a one-line view of an SDP body for the log.
func summary(body []byte) string {
	m, err := ParseMedia(body)
	if err != nil {
		return "unreadable sdp: " + err.Error()
	}
	return fmt.Sprintf("%s:%d %v %s", m.Addr, m.Port, codecNames(m.Codecs), m.Direction)
}

func codecNames(codecs []Codec) []string {
	names := make([]string, 0, len(codecs))
	for _, c := range codecs {
		names = append(names, c.Name)
	}
	return names
}
