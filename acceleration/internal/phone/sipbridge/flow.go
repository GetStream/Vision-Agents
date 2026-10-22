package sipbridge

import (
	"context"
	"fmt"
	"log/slog"
	"net/netip"
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

// MediaAddrs is where each side sends its audio from, as its SDP said while the call was
// set up. A re-INVITE later can move it.
type MediaAddrs struct {
	Customer, Stream string
}

// addrOf is the address an SDP body names, empty when it cannot be read.
func addrOf(body []byte) string {
	m, err := ParseMedia(body)
	if err != nil {
		return ""
	}
	return m.Addr
}

// flowA is for a customer trunk that accepts an INVITE without SDP. The customer makes the
// offer, so nothing happens in Stream until the phone is answered.
func flowA(ctx context.Context, cfg Config, customer, stream leg) (MediaAddrs, error) {
	log := cfg.Logger

	ringCtx, cancelRing := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelRing()
	log.Info("ringing the customer without an offer")
	offer, err := customer.Invite(ringCtx, nil)
	if err != nil {
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("customer leg: %w", err))
	}
	log.Info("customer answered", media(customerSide, offer)...)

	streamCtx, cancelStream := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelStream()
	answer, err := stream.Invite(streamCtx, offer)
	if err != nil {
		rejectCustomer(ctx, log, customer, offer)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream leg: %w", err))
	}
	log.Info("stream answered", media(streamSide, answer)...)

	if err := stream.Ack(ctx, nil); err != nil {
		rejectCustomer(ctx, log, customer, offer)
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream leg ack: %w", err))
	}
	if err := customer.Ack(ctx, answer); err != nil {
		hangUp(ctx, log, customer, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("customer leg ack: %w", err))
	}
	return MediaAddrs{Customer: addrOf(offer), Stream: addrOf(answer)}, nil
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
func flowB(ctx context.Context, cfg Config, customer, stream leg) (MediaAddrs, error) {
	log := cfg.Logger
	codecs, err := codecsFromNames(cfg.CustomerTrunk.Codecs)
	if err != nil {
		return MediaAddrs{}, err
	}
	streamSDP, customerSDP := newSDPSession(), newSDPSession()
	placeholder := placeholderOffer(cfg.FlowB.PlaceholderAddr, cfg.FlowB.PlaceholderPort, codecs)

	streamCtx, cancelStream := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelStream()
	log.Info("opening the session in stream with a placeholder address")
	answer, err := stream.Invite(streamCtx, streamSDP.build(placeholder))
	if err != nil {
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream leg: %w", err))
	}
	if err := stream.Ack(ctx, nil); err != nil {
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream leg ack: %w", err))
	}
	streamMedia, err := ParseMedia(answer)
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream answer: %w", err))
	}
	log.Info("stream answered", media(streamSide, answer)...)

	if _, err := stream.Reinvite(ctx, streamSDP.build(holdOffer(placeholder))); err != nil {
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("stream hold: %w", err))
	}
	log.Info("stream on hold while the phone rings")

	toCustomer, err := customerOffer(streamMedia)
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, err
	}
	ringCtx, cancelRing := context.WithTimeout(ctx, cfg.Call.RingTimeout)
	defer cancelRing()
	customerAnswer, err := customer.Invite(ringCtx, customerSDP.build(toCustomer))
	if err != nil {
		hangUp(ctx, log, nil, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("customer leg: %w", err))
	}
	if err := customer.Ack(ctx, nil); err != nil {
		hangUp(ctx, log, customer, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("customer leg ack: %w", err))
	}
	log.Info("customer answered", media(customerSide, customerAnswer)...)

	answered, err := ParseMedia(customerAnswer)
	if err == nil {
		var final Media
		if final, err = retargetOffer(answered, toCustomer); err == nil {
			_, err = stream.Reinvite(ctx, streamSDP.build(final))
		}
	}
	if err != nil {
		hangUp(ctx, log, customer, stream)
		return MediaAddrs{}, stack.Wrap(fmt.Errorf("pointing stream at the customer: %w", err))
	}
	log.Info("stream now sends RTP to the customer")
	return MediaAddrs{Customer: answered.Addr, Stream: streamMedia.Addr}, nil
}

// media is the log fields for one side's SDP body: where it sends RTP and what it speaks.
func media(s side, body []byte) []any {
	m, err := ParseMedia(body)
	if err != nil {
		return []any{"side", s.String(), "sdp_error", err.Error()}
	}
	return []any{
		"side", s.String(), "addr", m.Addr, "port", m.Port, "private", isPrivate(m.Addr),
		"codecs", codecNames(m.Codecs), "direction", m.Direction,
	}
}

var carrierGradeNAT = netip.MustParsePrefix("100.64.0.0/10")

// isPrivate is true for an address the other side cannot reach over the internet, the
// usual reason for one-way audio.
func isPrivate(addr string) bool {
	a, err := netip.ParseAddr(addr)
	if err != nil {
		return false
	}
	return a.IsPrivate() || a.IsLoopback() || a.IsLinkLocalUnicast() || carrierGradeNAT.Contains(a)
}

func codecNames(codecs []Codec) []string {
	names := make([]string, 0, len(codecs))
	for _, c := range codecs {
		names = append(names, c.Name)
	}
	return names
}
