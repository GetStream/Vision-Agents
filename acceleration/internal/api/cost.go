package api

import (
	"context"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// What an answer to a customer's backend says of what it cost, and of the credit its
// organization has left.
const (
	headerCostMicros     = "X-Stream-Cost-Micros"
	headerCostEstimated  = "X-Stream-Cost-Estimated"
	headerBalanceMicros  = "X-Stream-Balance-Micros"
	headerBalanceWarning = "X-Stream-Balance-Warning"

	balanceLow       = string(policy.BalanceLow)
	balanceExhausted = string(policy.BalanceExhausted)
)

// costed is what withCost keeps for one request: what its rows cost, and whether the answer
// is to say nothing of it.
type costed struct {
	spent *routing.Spent
	off   atomic.Bool
}

type costedKey struct{}

// leaveUncosted has the answer say nothing of what the request cost, for an operation that
// answers before its work runs, as createResponse does: its cost comes with its turn.
func leaveUncosted(ctx context.Context) {
	if state, ok := ctx.Value(costedKey{}).(*costed); ok {
		state.off.Store(true)
	}
}

// withCost tells a customer's backend what each request cost, in X-Stream-Cost-Micros, the sum
// of the rows the request recorded, and X-Stream-Cost-Estimated when one of them was the
// router's own count rather than the vendor's; and what its organization has left, in
// X-Stream-Balance-Micros, with X-Stream-Balance-Warning once that is low or exhausted. A
// refusal for money says the balance too. Both are read from memory.
//
// Money is the organization's business, not its end users', so an end user's device is told
// none of it. A socket is told in its frames, so its writer is left alone; its context still
// adds up what it spent. It sits inside withCustomer, which says who the caller is.
func (s *Server) withCost(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, known := CustomerFrom(r.Context()); !known || !ServerSideFrom(r.Context()) {
			next.ServeHTTP(w, r)
			return
		}
		ctx, spent := routing.WithSpent(r.Context())
		state := &costed{spent: spent}
		r = r.WithContext(context.WithValue(ctx, costedKey{}, state))
		if r.Header.Get("Upgrade") != "" {
			next.ServeHTTP(w, r)
			return
		}
		next.ServeHTTP(&costedResponse{ResponseWriter: w, server: s, request: r, state: state}, r)
	})
}

// costedResponse sets the cost headers the moment the answer begins, when the work a
// request-and-answer operation does is done.
type costedResponse struct {
	http.ResponseWriter
	server  *Server
	request *http.Request
	state   *costed
	stamped bool
}

func (c *costedResponse) WriteHeader(code int) {
	c.stamp(code)
	c.ResponseWriter.WriteHeader(code)
}

func (c *costedResponse) Write(body []byte) (int, error) {
	c.stamp(http.StatusOK)
	return c.ResponseWriter.Write(body)
}

// Unwrap lets flushing and deadlines reach the writer underneath through
// http.ResponseController.
func (c *costedResponse) Unwrap() http.ResponseWriter { return c.ResponseWriter }

// stamp sets the headers the spec declares on an answer with code: the cost on a success, and
// the balance on a success and on a refusal for money.
func (c *costedResponse) stamp(code int) {
	if c.stamped {
		return
	}
	c.stamped = true
	success := code >= 200 && code < 300
	if !success && code != http.StatusPaymentRequired {
		return
	}
	header := c.ResponseWriter.Header()
	if micros, rows, estimated := c.state.spent.Total(); success && rows > 0 && !c.state.off.Load() {
		header.Set(headerCostMicros, strconv.FormatInt(micros, 10))
		if estimated {
			header.Set(headerCostEstimated, "true")
		}
	}
	appID, _ := CustomerFrom(c.request.Context())
	if balance, ok := c.server.policies.Balance(appID, OrganizationFrom(c.request.Context())); ok {
		header.Set(headerBalanceMicros, strconv.FormatInt(balance.AvailableMicros, 10))
		if balance.State != policy.BalanceOK {
			header.Set(headerBalanceWarning, string(balance.State))
		}
	}
}

// unitCost is what a backend's socket says each unit of its work cost, on the frame that ends
// the unit: what was recorded since the unit before. A session records a unit's row before it
// passes on the event that ends it. Nil, for an end user's device, says nothing.
type unitCost struct {
	spent *routing.Spent
	mu    sync.Mutex
	told  int64
}

// unitCostOf is the socket's unitCost, nil when its caller is told nothing of money.
func unitCostOf(ctx context.Context) *unitCost {
	if _, ok := ctx.Value(costedKey{}).(*costed); !ok {
		return nil
	}
	return &unitCost{spent: routing.SpentFrom(ctx)}
}

// stamp puts on the frame what was recorded since the last frame stamped.
func (u *unitCost) stamp(ended frame) {
	if u == nil {
		return
	}
	total, _, _ := u.spent.Total()
	u.mu.Lock()
	defer u.mu.Unlock()
	ended["cost_micros"] = total - u.told
	u.told = total
}

// stampSpent puts on the frame what one unit's own Spent recorded, for a socket whose units
// overlap, as an LLM socket's responses do.
func stampSpent(ended frame, spent *routing.Spent) {
	if spent == nil {
		return
	}
	micros, _, _ := spent.Total()
	ended["cost_micros"] = micros
}

// balanceEvery is how often a backend's socket looks at its organization's balance, which is
// read from memory.
const balanceEvery = time.Second

// watchBalance sends a backend's socket a balance frame, with what X-Stream-Balance-Micros and
// X-Stream-Balance-Warning would say, each time its organization's balance goes down into low
// or exhausted while the socket is open: once per crossing, and again after it has gone back
// up and down. One already low when the socket opens is told at once. An end user's device is
// told nothing. The returned func stops it and waits.
func (s *Server) watchBalance(ctx context.Context, out *socket, appID, organizationID string) func() {
	if unitCostOf(ctx) == nil {
		return func() {}
	}
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		// Nothing is said before the socket has said what it is.
		select {
		case <-ctx.Done():
			return
		case <-out.started:
		}
		ticker := time.NewTicker(balanceEvery)
		defer ticker.Stop()
		told := policy.BalanceOK
		for {
			if balance, ok := s.policies.Balance(appID, organizationID); ok {
				if balance.State.Below(told) {
					out.frame(frame{"type": "balance", "balance_micros": balance.AvailableMicros, "warning": string(balance.State)})
				}
				told = balance.State
			}
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
	return func() {
		cancel()
		<-done
	}
}

// spentMicros is what the request's rows cost so far, for an answer whose usage object says
// so, as X-Stream-Cost-Micros does: nil when it recorded nothing, and for an end user's device,
// which is told nothing of money.
func spentMicros(ctx context.Context) *int64 {
	state, ok := ctx.Value(costedKey{}).(*costed)
	if !ok {
		return nil
	}
	micros, rows, _ := state.spent.Total()
	if rows == 0 {
		return nil
	}
	return &micros
}
