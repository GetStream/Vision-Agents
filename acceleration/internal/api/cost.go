package api

import (
	"context"
	"net/http"
	"strconv"
	"sync/atomic"

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
	c.stamp()
	c.ResponseWriter.WriteHeader(code)
}

func (c *costedResponse) Write(body []byte) (int, error) {
	c.stamp()
	return c.ResponseWriter.Write(body)
}

// Unwrap lets flushing and deadlines reach the writer underneath through
// http.ResponseController.
func (c *costedResponse) Unwrap() http.ResponseWriter { return c.ResponseWriter }

func (c *costedResponse) stamp() {
	if c.stamped {
		return
	}
	c.stamped = true
	header := c.ResponseWriter.Header()
	if micros, rows, estimated := c.state.spent.Total(); rows > 0 && !c.state.off.Load() {
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

// spentMicros is what the request's rows cost so far, for an answer whose usage object says
// so, or nil for an end user's device, which is told nothing of money.
func spentMicros(ctx context.Context) *int64 {
	state, ok := ctx.Value(costedKey{}).(*costed)
	if !ok {
		return nil
	}
	micros, _, _ := state.spent.Total()
	return &micros
}
