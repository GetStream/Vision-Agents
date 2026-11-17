package routing

import (
	"context"
	"sync/atomic"
)

// Spent adds up what the rows recorded for one request, or one socket, cost, so the answer can
// say so. Memory only; the rows are what is billed.
type Spent struct {
	micros    atomic.Int64
	rows      atomic.Int64
	estimated atomic.Bool
}

type spentKey struct{}

// WithSpent has every row recorded for work started with the returned context add its cost to
// the returned Spent.
func WithSpent(ctx context.Context) (context.Context, *Spent) {
	spent := &Spent{}
	return context.WithValue(ctx, spentKey{}, spent), spent
}

// SpentFrom is the Spent WithSpent put in ctx, nil when nobody asked.
func SpentFrom(ctx context.Context) *Spent {
	spent, _ := ctx.Value(spentKey{}).(*Spent)
	return spent
}

// add counts one row of the cost, estimated when its quantities were the router's own count.
func (s *Spent) add(micros int64, estimated bool) {
	if s == nil {
		return
	}
	s.micros.Add(micros)
	if estimated {
		s.estimated.Store(true)
	}
	s.rows.Add(1)
}

// Total is what every row recorded so far cost, how many there were, and whether any was an
// estimate.
func (s *Spent) Total() (micros, rows int64, estimated bool) {
	if s == nil {
		return 0, 0, false
	}
	return s.micros.Load(), s.rows.Load(), s.estimated.Load()
}
