package audioturn

import (
	"context"
	"errors"
	"time"
)

func ScoreAttempts(ctx context.Context, client *Client, requestID string, pcm []byte, primary bool) (Score, error, int, FailureClass, time.Duration, bool) {
	started := time.Now()
	budget, maxAttempts := GateLimit, 1
	if primary {
		budget, maxAttempts = PrimaryLimit, 1+eotPrimaryRetryLimit
	}
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	deadline, _ := ctx.Deadline()
	var lastErr error
	lastClass := FailureUnknown
	attempts := 0
	for attempts < maxAttempts && ctx.Err() == nil {
		attemptLimit := GateLimit
		if attempts > 0 {
			if time.Until(deadline) < eotPrimaryMinRetryWindow {
				break
			}
			attemptLimit = eotPrimaryRetryWindow
		}
		attemptCtx, stop := context.WithTimeout(ctx, attemptLimit)
		score, err := client.Score(attemptCtx, requestID, pcm)
		stop()
		attempts++
		if err == nil {
			return score, nil, attempts, "", time.Since(started), false
		}
		lastErr = err
		var failure *eotAttemptError
		typed := errors.As(err, &failure)
		if typed {
			lastClass = failure.class
		} else {
			lastClass = FailureUnknown
		}
		if attempts == maxAttempts || !typed || !failure.retryable() {
			break
		}
		delay := time.Duration(attempts) * 25 * time.Millisecond
		if failure.hasRetryAfter {
			delay = max(delay, failure.retryAfter)
		}
		if time.Until(deadline) < delay+eotPrimaryMinRetryWindow || !waitEOTRetry(ctx, delay) {
			break
		}
	}
	// Classify the context once, after the attempt or backoff that stopped the loop.
	stopped := ctx.Err()
	exhausted := errors.Is(stopped, context.DeadlineExceeded)
	if errors.Is(stopped, context.Canceled) {
		lastClass = FailureCanceled
		lastErr = &eotAttemptError{class: lastClass}
	} else if exhausted && (lastErr == nil || lastClass == FailureCanceled) {
		lastClass = FailureTimeout
		lastErr = &eotAttemptError{class: lastClass}
	}
	if lastErr == nil {
		lastErr = &eotAttemptError{class: lastClass}
	}
	return Score{}, lastErr, attempts, lastClass, time.Since(started), exhausted
}

func waitEOTRetry(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}
