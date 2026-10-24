package tools

import (
	"fmt"
	"time"
)

// Option is a policy one function asks for, given when it is registered.
type Option func(*Function) error

// Interruptible says whether the caller talking over the agent cancels a call. Off, which
// is the default, the call carries on and its result is given to the model when it comes.
func Interruptible(interruptible bool) Option {
	return func(function *Function) error {
		function.Interruptible = interruptible
		return nil
	}
}

// Timeout bounds one call, in place of the session's default.
func Timeout(timeout time.Duration) Option {
	return func(function *Function) error {
		if timeout <= 0 {
			return fmt.Errorf("a timeout must be positive, not %s", timeout)
		}
		function.Timeout = timeout
		return nil
	}
}

// Concurrency caps how many calls to this function run at once. A call over the cap waits
// for a place, and its timeout keeps counting while it does.
func Concurrency(calls int) Option {
	return func(function *Function) error {
		if calls <= 0 {
			return fmt.Errorf("concurrency must be at least one, not %d", calls)
		}
		function.Concurrency = calls
		return nil
	}
}
