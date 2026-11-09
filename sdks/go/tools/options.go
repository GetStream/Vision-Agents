package tools

import (
	"fmt"
	"regexp"
	"strings"
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

// placeholder is one {argument} in an approval question.
var placeholder = regexp.MustCompile(`\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// NeedsApproval has the router ask a voice caller before each call runs, such as
// "Book flight {flight_id}?", with each {argument} filled in from the call. The caller
// answers out loud, and has 15 seconds from the end of the question: only a yes runs the
// function, on the call's own id and with Call.Approved set, and its Timeout starts then. A
// speech-to-speech model asks in its own voice, and may say the question close to as written.
//
// It is asked on voice calls only. In writing, the question travels with the tool for the app
// to show, as the session's approval, and the call is run as it comes.
func NeedsApproval(question string) Option {
	return func(function *Function) error {
		question = strings.TrimSpace(question)
		if question == "" {
			return fmt.Errorf("an approval needs a question")
		}
		properties, _ := function.Parameters["properties"].(map[string]any)
		for _, found := range placeholder.FindAllStringSubmatch(question, -1) {
			if _, ok := properties[found[1]]; !ok {
				return fmt.Errorf("the approval question names {%s}, which is not one of its arguments", found[1])
			}
		}
		function.Approval = question
		return nil
	}
}
