package tools

import (
	"context"
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

// NeedsApproval has a person allow each call before it runs, asked with question, such as
// "Book flight {flight_id}?", with each {argument} filled in from the call.
//
// On a voice call the router asks the caller out loud. They have 15 seconds from the end of
// the question, and only a yes sends the call here, on its own id and with Call.Approved
// set; its Timeout starts then. A speech-to-speech model asks in its own voice, and may say
// the question close to as written.
//
// In writing nobody is asked out loud, so the call is held where it would run. A session
// hands it over as a tool_call event and runs it only once AnswerApproval allows it; a
// call that is declined, or never answered, does not run. A dispatch worker has nobody to
// ask, so it refuses a call that arrives without the caller's yes.
//
// Only a call a socket delivers is held. Registry.Call and Registry.Run ask nobody.
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

// Authorize decides whether each call may run, before its arguments are checked and before
// the function does. It is where to ask who Call.User is and what they may do: the model
// asks for whatever the conversation talked it into. An error refuses the call, which never
// runs. The model is told only that the call was refused, unless the error is a UserError,
// whose message it is told.
func Authorize(allow func(ctx context.Context, call Call) error) Option {
	return func(function *Function) error {
		if allow == nil {
			return fmt.Errorf("an authorization needs a function to ask")
		}
		function.authorize = allow
		return nil
	}
}
