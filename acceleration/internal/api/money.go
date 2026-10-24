package api

import (
	"errors"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
)

// moneyError is the answer for a refusal about money, and false for any other error. A
// suspension is the organization's, for every request, so it is a 403; a spent limit on one
// end user stops that user alone, so it is a 429 like the daily quota's, with Retry-After; a
// router that cannot share what it holds with the others is a 503 to retry shortly; the rest
// refuse only work that costs money, and are a 402 the caller fixes by paying or waiting for a
// reset.
func moneyError(err error) (APIError, bool) {
	for _, refusal := range []struct {
		kind   error
		answer func(message string) APIError
	}{
		{policy.ErrSuspended, func(message string) APIError {
			return APIError{Type: ErrorTypePermission, Code: codeAccountSuspended, Message: message}
		}},
		{policy.ErrNoCredits, func(message string) APIError { return paymentRequired(codeInsufficientCredits, message) }},
		{policy.ErrSystemLimit, func(message string) APIError { return paymentRequired(codeSystemLimitReached, message) }},
		{policy.ErrBudgetSpent, func(message string) APIError { return paymentRequired(codeSpendLimitReached, message) }},
		{policy.ErrUserLimit, func(message string) APIError {
			return APIError{Type: ErrorTypeRateLimited, Code: codeUserLimitReached, Message: message}
		}},
		{policy.ErrSpendCheckUnavailable, func(message string) APIError {
			return APIError{Type: ErrorTypeUnavailable, Code: codeSpendCheckUnavailable, Message: message}
		}},
	} {
		if found := refusalOf(err, refusal.kind); found != nil {
			return refusal.answer(strings.TrimPrefix(found.Error(), "policy: ")), true
		}
	}
	return APIError{}, false
}

// spendCheckRetry is how soon a caller refused because spend could not be shared is told to
// try again, in seconds.
const spendCheckRetry = "5"

// moneyHeaders is what a refusal about money answers beside its body: Retry-After, in seconds
// until the window ends, for a user's spent limit, and spendCheckRetry when spend could not be
// shared; nothing for the rest.
func moneyHeaders(err error) http.Header {
	if errors.Is(err, policy.ErrSpendCheckUnavailable) {
		return http.Header{"Retry-After": {spendCheckRetry}}
	}
	if !errors.Is(err, policy.ErrUserLimit) {
		return nil
	}
	resets, ok := policy.ResetsAt(err)
	if !ok {
		return nil
	}
	seconds := int64(math.Ceil(time.Until(resets).Seconds()))
	return http.Header{"Retry-After": {strconv.FormatInt(max(seconds, 1), 10)}}
}

// moneyAnswer is a refusal about money as an operation returns it, with its headers.
type moneyAnswer struct {
	APIError
	headers http.Header
}

func (m moneyAnswer) GetHeaders() http.Header { return m.headers }

// refusedRequest is a request routing refused, as the caller is told it: a refusal about money
// as moneyError says, and anything else as the caller's to change.
func refusedRequest(err error) error {
	if answer, ok := moneyError(err); ok {
		if headers := moneyHeaders(err); headers != nil {
			return moneyAnswer{APIError: answer, headers: headers}
		}
		return answer
	}
	return invalidRequest(err.Error())
}

// refusalOf is the error in err's tree that is kind or wraps it directly, which reads as the
// policy wrote it rather than with whatever the errors around it added; nil when there is none.
func refusalOf(err, kind error) error {
	if err == nil || !errors.Is(err, kind) {
		return nil
	}
	if err == kind {
		return err
	}
	switch wrapper := err.(type) {
	case interface{ Unwrap() []error }:
		for _, inner := range wrapper.Unwrap() {
			if found := refusalOf(inner, kind); found != nil {
				return found
			}
		}
	case interface{ Unwrap() error }:
		inner := wrapper.Unwrap()
		if inner == kind {
			return err
		}
		return refusalOf(inner, kind)
	}
	return nil
}
