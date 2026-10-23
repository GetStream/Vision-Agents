package api

import (
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
)

// moneyError is the answer for a refusal about money, and false for any other error. A
// suspension is the organization's, for every request, so it is a 403; the rest refuse only
// work that costs money, and are a 402 the caller fixes by paying or waiting for a reset.
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
	} {
		if found := refusalOf(err, refusal.kind); found != nil {
			return refusal.answer(strings.TrimPrefix(found.Error(), "policy: ")), true
		}
	}
	return APIError{}, false
}

// refusedRequest is a request routing refused, as the caller is told it: a refusal about money
// as moneyError says, and anything else as the caller's to change.
func refusedRequest(err error) APIError {
	if answer, ok := moneyError(err); ok {
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
