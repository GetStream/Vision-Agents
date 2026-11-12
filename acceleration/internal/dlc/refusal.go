package dlc

import (
	"errors"
	"fmt"
)

const (
	codeOptedOut         = "opted_out"
	codeSandboxRecipient = "sandbox_recipient"
	codeSandboxMinutes   = "sandbox_minutes"
	codeSandboxMessages  = "sandbox_messages"
	codeCountry          = "country"
	codePerDay           = "per_day"
	codeRecipientsPerDay = "recipients_per_day"
	codeConcurrent       = "concurrent"
)

// refusal is an ErrRefused error that says which rule refused.
type refusal struct {
	code string
	err  error
}

func (r *refusal) Error() string { return r.err.Error() }
func (r *refusal) Unwrap() error { return r.err }

func refused(code string, err error) error { return &refusal{code: code, err: err} }

// NewRefusal is a refusal of the gate with the given code, for a test of what a caller does
// with one: the gate itself needs a database to refuse.
func NewRefusal(code string) error {
	return refused(code, fmt.Errorf("%w: %s", ErrRefused, code))
}

// RefusalCode is which rule of the gate refused err, or "" when err is no refusal of the gate
// with a code.
func RefusalCode(err error) string {
	var r *refusal
	if errors.As(err, &r) {
		return r.code
	}
	return ""
}

// RefusalResets reports whether a refusal of code stops holding by itself, so trying again
// later may pass: the limits of a day or of calls at once. A country, an opt-out or a sandbox
// recipient will not change.
func RefusalResets(code string) bool {
	switch code {
	case codeConcurrent, codePerDay, codeRecipientsPerDay, codeSandboxMinutes, codeSandboxMessages:
		return true
	}
	return false
}

// TierLimit is which of its organization's tier limits refused err: country, per_day,
// recipients_per_day or concurrent. It is "" for any other error, a refusal of the sandbox or of
// an opt-out among them.
func TierLimit(err error) string {
	switch code := RefusalCode(err); code {
	case codeCountry, codePerDay, codeRecipientsPerDay, codeConcurrent:
		return code
	}
	return ""
}
