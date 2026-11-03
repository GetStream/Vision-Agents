package dlc

import "errors"

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
