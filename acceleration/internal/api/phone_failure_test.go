package api

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

func vendorRefusal(code string) error {
	refused := &phone.VendorError{Vendor: "twilio", Path: "/2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json",
		Status: http.StatusBadRequest, Code: code, Message: "the vendor's own words"}
	// Wrapped the way the service wraps what a provider returns.
	return stack.Wrap(fmt.Errorf("phone: buy: %w", refused))
}

func TestANumberThatNeedsAnAddressIsAnsweredInPlainWords(t *testing.T) {
	logged := &bytes.Buffer{}
	logger := slog.New(slog.NewTextHandler(logged, nil))
	failure := stack.Wrap(fmt.Errorf("%w: %w", phone.ErrAddressRequired, vendorRefusal("21631")))
	handler, _ := served(t, func() error { return phoneFailure(logger, failure) })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	answer := answered(t, response.Body.Bytes())
	require.Equal(t, "phone_number_needs_address", answer.Code)
	require.Equal(t, "This number needs a verified address, which is not supported. "+
		"Choose a number in the US or Canada, or contact support.", answer.Message)
	require.NotContains(t, response.Body.String(), "AC123")
}

func TestAVendorFailureIsLoggedAndAnsweredWithoutItsWords(t *testing.T) {
	logged := &bytes.Buffer{}
	logger := slog.New(slog.NewTextHandler(logged, nil))
	handler, _ := served(t, func() error { return phoneFailure(logger, vendorRefusal("20003")) })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusServiceUnavailable, response.Code)
	answer := answered(t, response.Body.Bytes())
	require.Equal(t, "phone_vendor_failed", answer.Code)
	require.Equal(t, "The phone vendor could not complete the request. "+
		"Try again, or contact support if it keeps failing.", answer.Message)
	for _, secret := range []string{"AC123", "/2010-04-01", "the vendor's own words", "20003"} {
		require.NotContains(t, response.Body.String(), secret)
	}
	for _, detail := range []string{"AC123", "20003", "the vendor's own words"} {
		require.Contains(t, logged.String(), detail, "the log keeps what debugging needs")
	}
}

func TestAFailedCallIsNotAnsweredWithTryAgain(t *testing.T) {
	handler, _ := served(t, func() error { return callFailure(slog.New(slog.DiscardHandler), vendorRefusal("20003")) })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusServiceUnavailable, response.Code)
	answer := answered(t, response.Body.Bytes())
	require.Equal(t, "phone_vendor_failed", answer.Code)
	require.Equal(t, "The phone vendor did not confirm the call, so it can still ring. "+
		"Do not place it again at once. Contact support if this keeps happening.", answer.Message)
	require.NotContains(t, response.Body.String(), "Try again")
}

func TestACallFailureOfOurOwnStillSaysWhatToFix(t *testing.T) {
	handler, _ := served(t, func() error {
		return callFailure(slog.New(slog.DiscardHandler), errors.New("phone: a call needs someone to call"))
	})

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Equal(t, "phone: a call needs someone to call", answered(t, response.Body.Bytes()).Message)
}

func TestAnErrorOfOurOwnStillSaysWhatToFix(t *testing.T) {
	handler, _ := served(t, func() error {
		return phoneFailure(slog.New(slog.DiscardHandler), errors.New("phone: a call needs someone to call"))
	})

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Equal(t, "phone: a call needs someone to call", answered(t, response.Body.Bytes()).Message)
}

func TestASkippedVendorIsAnsweredWithoutItsWords(t *testing.T) {
	logged := &bytes.Buffer{}
	logger := slog.New(slog.NewTextHandler(logged, nil))
	err := vendorRefusal("20003")
	skip := phone.Skip{Vendor: "twilio", Reason: err.Error(), Err: err}

	reason := skipReason(logger, skip)

	require.Equal(t, "The phone vendor could not complete the request.", reason)
	for _, secret := range []string{"AC123", "/2010-04-01", "the vendor's own words"} {
		require.NotContains(t, reason, secret)
		require.Contains(t, logged.String(), secret, "the log keeps what debugging needs")
	}
}

func TestASkipForOurOwnReasonKeepsItsWords(t *testing.T) {
	skip := phone.Skip{Vendor: "telnyx", Reason: "cannot search by region"}

	require.Equal(t, "cannot search by region", skipReason(slog.New(slog.DiscardHandler), skip))
}

func TestAVendorErrorSayingNotANumberIsNotTakenForANumberWeDoNotHold(t *testing.T) {
	vendors := stack.Wrap(&phone.VendorError{Vendor: "twilio", Path: "/2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json",
		Status: http.StatusNotFound, Message: "+15125551234 is not a number on this account"})
	ours := stack.Wrap(errors.New("phone: +15125551234 is not a number you hold"))

	require.False(t, isNumberNotHeld(vendors))
	require.True(t, isNumberNotHeld(ours))
	require.False(t, isNumberNotHeld(nil))
}
