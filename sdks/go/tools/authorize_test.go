package tools

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"
)

// AuthorizeSuite is how a function's Authorize decides which calls run, and what a failed
// call leaves for the model to read.
type AuthorizeSuite struct {
	suite.Suite
	registry *Registry
	// ran counts the runs of refund.
	ran atomic.Int64
}

func TestAuthorizeSuite(t *testing.T) {
	suite.Run(t, new(AuthorizeSuite))
}

func (s *AuthorizeSuite) SetupTest() {
	s.registry = NewRegistry()
	s.ran.Store(0)
}

// errNotStaff is a refusal whose words are the caller's own business.
var errNotStaff = errors.New("user jean has role customer, refund needs role staff")

// staffOnly allows a call made for a user called ada, and refuses anybody else with refusal.
func staffOnly(refusal error) Option {
	return Authorize(func(_ context.Context, call Call) error {
		if call.User.ID == "ada" {
			return nil
		}
		return refusal
	})
}

// refund registers refund, which takes an amount from 1 to 100 and counts each run.
func (s *AuthorizeSuite) refund(options ...Option) {
	s.Require().NoError(Register(s.registry, "refund", "Refund an order", func(context.Context, struct {
		Amount int `json:"amount" minimum:"1" maximum:"100"`
	}) (any, error) {
		s.ran.Add(1)
		return "refunded", nil
	}, options...))
}

func (s *AuthorizeSuite) TestACallItsAuthorizeRefusesDoesNotRun() {
	s.refund(staffOnly(errNotStaff))

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", User: User{ID: "jean"}}, "refund", `{"amount": 20}`)

	s.Require().ErrorIs(err, errNotStaff)
	s.Zero(s.ran.Load())
}

func (s *AuthorizeSuite) TestACallItsAuthorizeAllowsRuns() {
	s.refund(staffOnly(errNotStaff))

	output, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", User: User{ID: "ada"}}, "refund", `{"amount": 20}`)

	s.Require().NoError(err)
	s.Equal("refunded", output.Text)
	s.EqualValues(1, s.ran.Load())
}

func (s *AuthorizeSuite) TestARefusalTellsTheModelOnlyThatTheCallWasRefused() {
	s.refund(staffOnly(errNotStaff))

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", User: User{ID: "jean"}}, "refund", `{"amount": 20}`)

	var said UserError
	s.Require().ErrorAs(err, &said)
	s.Equal(UserError("refund was refused"), said)
}

func (s *AuthorizeSuite) TestARefusalWrittenForTheModelIsWhatItIsTold() {
	s.refund(staffOnly(fmt.Errorf("checking jean: %w", UserError("refunds are for staff to make"))))

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", User: User{ID: "jean"}}, "refund", `{"amount": 20}`)

	var said UserError
	s.Require().ErrorAs(err, &said)
	s.Equal(UserError("refunds are for staff to make"), said)
	s.Zero(s.ran.Load())
}

func (s *AuthorizeSuite) TestAuthorizeIsAskedBeforeTheArgumentsAreChecked() {
	s.refund(staffOnly(errNotStaff))

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", User: User{ID: "jean"}}, "refund", `{"amount": 5000}`)

	// Somebody who may not call it learns nothing about what it takes.
	s.Require().ErrorIs(err, errNotStaff)
	s.NotContains(err.Error(), "maximum")
}

func (s *AuthorizeSuite) TestAnAuthorizationWithNothingToAskIsRefusedAtRegistration() {
	err := Register(s.registry, "refund", "Refund an order", func(context.Context, struct{}) (any, error) {
		return nil, nil
	}, Authorize(nil))

	s.Error(err)
	s.False(s.registry.Holds("refund"))
}

func (s *AuthorizeSuite) TestArgumentsTheSchemaRefusesAreWrittenForTheModel() {
	s.refund()

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1"}, "refund", `{"amount": 500}`)

	var said UserError
	s.Require().ErrorAs(err, &said)
	s.Contains(string(said), "/amount: maximum: got 500, want 100")
	s.Zero(s.ran.Load())
}

func (s *AuthorizeSuite) TestAFunctionsOwnFailureIsNotWrittenForTheModel() {
	s.Require().NoError(Register(s.registry, "lookup_order", "Look up an order", func(context.Context, struct{}) (any, error) {
		return nil, errors.New("pq: password authentication failed for user orders_rw")
	}))

	_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1"}, "lookup_order", "{}")

	var said UserError
	s.Require().Error(err)
	s.False(errors.As(err, &said))
}

func (s *AuthorizeSuite) TestACopyOfACallThatFailedForTheModelIsStillWrittenForIt() {
	s.Require().NoError(Register(s.registry, "lookup_order", "Look up an order", func(context.Context, struct{}) (any, error) {
		return nil, fmt.Errorf("orders: %w", UserError("there is no order 1042"))
	}))
	call := Call{ID: "call-1", SessionID: "session-1"}
	_, first := s.registry.Run(s.T().Context(), call, "lookup_order", "{}")
	s.Require().Error(first)

	// The router sends a call again when its first answer never arrived.
	_, again := s.registry.Run(s.T().Context(), call, "lookup_order", "{}")

	var said UserError
	s.Require().ErrorAs(again, &said)
	s.Equal(UserError("there is no order 1042"), said)
}

func (s *AuthorizeSuite) TestACopyOfACallThatFailedOtherwiseIsStillNotWrittenForTheModel() {
	s.Require().NoError(Register(s.registry, "lookup_order", "Look up an order", func(context.Context, struct{}) (any, error) {
		return nil, errors.New("pq: password authentication failed for user orders_rw")
	}))
	call := Call{ID: "call-1", SessionID: "session-1"}
	_, first := s.registry.Run(s.T().Context(), call, "lookup_order", "{}")
	s.Require().Error(first)

	_, again := s.registry.Run(s.T().Context(), call, "lookup_order", "{}")

	var said UserError
	s.Require().Error(again)
	s.False(errors.As(again, &said))
}
