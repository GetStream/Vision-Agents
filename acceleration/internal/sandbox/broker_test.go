package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/suite"
)

type TokenSuite struct {
	suite.Suite
}

func TestTokenSuite(t *testing.T) {
	suite.Run(t, new(TokenSuite))
}

// TestATokenPrintsNoValue: every verb a log line or an error might use, and JSON.
func (s *TokenSuite) TestATokenPrintsNoValue() {
	token := Token{Value: "gho_secret", Revision: 3}
	encoded, err := json.Marshal(token)
	s.Require().NoError(err)

	s.NotContains(fmt.Sprintf("%v %+v %#v %s", token, token, token, token)+string(encoded), "gho_secret")
	s.Contains(fmt.Sprint(token), "Revision:3")
}

type ErrorClassSuite struct {
	suite.Suite
}

func TestErrorClassSuite(t *testing.T) {
	suite.Run(t, new(ErrorClassSuite))
}

func (s *ErrorClassSuite) TestAMissingCredentialIsNoCredential() {
	s.Equal("no_credential", ErrorClass(ErrNoCredential))
	s.Equal("no_credential", ErrorClass(fmt.Errorf("refresh: %w", ErrNoCredential)))
}

func (s *ErrorClassSuite) TestACanceledContextIsCanceled() {
	s.Equal("canceled", ErrorClass(context.Canceled))
	s.Equal("canceled", ErrorClass(fmt.Errorf("refresh: %w", context.Canceled)))
}

func (s *ErrorClassSuite) TestAnExpiredDeadlineIsDeadline() {
	s.Equal("deadline", ErrorClass(context.DeadlineExceeded))
	s.Equal("deadline", ErrorClass(fmt.Errorf("refresh: %w", context.DeadlineExceeded)))
}

// TestANetworkErrorIsNetwork: a *net.OpError is a net.Error, the kind a refused or reset
// connection to a provider gives.
func (s *ErrorClassSuite) TestANetworkErrorIsNetwork() {
	dial := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}

	s.Equal("network", ErrorClass(dial))
	s.Equal("network", ErrorClass(fmt.Errorf("refresh: %w", dial)))
}

// TestAnyOtherErrorIsOther: an error that is none of the above, such as a provider's answer.
func (s *ErrorClassSuite) TestAnyOtherErrorIsOther() {
	other := errors.New("invalid_grant: token for acct-1 revoked")

	s.Equal("other", ErrorClass(other))
	s.Equal("other", ErrorClass(fmt.Errorf("refresh: %w", other)))
}

// TestNoCredentialWinsOverTheCauseThatCameWithIt: a reconnect is what the user must do, whatever
// else the error carries.
func (s *ErrorClassSuite) TestNoCredentialWinsOverTheCauseThatCameWithIt() {
	err := fmt.Errorf("refresh: %w", errors.Join(context.Canceled, ErrNoCredential))

	s.Equal("no_credential", ErrorClass(err))
}

// TestACanceledContextWinsOverANetworkErrorAroundIt: a cancelled dial is a *net.OpError that
// wraps context.Canceled, and is the caller's doing, not the network's.
func (s *ErrorClassSuite) TestACanceledContextWinsOverANetworkErrorAroundIt() {
	canceled := &net.OpError{Op: "dial", Net: "tcp", Err: context.Canceled}
	expired := &net.OpError{Op: "dial", Net: "tcp", Err: context.DeadlineExceeded}

	s.Equal("canceled", ErrorClass(canceled))
	s.Equal("deadline", ErrorClass(expired))
}
