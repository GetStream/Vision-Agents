package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type AccessCredentialSuite struct {
	suite.Suite
	credential AccessCredential
}

func TestAccessCredentialSuite(t *testing.T) {
	suite.Run(t, new(AccessCredentialSuite))
}

func (s *AccessCredentialSuite) SetupTest() {
	s.credential = NewAccessCredential("bearer", time.Unix(1_800_000_000, 0).UTC(), json.RawMessage(`"tok-very-secret"`))
}

func (s *AccessCredentialSuite) TestTheIssuingSchemeReadsTheSecretBack() {
	s.JSONEq(`"tok-very-secret"`, string(s.credential.Secret()))
}

func (s *AccessCredentialSuite) TestNoFormatVerbPrintsTheSecret() {
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		s.NotContains(fmt.Sprintf(verb, s.credential), "tok-very-secret", verb)
	}
}

func (s *AccessCredentialSuite) TestJSONLeavesTheSecretOut() {
	raw, err := json.Marshal(s.credential)
	s.Require().NoError(err)
	s.NotContains(string(raw), "tok-very-secret")
}

func (s *AccessCredentialSuite) TestALogLineLeavesTheSecretOut() {
	var text, structured strings.Builder
	slog.New(slog.NewTextHandler(&text, nil)).Info("resolved", "credential", s.credential)
	slog.New(slog.NewJSONHandler(&structured, nil)).Info("resolved", "credential", s.credential)
	s.NotContains(text.String(), "tok-very-secret")
	s.NotContains(structured.String(), "tok-very-secret")
	s.Contains(text.String(), "bearer")
}

type StoredCredentialsSuite struct {
	suite.Suite
	stored StoredCredentials
}

func TestStoredCredentialsSuite(t *testing.T) {
	suite.Run(t, new(StoredCredentialsSuite))
}

func (s *StoredCredentialsSuite) SetupTest() {
	s.stored = StoredCredentials{Scheme: "oauth2_code", Version: 1, Payload: json.RawMessage(`{"refresh_token":"rt-very-secret"}`)}
}

func (s *StoredCredentialsSuite) TestNoFormatVerbPrintsThePayload() {
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		out := fmt.Sprintf(verb, s.stored)
		s.NotContains(out, "rt-very-secret", verb)
		s.Contains(out, "oauth2_code", verb)
	}
}

func (s *StoredCredentialsSuite) TestALogLineLeavesThePayloadOut() {
	var text, structured strings.Builder
	slog.New(slog.NewTextHandler(&text, nil)).Info("rotated", "credentials", s.stored)
	slog.New(slog.NewJSONHandler(&structured, nil)).Info("rotated", "credentials", s.stored)
	s.NotContains(text.String(), "rt-very-secret")
	s.NotContains(structured.String(), "rt-very-secret")
	s.Contains(structured.String(), "oauth2_code")
}

type OutcomeErrorSuite struct {
	suite.Suite
}

func TestOutcomeErrorSuite(t *testing.T) {
	suite.Run(t, new(OutcomeErrorSuite))
}

func (s *OutcomeErrorSuite) TestTheResolverReadsTheOutcomeThroughAWrappedError() {
	cause := errors.New("token endpoint answered 400 invalid_grant")
	err := fmt.Errorf("access credential: %w", &OutcomeError{Outcome: Outcome{Kind: OutcomeInvalidGrant}, Err: cause})
	var failed *OutcomeError
	s.Require().ErrorAs(err, &failed)
	s.Equal(OutcomeInvalidGrant, failed.Outcome.Kind)
	s.ErrorIs(err, cause)
	s.Equal("access credential: invalid_grant: token endpoint answered 400 invalid_grant", err.Error())
}

func (s *StoredCredentialsSuite) TestJSONKeepsThePayloadForSealing() {
	raw, err := json.Marshal(s.stored)
	s.Require().NoError(err)
	var back StoredCredentials
	s.Require().NoError(json.Unmarshal(raw, &back))
	s.Equal(s.stored, back)
}
