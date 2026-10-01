package core

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type CredentialSuite struct {
	suite.Suite
	credential Credential
}

func TestCredentialSuite(t *testing.T) {
	suite.Run(t, new(CredentialSuite))
}

func (s *CredentialSuite) SetupTest() {
	s.credential = NewCredential("bearer", time.Unix(1_800_000_000, 0).UTC(), json.RawMessage(`"tok-very-secret"`))
}

func (s *CredentialSuite) TestTheMintingSchemeReadsTheSecretBack() {
	s.JSONEq(`"tok-very-secret"`, string(s.credential.Secret()))
}

func (s *CredentialSuite) TestNoFormatVerbPrintsTheSecret() {
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		s.NotContains(fmt.Sprintf(verb, s.credential), "tok-very-secret", verb)
	}
}

func (s *CredentialSuite) TestJSONLeavesTheSecretOut() {
	raw, err := json.Marshal(s.credential)
	s.Require().NoError(err)
	s.NotContains(string(raw), "tok-very-secret")
}

func (s *CredentialSuite) TestALogLineLeavesTheSecretOut() {
	var text, structured strings.Builder
	slog.New(slog.NewTextHandler(&text, nil)).Info("resolved", "credential", s.credential)
	slog.New(slog.NewJSONHandler(&structured, nil)).Info("resolved", "credential", s.credential)
	s.NotContains(text.String(), "tok-very-secret")
	s.NotContains(structured.String(), "tok-very-secret")
	s.Contains(text.String(), "bearer")
}
