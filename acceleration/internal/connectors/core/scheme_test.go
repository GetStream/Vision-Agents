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

type MaterialSuite struct {
	suite.Suite
	material Material
}

func TestMaterialSuite(t *testing.T) {
	suite.Run(t, new(MaterialSuite))
}

func (s *MaterialSuite) SetupTest() {
	s.material = Material{Scheme: "oauth2_code", Version: 1, Payload: json.RawMessage(`{"refresh_token":"rt-very-secret"}`)}
}

func (s *MaterialSuite) TestNoFormatVerbPrintsThePayload() {
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		out := fmt.Sprintf(verb, s.material)
		s.NotContains(out, "rt-very-secret", verb)
		s.Contains(out, "oauth2_code", verb)
	}
}

func (s *MaterialSuite) TestALogLineLeavesThePayloadOut() {
	var text, structured strings.Builder
	slog.New(slog.NewTextHandler(&text, nil)).Info("rotated", "material", s.material)
	slog.New(slog.NewJSONHandler(&structured, nil)).Info("rotated", "material", s.material)
	s.NotContains(text.String(), "rt-very-secret")
	s.NotContains(structured.String(), "rt-very-secret")
	s.Contains(structured.String(), "oauth2_code")
}

func (s *MaterialSuite) TestJSONKeepsThePayloadForSealing() {
	raw, err := json.Marshal(s.material)
	s.Require().NoError(err)
	var back Material
	s.Require().NoError(json.Unmarshal(raw, &back))
	s.Equal(s.material, back)
}
