package main

import (
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// ConnectorSealerSuite covers how the router builds the keyring connector credentials are
// sealed under, starting from the hosted deployment's proxy mode with no key at all.
type ConnectorSealerSuite struct {
	suite.Suite
	settings config.Config
}

func TestConnectorSealerSuite(t *testing.T) {
	suite.Run(t, new(ConnectorSealerSuite))
}

func (s *ConnectorSealerSuite) SetupTest() {
	// The keyring is read from every ROUTER_AUTH_KEK_V<n> in the environment, so any the
	// shell running the tests happens to hold are cleared too.
	variables := []string{authKEKEnvVar, authKEKVersionEnvVar}
	for _, variable := range os.Environ() {
		if name, _, _ := strings.Cut(variable, "="); strings.HasPrefix(name, authKEKEnvVar+"_V") {
			variables = append(variables, name)
		}
	}
	for _, variable := range variables {
		s.T().Setenv(variable, "")
		s.Require().NoError(os.Unsetenv(variable))
	}
	s.settings = config.Defaults()
	s.settings.Auth.Mode = string(auth.Proxy)
}

func (s *ConnectorSealerSuite) TestProxyStartsWithoutAKEKWhenConnectorsAreOff() {
	sealer, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	s.Nil(sealer)

	_, err = newAuthenticator(s.settings, nil, slog.New(slog.DiscardHandler))
	s.NoError(err)
}

func (s *ConnectorSealerSuite) TestProxyWithConnectorsOnRefusesToStartWithoutAKeyring() {
	s.settings.Connectors.Enabled = true

	_, err := newConnectorSealer(s.settings)
	s.ErrorContains(err, "connectors.enabled needs a key encryption keyring")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1")
}

func (s *ConnectorSealerSuite) TestProxyWithConnectorsOnBuildsTheSealerFromTheKeyring() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	sealer, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	s.Equal(1, sealer.CurrentVersion())
}

func (s *ConnectorSealerSuite) TestAuthKEKIsVersionOne() {
	s.settings.Connectors.Enabled = true
	s.settings.Auth.KEK = "first-key"
	apiSecrets, err := auth.NewSealer("first-key")
	s.Require().NoError(err)
	sealed, err := apiSecrets.Seal("vas_live_s3cret")
	s.Require().NoError(err)

	sealer, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	opened, err := sealer.OpenWithAADVersion(sealed, nil, 1)
	s.Require().NoError(err)
	s.Equal("vas_live_s3cret", opened)
}

func (s *ConnectorSealerSuite) TestARowSealedUnderVersionOneOpensAfterVersionTwoIsAdded() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	before, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	sealed, err := before.SealWithAAD("connector secret", []byte("connection"))
	s.Require().NoError(err)

	s.T().Setenv("ROUTER_AUTH_KEK_V2", "second-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")
	after, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	s.Equal(2, after.CurrentVersion())
	opened, err := after.OpenWithAADVersion(sealed, []byte("connection"), 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestAVersionWithoutItsKeyIsRefused() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")

	_, err := newConnectorSealer(s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V2")
}

func (s *ConnectorSealerSuite) TestAVersionThatIsNotANumberIsRefused() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "two")

	_, err := newConnectorSealer(s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_VERSION to be a positive integer")
}

func (s *ConnectorSealerSuite) TestAKeyPastedIntoTheVersionIsNotRepeatedInTheError() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "pasted-key-encryption-key")

	_, err := newConnectorSealer(s.settings)
	s.Require().Error(err)
	s.NotContains(err.Error(), "pasted-key-encryption-key")
}

func (s *ConnectorSealerSuite) TestMovingTheVersionBackKeepsNewerKeysLoaded() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V2", "second-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")
	forward, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	sealed, err := forward.SealWithAAD("connector secret", []byte("connection"))
	s.Require().NoError(err)

	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "1")
	back, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	s.Equal(1, back.CurrentVersion())
	opened, err := back.OpenWithAADVersion(sealed, []byte("connection"), 2)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestOnlyThePlainSpellingOfAVersionIsAKey() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V01", "another-key")

	sealer, err := newConnectorSealer(s.settings)
	s.Require().NoError(err)
	sealed, err := sealer.SealWithAAD("connector secret", nil)
	s.Require().NoError(err)
	withFirst, err := auth.NewSealerWithKeyring(1, map[int]string{1: "first-key"})
	s.Require().NoError(err)
	opened, err := withFirst.OpenWithAADVersion(sealed, nil, 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestTwoDifferentVersionOneKeysAreRefused() {
	s.settings.Connectors.Enabled = true
	s.settings.Auth.KEK = "first-key"
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "another-key")

	_, err := newConnectorSealer(s.settings)
	s.ErrorContains(err, "both version 1 and differ")
}
