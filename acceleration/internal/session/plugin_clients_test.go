package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type PluginLoginSealSuite struct {
	suite.Suite
	secrets *auth.Sealer
}

func TestPluginLoginSealSuite(t *testing.T) {
	suite.Run(t, new(PluginLoginSealSuite))
}

func (s *PluginLoginSealSuite) SetupTest() {
	secrets, err := auth.NewSealer("the suite's key encryption key")
	s.Require().NoError(err)
	s.secrets = secrets
}

func (s *PluginLoginSealSuite) TestASealedLoginOpensToItsTokens() {
	login := s.alices()
	s.Require().NoError(SealPluginLogin(s.secrets, &login))
	stored := store.PluginConnection{CustomerID: login.CustomerID, ConfigID: login.ConfigID, UserID: login.UserID,
		PluginID: login.PluginID, AccessTokenSealed: login.AccessTokenSealed,
		RefreshTokenSealed: login.RefreshTokenSealed, KEKVersion: login.KEKVersion}

	s.Require().NoError(OpenPluginLogin(s.secrets, &stored))

	s.Equal("alices-access", stored.AccessToken)
	s.Equal("alices-refresh", stored.RefreshToken)
}

// A token sealed for one login is worthless on any other row, so a database writer who can
// copy ciphertext between rows cannot hand one user's login to another.
func (s *PluginLoginSealSuite) TestATokenCopiedOntoAnotherLoginDoesNotOpen() {
	alices := s.alices()
	s.Require().NoError(SealPluginLogin(s.secrets, &alices))
	bobs := store.PluginConnection{CustomerID: alices.CustomerID, ConfigID: alices.ConfigID, UserID: "bob",
		PluginID: alices.PluginID, AccessTokenSealed: alices.AccessTokenSealed, KEKVersion: alices.KEKVersion}

	s.Error(OpenPluginLogin(s.secrets, &bobs))
	s.Empty(bobs.AccessToken)
}

func (s *PluginLoginSealSuite) TestTheTwoTokensOfALoginCannotBeSwapped() {
	login := s.alices()
	s.Require().NoError(SealPluginLogin(s.secrets, &login))
	login.AccessTokenSealed, login.RefreshTokenSealed = login.RefreshTokenSealed, login.AccessTokenSealed

	s.Error(OpenPluginLogin(s.secrets, &login))
}

// A login stored before tokens were sealed is still used, until it is written again.
func (s *PluginLoginSealSuite) TestALoginStoredInTheClearIsReadAsItIs() {
	login := s.alices()

	s.Require().NoError(OpenPluginLogin(nil, &login))

	s.Equal("alices-access", login.AccessToken)
}

func (s *PluginLoginSealSuite) TestATokenIsNotStoredWithoutAKey() {
	login := s.alices()

	s.ErrorIs(SealPluginLogin(nil, &login), ErrNoLoginKey)
}

func (s *PluginLoginSealSuite) alices() store.PluginConnection {
	return store.PluginConnection{CustomerID: "acme", ConfigID: "support", UserID: "alice", PluginID: "google_calendar",
		AccessToken: "alices-access", RefreshToken: "alices-refresh"}
}
