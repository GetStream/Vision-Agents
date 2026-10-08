package streamapp

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SealedSuite covers how a Stream app key's secret is sealed to the row it is kept on.
type SealedSuite struct {
	suite.Suite
	sealer *auth.Sealer
	app    store.StreamApp
}

func TestSealedSuite(t *testing.T) {
	suite.Run(t, new(SealedSuite))
}

func (s *SealedSuite) SetupTest() {
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "first-key"})
	s.Require().NoError(err)
	s.sealer = sealer
	s.app = store.StreamApp{CustomerID: "4242", StreamAppPK: 4242}
}

func (s *SealedSuite) seal(apiKey, secret string) store.StreamAppKey {
	key, err := SealKey(s.sealer, s.app.CustomerID, s.app.StreamAppPK, apiKey, secret)
	s.Require().NoError(err)
	return key
}

func (s *SealedSuite) TestAStreamAppKeyOpensOnlyOnItsOwnRow() {
	key := s.seal("own-key", "a-long-stream-secret")

	opened, stale, err := OpenKey(s.sealer, s.app, key)

	s.Require().NoError(err)
	s.Equal("a-long-stream-secret", opened.Reveal())
	s.False(stale)
	s.Equal("cret", key.Last4)
	s.Equal(1, key.KEKVersion)
}

func (s *SealedSuite) TestASwappedRowDoesNotOpen() {
	// Copying ciphertext onto another row, whatever part of it differs, opens nothing.
	key := s.seal("own-key", "a-long-stream-secret")
	for name, row := range map[string]struct {
		app store.StreamApp
		key string
	}{
		"another customer": {store.StreamApp{CustomerID: "7", StreamAppPK: 4242}, "own-key"},
		"another app":      {store.StreamApp{CustomerID: "4242", StreamAppPK: 7}, "own-key"},
		"another key":      {s.app, "other-key"},
		"parts run on":     {store.StreamApp{CustomerID: "42", StreamAppPK: 424}, "own-key"},
	} {
		s.Run(name, func() {
			moved := key
			moved.APIKey = row.key
			_, _, err := OpenKey(s.sealer, row.app, moved)
			s.Error(err)
		})
	}
}

func (s *SealedSuite) TestASecretSealedForSomethingElseDoesNotOpenAsAKey() {
	sealed, err := s.sealer.Seal("a-long-stream-secret")
	s.Require().NoError(err)

	_, _, err = OpenKey(s.sealer, s.app, store.StreamAppKey{APIKey: "own-key", Sealed: sealed, KEKVersion: 1})

	s.Error(err)
}

func (s *SealedSuite) TestAKeySealedUnderAnOldVersionSaysSo() {
	key := s.seal("own-key", "a-long-stream-secret")
	newer, err := auth.NewSealerWithKeyring(2, map[int]string{1: "first-key", 2: "second-key"})
	s.Require().NoError(err)

	opened, stale, err := OpenKey(newer, s.app, key)

	s.Require().NoError(err)
	s.Equal("a-long-stream-secret", opened.Reveal())
	s.True(stale)
}

func (s *SealedSuite) TestAShortSecretShowsNoneOfItself() {
	s.Empty(s.seal("own-key", "short").Last4)
}
