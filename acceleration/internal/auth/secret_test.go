package auth

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

// KeyringSuite covers a sealer holding more than one key encryption key, which is how a
// key is rotated without reissuing what it sealed.
type KeyringSuite struct {
	suite.Suite
	record []byte
}

func TestKeyringSuite(t *testing.T) {
	suite.Run(t, new(KeyringSuite))
}

func (s *KeyringSuite) SetupTest() {
	s.record = []byte("tenant/connection/revision")
}

func (s *KeyringSuite) TestARowSealedUnderVersionOneOpensAfterVersionTwoIsAdded() {
	before, err := NewSealer("old-key")
	s.Require().NoError(err)
	sealed, err := before.SealWithAAD("connector secret", s.record)
	s.Require().NoError(err)

	rotated, err := NewSealerWithKeyring(2, map[int]string{1: "old-key", 2: "new-key"})
	s.Require().NoError(err)
	opened, err := rotated.OpenWithAADVersion(sealed, s.record, 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *KeyringSuite) TestNewRowsAreSealedUnderTheCurrentVersion() {
	rotated, err := NewSealerWithKeyring(2, map[int]string{1: "old-key", 2: "new-key"})
	s.Require().NoError(err)
	s.Equal(2, rotated.CurrentVersion())

	sealed, err := rotated.SealWithAAD("rotated secret", s.record)
	s.Require().NoError(err)
	opened, err := rotated.OpenWithAADVersion(sealed, s.record, 2)
	s.Require().NoError(err)
	s.Equal("rotated secret", opened)

	_, err = rotated.OpenWithAADVersion(sealed, s.record, 1)
	s.Error(err)
}

func (s *KeyringSuite) TestAWrongAADFailsToOpen() {
	sealer, err := NewSealer("a passphrase")
	s.Require().NoError(err)
	sealed, err := sealer.SealWithAAD("connector secret", s.record)
	s.Require().NoError(err)

	_, err = sealer.OpenWithAADVersion(sealed, []byte("tenant/another-connection/revision"), 1)
	s.Error(err)
	_, err = sealer.Open(sealed)
	s.Error(err, "a secret bound to a record does not open without it")
}

func (s *KeyringSuite) TestAVersionNotInTheKeyringFailsToOpen() {
	sealer, err := NewSealer("a passphrase")
	s.Require().NoError(err)
	sealed, err := sealer.Seal("vas_live_s3cret")
	s.Require().NoError(err)

	_, err = sealer.OpenWithAADVersion(sealed, nil, 2)
	s.ErrorContains(err, "version 2 is unavailable")
}

func (s *KeyringSuite) TestTheCurrentVersionMustBeInTheKeyring() {
	_, err := NewSealerWithKeyring(2, map[int]string{1: "old-key"})
	s.ErrorContains(err, "version 2 is required")
}

func (s *KeyringSuite) TestAnEmptyKeyIsRefused() {
	_, err := NewSealerWithKeyring(2, map[int]string{1: "", 2: "new-key"})
	s.Error(err)
}
