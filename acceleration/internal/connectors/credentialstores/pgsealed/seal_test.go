package pgsealed

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SealSuite covers sealing and opening StoredCredentials, which needs no database.
type SealSuite struct {
	suite.Suite
	store  *CredentialStore
	ref    core.ConnectionRef
	stored core.StoredCredentials
}

func TestSealSuite(t *testing.T) {
	suite.Run(t, new(SealSuite))
}

func (s *SealSuite) SetupTest() {
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "test key one"})
	s.Require().NoError(err)
	s.store = &CredentialStore{sealer: sealer}
	s.ref = core.ConnectionRef{CustomerID: "acme-app", ConnectionID: "conn-1"}
	s.stored = core.StoredCredentials{Scheme: "test_rotating", Version: 1, Payload: json.RawMessage(`{"refresh_token":"secret-refresh"}`)}
}

// sealed is a connection holding stored sealed for ref at revision.
func (s *SealSuite) sealed(ref core.ConnectionRef, revision int) *store.ConnectorConnection {
	blob, version, err := s.store.seal(ref, revision, s.stored)
	s.Require().NoError(err)
	return &store.ConnectorConnection{Revision: revision, CredentialsSealed: blob, CredentialsKEKVersion: version}
}

func (s *SealSuite) TestSealedCredentialsOpenForTheirRowAndRevision() {
	connection := s.sealed(s.ref, 2)

	opened, ok, err := s.store.open(s.ref, connection)

	s.Require().NoError(err)
	s.True(ok)
	s.Equal(s.stored, opened)
	s.NotContains(string(connection.CredentialsSealed), "secret-refresh")
	s.Equal(1, connection.CredentialsKEKVersion)
}

func (s *SealSuite) TestSealedCredentialsDoNotOpenAtAnotherRevision() {
	connection := s.sealed(s.ref, 2)
	connection.Revision = 3

	_, ok, err := s.store.open(s.ref, connection)
	s.Require().NoError(err, "a blob that does not authenticate is not a missing key")
	s.False(ok)
}

func (s *SealSuite) TestAKeyVersionTheKeyringLacksIsAnErrorNotAnUnreadableBlob() {
	connection := s.sealed(s.ref, 2)
	connection.CredentialsKEKVersion = 2

	_, _, err := s.store.open(s.ref, connection)
	s.ErrorIs(err, auth.ErrKeyVersionUnavailable)
}

func (s *SealSuite) TestNoCredentialsAreAnEmptyBlobAtKeyVersionZero() {
	blob, version, err := s.store.seal(s.ref, 2, core.StoredCredentials{})
	s.Require().NoError(err)
	s.Empty(blob)
	s.Zero(version)

	opened, ok, err := s.store.open(s.ref, &store.ConnectorConnection{Revision: 2, CredentialsSealed: blob})
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(core.StoredCredentials{}, opened)
}

func (s *SealSuite) TestTheAADTellsApartTwoSplitsOfTheSameCharacters() {
	one := credentialsAAD(core.ConnectionRef{CustomerID: "a:1", ConnectionID: "b"}, 1)
	other := credentialsAAD(core.ConnectionRef{CustomerID: "a", ConnectionID: "1:b"}, 1)
	s.NotEqual(string(one), string(other))
}
