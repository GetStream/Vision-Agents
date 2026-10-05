package pgsealed

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SealSuite covers sealing and opening Material, which needs no database.
type SealSuite struct {
	suite.Suite
	backend *Backend
	ref     core.ConnectionRef
	m       core.Material
}

func TestSealSuite(t *testing.T) {
	suite.Run(t, new(SealSuite))
}

func (s *SealSuite) SetupTest() {
	sealer, err := auth.NewSealerWithKeyring(1, map[int]string{1: "test key one"})
	s.Require().NoError(err)
	s.backend = &Backend{sealer: sealer}
	s.ref = core.ConnectionRef{CustomerID: "acme-app", ConnectionID: "conn-1"}
	s.m = core.Material{Scheme: "test_rotating", Version: 1, Payload: json.RawMessage(`{"refresh_token":"secret-refresh"}`)}
}

// sealed is a connection holding m sealed for ref at revision.
func (s *SealSuite) sealed(ref core.ConnectionRef, revision int) *store.ConnectorConnection {
	blob, version, err := s.backend.seal(ref, revision, s.m)
	s.Require().NoError(err)
	return &store.ConnectorConnection{Revision: revision, MaterialSealed: blob, MaterialKEKVersion: version}
}

func (s *SealSuite) TestSealedMaterialOpensForItsRowAndRevision() {
	connection := s.sealed(s.ref, 2)

	opened, ok, err := s.backend.open(s.ref, connection)

	s.Require().NoError(err)
	s.True(ok)
	s.Equal(s.m, opened)
	s.NotContains(string(connection.MaterialSealed), "secret-refresh")
	s.Equal(1, connection.MaterialKEKVersion)
}

func (s *SealSuite) TestSealedMaterialDoesNotOpenAtAnotherRevision() {
	connection := s.sealed(s.ref, 2)
	connection.Revision = 3

	_, ok, err := s.backend.open(s.ref, connection)
	s.Require().NoError(err, "a blob that does not authenticate is not a missing key")
	s.False(ok)
}

func (s *SealSuite) TestAKeyVersionTheKeyringLacksIsAnErrorNotAnUnreadableBlob() {
	connection := s.sealed(s.ref, 2)
	connection.MaterialKEKVersion = 2

	_, _, err := s.backend.open(s.ref, connection)
	s.ErrorIs(err, auth.ErrKeyVersionUnavailable)
}

func (s *SealSuite) TestNoMaterialIsAnEmptyBlobAtKeyVersionZero() {
	blob, version, err := s.backend.seal(s.ref, 2, core.Material{})
	s.Require().NoError(err)
	s.Empty(blob)
	s.Zero(version)

	opened, ok, err := s.backend.open(s.ref, &store.ConnectorConnection{Revision: 2, MaterialSealed: blob})
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(core.Material{}, opened)
}

func (s *SealSuite) TestTheAADTellsApartTwoSplitsOfTheSameCharacters() {
	one := materialAAD(core.ConnectionRef{CustomerID: "a:1", ConnectionID: "b"}, 1)
	other := materialAAD(core.ConnectionRef{CustomerID: "a", ConnectionID: "1:b"}, 1)
	s.NotEqual(string(one), string(other))
}
