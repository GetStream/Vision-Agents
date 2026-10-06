//go:build integration

package store

import "github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"

// withAccount gives a connection the metadata a consent for team and user would leave.
func withAccount(team, user string) func(*ConnectorConnection) {
	return func(connection *ConnectorConnection) {
		connection.Metadata = map[string]string{"team_id": team, "user_id": user}
	}
}

func (s *StoreSuite) TestConnectionsByIdentityFindTheAccountInEveryCustomer() {
	mine := s.connection("acme-app", withAccount("T1", "U1"))
	theirs := s.connection("other-app", withAccount("T1", "U1"))
	s.connection("acme-app", withAccount("T1", "U2"))
	s.connection("acme-app", withAccount("T2", "U1"))

	refs, err := s.store.ConnectionsByIdentity(s.ctx, "acme", nil, map[string]string{"team_id": "T1", "user_id": "U1"})

	s.Require().NoError(err)
	s.ElementsMatch([]core.ConnectionRef{
		{CustomerID: "acme-app", ConnectionID: mine.ID},
		{CustomerID: "other-app", ConnectionID: theirs.ID},
	}, refs)
}

// Fewer parts than the identity, such as a workspace uninstall that names no user, are every
// account that has them.
func (s *StoreSuite) TestConnectionsByIdentityWithFewerPartsFindEveryAccountWithThem() {
	alice := s.connection("acme-app", withAccount("T1", "U1"))
	bob := s.connection("acme-app", withAccount("T1", "U2"))
	s.connection("acme-app", withAccount("T2", "U1"))

	refs, err := s.store.ConnectionsByIdentity(s.ctx, "acme", nil, map[string]string{"team_id": "T1"})

	s.Require().NoError(err)
	s.ElementsMatch([]core.ConnectionRef{
		{CustomerID: "acme-app", ConnectionID: alice.ID},
		{CustomerID: "acme-app", ConnectionID: bob.ID},
	}, refs)
}

func (s *StoreSuite) TestConnectionsByIdentityLeaveOutADeletedConnection() {
	gone := s.connection("acme-app", withAccount("T1", "U1"))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", gone.ID))

	refs, err := s.store.ConnectionsByIdentity(s.ctx, "acme", nil, map[string]string{"team_id": "T1"})

	s.Require().NoError(err)
	s.Empty(refs)
}

func (s *StoreSuite) TestConnectionsByIdentityNeedAPart() {
	_, err := s.store.ConnectionsByIdentity(s.ctx, "acme", nil, nil)

	s.ErrorContains(err, "at least one identity part are required")
}
