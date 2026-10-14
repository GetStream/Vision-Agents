//go:build integration

package store

import (
	"time"
)

func (s *StoreSuite) TestALaterValidationReplacesTheLastOne() {
	connection := s.connection("acme-app", userOwned("alice"))
	first := time.Now().UTC().Truncate(time.Microsecond)
	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Status: "failed", Code: "503", Error: "Service Unavailable", CheckedAt: first}))

	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Status: "connected", CheckedAt: first.Add(time.Second)}))

	got, err := s.store.ConnectorConnectionValidations(s.ctx, []string{connection.ID})
	s.Require().NoError(err)
	s.Equal("connected", got[connection.ID].Status)
	s.Empty(got[connection.ID].Code)
	s.Empty(got[connection.ID].Error)
	s.True(first.Add(time.Second).Equal(got[connection.ID].CheckedAt))
}

// TestAValidationThatRanEarlierNeverReplacesALaterOne: of two validates at once, the one that
// ran first may write last; what the later one found stays.
func (s *StoreSuite) TestAValidationThatRanEarlierNeverReplacesALaterOne() {
	connection := s.connection("acme-app", userOwned("alice"))
	later := time.Now().UTC().Truncate(time.Microsecond)
	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Status: "connected", CheckedAt: later}))

	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Status: "failed", Code: "500", CheckedAt: later.Add(-time.Second)}))

	got, err := s.store.ConnectorConnectionValidations(s.ctx, []string{connection.ID})
	s.Require().NoError(err)
	s.Equal("connected", got[connection.ID].Status)
	s.True(later.Equal(got[connection.ID].CheckedAt))
}

// TestAValidationOfOlderCredentialsNeverReplacesOneOfNewerOnes: a validate of credentials
// replaced while it ran writes last, after a validate of the new ones; even though it finished
// later, what the new credentials' validate found stays.
func (s *StoreSuite) TestAValidationOfOlderCredentialsNeverReplacesOneOfNewerOnes() {
	connection := s.connection("acme-app", userOwned("alice"))
	checked := time.Now().UTC().Truncate(time.Microsecond)
	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Revision: 3, Status: "connected", CheckedAt: checked}))

	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Revision: 2, Status: "failed", Code: "503", CheckedAt: checked.Add(time.Second)}))

	got, err := s.store.ConnectorConnectionValidations(s.ctx, []string{connection.ID})
	s.Require().NoError(err)
	s.Equal("connected", got[connection.ID].Status)
	s.Equal(3, got[connection.ID].Revision)
}

// TestAValidationOfNewerCredentialsReplacesOneThatRanLater: the newer credentials' result
// replaces the older ones' whatever the clocks of the two routers say.
func (s *StoreSuite) TestAValidationOfNewerCredentialsReplacesOneThatRanLater() {
	connection := s.connection("acme-app", userOwned("alice"))
	checked := time.Now().UTC().Truncate(time.Microsecond)
	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Revision: 2, Status: "failed", Code: "503", CheckedAt: checked}))

	s.Require().NoError(s.store.PutConnectorConnectionValidation(s.ctx, &ConnectorConnectionValidation{
		ConnectionID: connection.ID, Revision: 3, Status: "connected", CheckedAt: checked.Add(-time.Second)}))

	got, err := s.store.ConnectorConnectionValidations(s.ctx, []string{connection.ID})
	s.Require().NoError(err)
	s.Equal("connected", got[connection.ID].Status)
	s.Equal(3, got[connection.ID].Revision)
}

func (s *StoreSuite) TestAConnectionNeverValidatedHasNoValidation() {
	connection := s.connection("acme-app", userOwned("alice"))

	got, err := s.store.ConnectorConnectionValidations(s.ctx, []string{connection.ID})

	s.Require().NoError(err)
	s.Empty(got)
}
