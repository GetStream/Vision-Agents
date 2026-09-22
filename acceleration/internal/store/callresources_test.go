//go:build integration

package store

func (s *StoreSuite) TestReleasingACallReturnsItsResourceAndIsIdempotent() {
	s.Require().NoError(s.store.RecordCallResource(s.ctx, &CallResource{
		TrunkID: "trunk-1", RouteID: "route-1",
		CallType: "default", CallID: "call-1", CustomerID: "acme",
	}))

	released, err := s.store.ReleaseCallResources(s.ctx, "default", "call-1")
	s.Require().NoError(err)
	s.Require().Len(released, 1)
	s.Equal("trunk-1", released[0].TrunkID)
	s.Equal("route-1", released[0].RouteID)

	again, err := s.store.ReleaseCallResources(s.ctx, "default", "call-1")
	s.Require().NoError(err)
	s.Empty(again, "a retried delivery finds nothing left to release")
}

func (s *StoreSuite) TestReleasingAnUnknownCallReturnsNothing() {
	released, err := s.store.ReleaseCallResources(s.ctx, "default", "nobody-called")
	s.Require().NoError(err)
	s.Empty(released)
}

func (s *StoreSuite) TestATransferAddsASecondTrunkToTheSameCall() {
	s.Require().NoError(s.store.RecordCallResource(s.ctx, &CallResource{
		TrunkID: "call-trunk", RouteID: "call-route",
		CallType: "default", CallID: "call-1", CustomerID: "acme",
	}))
	s.Require().NoError(s.store.RecordCallResource(s.ctx, &CallResource{
		TrunkID: "transfer-trunk", RouteID: "transfer-route",
		CallType: "default", CallID: "call-1", CustomerID: "acme",
	}))

	released, err := s.store.ReleaseCallResources(s.ctx, "default", "call-1")
	s.Require().NoError(err)
	s.Require().Len(released, 2, "both the call's own trunk and the transfer's trunk are released")

	trunkIDs := []string{released[0].TrunkID, released[1].TrunkID}
	s.Contains(trunkIDs, "call-trunk")
	s.Contains(trunkIDs, "transfer-trunk")
}

func (s *StoreSuite) TestRecordCallResourceRequiresATrunk() {
	err := s.store.RecordCallResource(s.ctx, &CallResource{
		RouteID: "route-1", CallType: "default", CallID: "call-1", CustomerID: "acme",
	})
	s.ErrorContains(err, "trunk")
}

func (s *StoreSuite) TestRecordCallResourceRequiresARoute() {
	err := s.store.RecordCallResource(s.ctx, &CallResource{
		TrunkID: "trunk-1", CallType: "default", CallID: "call-1", CustomerID: "acme",
	})
	s.ErrorContains(err, "route")
}

func (s *StoreSuite) TestRecordCallResourceRequiresACall() {
	err := s.store.RecordCallResource(s.ctx, &CallResource{
		TrunkID: "trunk-1", RouteID: "route-1", CallType: "default", CustomerID: "acme",
	})
	s.ErrorContains(err, "call")
}
