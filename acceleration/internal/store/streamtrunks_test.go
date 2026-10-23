//go:build integration

package store

import (
	"context"
	"sync"
	"time"
)

// trunkIDs are the ids of the trunks one group sees.
func (s *StoreSuite) trunkIDs(group TrunkGroup) []string {
	var ids []string
	s.Require().NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *TrunkGroupTx) error {
		trunks, err := tx.Trunks()
		for _, trunk := range trunks {
			ids = append(ids, trunk.ID)
		}
		return err
	}))
	return ids
}

// savedTrunk stores a trunk in a group, defaulting the fields a test does not care about.
func (s *StoreSuite) savedTrunk(group TrunkGroup, id string) {
	s.Require().NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *TrunkGroupTx) error {
		return tx.SaveTrunk(StreamTrunk{
			ID: id, Numbers: []string{"+15125550000"}, BridgeURI: "sip:" + id + "@sip.example.test",
			BridgeUsername: id, BridgePasswordSealed: []byte("sealed"), BridgePasswordKEKVersion: 1,
		})
	}))
}

func (s *StoreSuite) TestATrunkGroupSeesOnlyItsOwnTrunks() {
	group := TrunkGroup{CustomerID: "acme", StreamAppPK: 7, Vendor: "twilio"}
	s.savedTrunk(group, "trunk-own")
	s.savedTrunk(TrunkGroup{CustomerID: "globex", StreamAppPK: 7, Vendor: "twilio"}, "trunk-other-customer")
	s.savedTrunk(TrunkGroup{CustomerID: "acme", StreamAppPK: 7, Vendor: "telnyx"}, "trunk-other-vendor")
	s.savedTrunk(TrunkGroup{CustomerID: "acme", StreamAppPK: 8, Vendor: "twilio"}, "trunk-other-app")

	s.Equal([]string{"trunk-own"}, s.trunkIDs(group))
}

func (s *StoreSuite) TestTheDeploymentsAppIsAGroupOfItsOwn() {
	s.savedTrunk(TrunkGroup{CustomerID: "acme", Vendor: "twilio"}, "trunk-deployment")
	s.savedTrunk(TrunkGroup{CustomerID: "acme", StreamAppPK: 7, Vendor: "twilio"}, "trunk-own-app")

	s.Equal([]string{"trunk-deployment"}, s.trunkIDs(TrunkGroup{CustomerID: "acme", Vendor: "twilio"}))
}

func (s *StoreSuite) TestTwoTransactionsOnOneGroupRunOneAfterTheOther() {
	group := TrunkGroup{CustomerID: "acme", StreamAppPK: 7, Vendor: "twilio"}
	firstHolds, firstMayFinish := make(chan struct{}), make(chan struct{})
	var seen []string
	var wait sync.WaitGroup
	wait.Add(2)

	go func() {
		defer wait.Done()
		s.NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *TrunkGroupTx) error {
			close(firstHolds)
			<-firstMayFinish
			return tx.SaveTrunk(StreamTrunk{ID: "trunk-first", Numbers: []string{"+15125550000"},
				BridgeURI: "sip:x@y", BridgeUsername: "x", BridgePasswordSealed: []byte("sealed"), BridgePasswordKEKVersion: 1})
		}))
	}()
	<-firstHolds
	go func() {
		defer wait.Done()
		s.NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *TrunkGroupTx) error {
			trunks, err := tx.Trunks()
			for _, trunk := range trunks {
				seen = append(seen, trunk.ID)
			}
			return err
		}))
	}()
	// Without the lock the second would read before the first commits and see nothing.
	time.Sleep(100 * time.Millisecond)
	close(firstMayFinish)
	wait.Wait()

	s.Equal([]string{"trunk-first"}, seen, "the second waited for the first and saw its trunk")
}

func (s *StoreSuite) TestSettingALineIsReadBackOnTheNumber() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US", CustomerID: "acme", PurchasedAt: s.base,
	}))
	group := TrunkGroup{CustomerID: "acme", StreamAppPK: 7, Vendor: "twilio"}

	s.Require().NoError(s.store.InTrunkGroup(s.ctx, group, func(_ context.Context, tx *TrunkGroupTx) error {
		return tx.SetLine("+15125551234", "trunk-1", "route-1", NumberBridge{
			URI: "sip:+15125551234@sip.example.test", Username: "trunk-1", PasswordSealed: []byte("sealed"), KEKVersion: 2,
		})
	}))

	held, err := s.store.Number(s.ctx, "acme", "+15125551234")
	s.Require().NoError(err)
	s.Equal("trunk-1", held.StreamTrunkID)
	s.Equal("route-1", held.StreamRouteID)
	s.Equal(int64(7), held.StreamAppPK)
	s.Equal("sip:+15125551234@sip.example.test", held.BridgeURI)
	s.Equal("trunk-1", held.BridgeUsername)
	s.Equal([]byte("sealed"), held.BridgePasswordSealed)
	s.Equal(2, held.BridgePasswordKEKVersion)
	s.Nil(held.AttachedAt, "a line is not an attachment")
}

func (s *StoreSuite) TestReleasingANumberClearsItsAttachedAt() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US", CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(s.ctx, "acme", "+15125551234", NumberAttachment{TrunkID: "trunk-1", CallType: "default"}))
	attached, err := s.store.Number(s.ctx, "acme", "+15125551234")
	s.Require().NoError(err)
	s.Require().NotNil(attached.AttachedAt)

	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base))

	numbers, err := s.store.CustomerNumbers(s.ctx, "acme", true)
	s.Require().NoError(err)
	s.Require().Len(numbers, 1)
	s.Nil(numbers[0].AttachedAt)
}

func (s *StoreSuite) TestTwoCallResourcesMayShareATrunk() {
	for _, call := range []string{"call-1", "call-2"} {
		s.Require().NoError(s.store.RecordCallResource(s.ctx, &CallResource{
			TrunkID: "shared-trunk", RouteID: "route-1", CallType: "default", CallID: call, CustomerID: "acme", Shared: true,
		}))
	}

	s.Require().NoError(s.store.DeleteCallResource(s.ctx, "default", "call-1", "shared-trunk"))

	gone, err := s.store.CallResourcesInApp(s.ctx, AppScope{Unpinned: true}, "default", "call-1")
	s.Require().NoError(err)
	s.Empty(gone)
	kept, err := s.store.CallResourcesInApp(s.ctx, AppScope{Unpinned: true}, "default", "call-2")
	s.Require().NoError(err)
	s.Require().Len(kept, 1)
	s.True(kept[0].Shared)
}
