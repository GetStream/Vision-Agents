//go:build integration

package phone

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type SIPTrunkSuite struct {
	suite.Suite
	ctx     context.Context
	store   *store.Store
	trunks  *trunkProvider
	service *Service
}

func TestSIPTrunkSuite(t *testing.T) { suite.Run(t, new(SIPTrunkSuite)) }

func (s *SIPTrunkSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "phone_sip_trunks"))
	s.Require().NoError(err)
	s.store = opened

	var database string
	s.Require().NoError(opened.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s", database)
	_, err = opened.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
}

func (s *SIPTrunkSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *SIPTrunkSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE phone_numbers, sip_trunks CASCADE")
	s.Require().NoError(err)
	sealer, err := auth.NewSealer("sip trunk suite")
	s.Require().NoError(err)
	config, err := DefaultConfig()
	s.Require().NoError(err)
	s.trunks = &trunkProvider{}
	s.service, err = NewService(ServiceOptions{
		Registry: NewRegistry(config), Store: s.store, Sealer: sealer, SIPTrunks: s.trunks,
	})
	s.Require().NoError(err)
}

// placed opens the outbound leg customerID placed through its own trunk, as vendorCallID.
func (s *SIPTrunkSuite) placed(customerID, vendorCallID string) {
	_, err := s.store.OpenCallLeg(s.ctx, &store.CallLeg{
		CustomerID: customerID, CallType: "default", CallID: "session-" + vendorCallID,
		Vendor: SIPTrunkVendor, VendorCallID: vendorCallID, Direction: store.Outbound,
		LocalNumber: "+15125550100", RemoteNumber: "+15125550101",
	})
	s.Require().NoError(err)
}

func (s *SIPTrunkSuite) TestDigitsOnACallThroughATrunkReachTheTrunkProvider() {
	s.placed("acme", "call-1")

	err := s.service.SendDigits(s.ctx, "acme", SIPTrunkVendor, "call-1", "12")
	s.EqualError(err, "phone: digits cannot be pressed on a call through a customer's own sip trunk")
	s.Equal([]string{"call-1:12"}, s.trunks.digits)
}

func (s *SIPTrunkSuite) TestALineCannotPressDigitsOnACallAnotherCustomerPlaced() {
	s.placed("acme", "call-2")
	line := s.service.Line(LineOptions{
		Owner: routing.Owner{CustomerID: "globex"}, Vendor: SIPTrunkVendor, VendorCallID: "call-2",
	})

	s.ErrorIs(line.SendDigits(s.ctx, "12"), ErrCallNotFound)
	s.Empty(s.trunks.digits)
}

func (s *SIPTrunkSuite) create(customerID, password string) store.SIPTrunk {
	trunk, err := s.service.CreateSIPTrunk(s.ctx, customerID, SIPTrunkSettings{
		Name: "main", Host: "trunk.example.com", Username: "agent", Password: &password, LateOffer: true,
	})
	s.Require().NoError(err)
	return trunk
}

func (s *SIPTrunkSuite) number(customerID, trunkID, e164 string) store.PhoneNumber {
	number, err := s.service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner: routing.Owner{CustomerID: customerID}, TrunkID: trunkID, E164: e164, Country: "us",
	})
	s.Require().NoError(err)
	return number
}

func (s *SIPTrunkSuite) TestAPasswordIsStoredSealedAndOpensForADial() {
	trunk := s.create("acme", "s3cret")
	s.NotContains(string(trunk.PasswordSealed), "s3cret")
	s.Equal(1, trunk.PasswordKEKVersion)

	held := s.number("acme", trunk.ID, "+15550000201")
	provider, opened, err := s.service.dialVendor(s.ctx, held)
	s.Require().NoError(err)
	s.Same(s.trunks, provider)
	s.Equal(&SIPTrunk{
		Host: "trunk.example.com", Port: 5060, Transport: "tcp", Username: "agent",
		Password: "s3cret", LateOffer: true, Codecs: []string{"PCMU", "PCMA"},
	}, opened)
}

func (s *SIPTrunkSuite) TestUpdatingWithoutAPasswordKeepsTheStoredOne() {
	trunk := s.create("acme", "s3cret")
	settings := SettingsOf(trunk)
	settings.Name = "renamed"

	updated, err := s.service.UpdateSIPTrunk(s.ctx, "acme", trunk.ID, settings)
	s.Require().NoError(err)
	s.Equal("renamed", updated.Name)
	s.Equal(trunk.PasswordSealed, updated.PasswordSealed)

	held := s.number("acme", trunk.ID, "+15550000202")
	_, opened, err := s.service.dialVendor(s.ctx, held)
	s.Require().NoError(err)
	s.Equal("s3cret", opened.Password)
}

func (s *SIPTrunkSuite) TestANewPasswordReplacesTheOldOne() {
	trunk := s.create("acme", "old")
	settings := SettingsOf(trunk)
	fresh := "new"
	settings.Password = &fresh

	_, err := s.service.UpdateSIPTrunk(s.ctx, "acme", trunk.ID, settings)
	s.Require().NoError(err)

	held := s.number("acme", trunk.ID, "+15550000203")
	_, opened, err := s.service.dialVendor(s.ctx, held)
	s.Require().NoError(err)
	s.Equal("new", opened.Password)
}

func (s *SIPTrunkSuite) TestADialRefusesATrunkWithoutAPassword() {
	trunk := s.create("acme", "s3cret")
	held := s.number("acme", trunk.ID, "+15550000204")
	// What a data move leaves behind.
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE sip_trunks SET password_sealed = ''::bytea, password_kek_version = 0 WHERE id = ?", trunk.ID)
	s.Require().NoError(err)

	_, _, err = s.service.dialVendor(s.ctx, held)
	s.EqualError(err, "phone: sip trunk "+trunk.ID+" has no password; set one before calling through it")
	s.ErrorIs(err, ErrInvalid)
}

func (s *SIPTrunkSuite) TestADialRefusesAPasswordSealedAtNoKey() {
	trunk := s.create("acme", "s3cret")
	held := s.number("acme", trunk.ID, "+15550000211")
	// A password set again and then overwritten by a later data move keeps its sealed
	// bytes, but at version 0 no key opens them.
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE sip_trunks SET password_kek_version = 0 WHERE id = ?", trunk.ID)
	s.Require().NoError(err)

	_, _, err = s.service.dialVendor(s.ctx, held)
	s.EqualError(err, "phone: sip trunk "+trunk.ID+" has no password; set one before calling through it")
}

func (s *SIPTrunkSuite) TestAnotherCustomersTrunkCannotTakeANumber() {
	trunk := s.create("acme", "s3cret")

	_, err := s.service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner: routing.Owner{CustomerID: "other"}, TrunkID: trunk.ID, E164: "+15550000205", Country: "US",
	})
	s.ErrorIs(err, store.ErrNoSIPTrunk)
}

func (s *SIPTrunkSuite) TestANumberIsCheckedBeforeItIsAdded() {
	trunk := s.create("acme", "s3cret")

	_, err := s.service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner: routing.Owner{CustomerID: "acme"}, TrunkID: trunk.ID, E164: "5550000206", Country: "USA",
	})
	s.ErrorIs(err, ErrInvalidSIPTrunk)
	s.EqualError(err, `sip_trunk: invalid: e164 "5550000206" is not an E.164 number, e.g. +15551234567; `+
		`country "USA" must be a two-letter code, e.g. US`)
}

func (s *SIPTrunkSuite) TestABadTagIsRefusedWithTheNumber() {
	trunk := s.create("acme", "s3cret")

	_, err := s.service.AddTrunkNumber(s.ctx, TrunkNumber{
		Owner:   routing.Owner{CustomerID: "acme", Tags: routing.Tags{"bad key": "x"}},
		TrunkID: trunk.ID, E164: "+15550000210", Country: "US",
	})
	s.EqualError(err, `sip_trunk: invalid: tag key "bad key" must match ^[a-zA-Z0-9_.-]{1,64}$`)
}

func (s *SIPTrunkSuite) TestReleasingATrunkNumberAsksNoVendor() {
	trunk := s.create("acme", "s3cret")
	s.number("acme", trunk.ID, "+15550000207")

	// trunkProvider does not override ReleaseNumber, so reaching it would panic.
	s.Require().NoError(s.service.Release(s.ctx, "acme", "+15550000207"))
	s.NoError(s.service.DeleteSIPTrunk(s.ctx, "acme", trunk.ID), "a released number no longer holds its trunk")
}

// A deployment without Stream is ErrNoStream, which the router answers 400, not 500.
func (s *SIPTrunkSuite) TestAttachingWithoutStreamSaysSo() {
	number := store.PhoneNumber{E164: "+15550000212", Vendor: "telnyx", CustomerID: "acme"}
	s.Require().NoError(s.store.RecordNumber(s.ctx, &number))

	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: "acme", E164: number.E164})

	s.EqualError(err, "phone: attaching a number needs stream credentials")
	s.ErrorIs(err, ErrNoStream)
}

func (s *SIPTrunkSuite) TestAttachingATrunkNumberIsRefused() {
	trunk := s.create("acme", "s3cret")
	s.number("acme", trunk.ID, "+15550000208")

	_, err := s.service.Attach(s.ctx, Attachment{CustomerID: "acme", E164: "+15550000208"})
	s.EqualError(err, "phone: inbound calls to a number on the customer's own sip trunk are not supported")
}

func (s *SIPTrunkSuite) TestABoughtNumberStillGoesThroughTheRegistry() {
	// bics is declared and not implemented, so it opens without credentials as the stub.
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: "+15550000209", Vendor: "bics", Country: "US", CustomerID: "acme",
	}))
	held, err := s.store.Number(s.ctx, "acme", "+15550000209")
	s.Require().NoError(err)

	provider, trunk, err := s.service.dialVendor(s.ctx, held)
	s.Require().NoError(err)
	s.Equal("bics", provider.Vendor())
	s.Nil(trunk)

	allowed, err := s.service.allowlistFor(SIPTrunkVendor)
	s.Require().NoError(err)
	s.Empty(allowed, "the bridge presents the Stream trunk's password")
}

func (s *SIPTrunkSuite) TestResumingATrunkCallOpensItsTrunkPassword() {
	trunk := s.create("acme", "s3cret")

	s.shareLine("stream-trunk-1", "+15125550100")

	resumed, err := s.service.ResumeTrunkCall(s.ctx, TrunkCallRef{
		CustomerID: "acme", SIPTrunkID: trunk.ID, StreamTrunkID: "stream-trunk-1", From: "+15125550100",
	})

	s.Require().NoError(err)
	s.Equal("s3cret", resumed.Trunk.Password)
}

func (s *SIPTrunkSuite) TestResumingAnotherCustomersTrunkIsRefused() {
	trunk := s.create("acme", "s3cret")

	_, err := s.service.ResumeTrunkCall(s.ctx, TrunkCallRef{CustomerID: "globex", SIPTrunkID: trunk.ID})

	s.ErrorIs(err, store.ErrNoSIPTrunk)
}

func (s *SIPTrunkSuite) TestACallOnItsOwnStreamTrunkCannotBeResumed() {
	trunk := s.create("acme", "s3cret")

	_, err := s.service.ResumeTrunkCall(s.ctx, TrunkCallRef{
		CustomerID: "acme", SIPTrunkID: trunk.ID,
	})

	s.ErrorIs(err, store.ErrNoStreamTrunk)
}

// shareLine saves acme a shared Stream trunk for number, whose bridge password is
// bridge-secret.
func (s *SIPTrunkSuite) shareLine(id, number string) {
	sealed, err := s.service.sealer.SealWithAAD("bridge-secret", trunkAAD("acme", id))
	s.Require().NoError(err)
	s.Require().NoError(s.store.InTrunkGroup(s.ctx, store.TrunkGroup{CustomerID: "acme", Vendor: SIPTrunkVendor},
		func(_ context.Context, tx *store.TrunkGroupTx) error {
			return tx.SaveTrunk(store.StreamTrunk{
				ID: id, Numbers: []string{number}, BridgeURI: "sip:old@shared.example.test",
				BridgeUsername: "bridge-user", BridgePasswordSealed: sealed,
				BridgePasswordKEKVersion: s.service.sealer.CurrentVersion(),
			})
		}))
}

func (s *SIPTrunkSuite) TestResumingACallOnASharedLineOpensItsBridgePassword() {
	trunk := s.create("acme", "s3cret")
	s.shareLine("stream-trunk-1", "+15125550100")

	resumed, err := s.service.ResumeTrunkCall(s.ctx, TrunkCallRef{
		CustomerID: "acme", SIPTrunkID: trunk.ID, StreamTrunkID: "stream-trunk-1", From: "+15125550100",
	})

	s.Require().NoError(err)
	s.Equal(Bridge{URI: "sip:+15125550100@shared.example.test", Username: "bridge-user", Password: "bridge-secret"}, resumed.Bridge)
}

func (s *SIPTrunkSuite) TestAResumedCallReportsHowItEndedWithItsToken() {
	trunk := s.create("acme", "s3cret")
	token, err := s.service.outcomes.Placed(s.ctx, outcome.Call{
		Key: outcome.Key{
			CustomerID: "acme", CallType: "default", CallID: "session-resumed",
			Local: "+15125550100", Remote: "+15125550101",
		},
		Vendor: SIPTrunkVendor, Reports: true,
	})
	s.Require().NoError(err)
	s.shareLine("stream-trunk-1", "+15125550100")
	resumed, err := s.service.ResumeTrunkCall(s.ctx, TrunkCallRef{
		CustomerID: "acme", SIPTrunkID: trunk.ID, StreamTrunkID: "stream-trunk-1", From: "+15125550100",
		CallID: "session-resumed", Token: token,
	})
	s.Require().NoError(err)

	resumed.Ended(CallStatus{Result: CallCompleted, Word: "bye", SIPCode: 200})

	call, err := s.service.outcomes.ByToken(s.ctx, token)
	s.Require().NoError(err)
	s.Equal(outcome.Completed, call.Outcome)
}
