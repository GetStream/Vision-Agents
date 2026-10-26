//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/xml"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TelnyxHookSuite is Telnyx asking the router where a call to one of the numbers on its
// TeXML application goes, signed with the account's key as Telnyx signs it.
type TelnyxHookSuite struct {
	RouterSuite
	private ed25519.PrivateKey
	logged  *lockedLog
	e164    string
}

func TestTelnyxHookSuite(t *testing.T) {
	runSuite(t, new(TelnyxHookSuite))
}

func (s *TelnyxHookSuite) SetupSuite() {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.private = private
	s.telnyxPublicKey = public
	s.telnyxConnectionID = "1234567890"
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *TelnyxHookSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.e164 = s.utils.number()
}

func (s *TelnyxHookSuite) TestACallIsSentToTheNumbersOwnTrunk() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-pa$$<word>")

	status, contentType, body := s.deliver(s.call(s.e164), true)

	s.Require().Equal(http.StatusOK, status, body)
	s.Equal("application/xml", contentType)
	var parsed struct {
		Dial struct {
			Sip struct {
				URI      string `xml:",chardata"`
				Username string `xml:"username,attr"`
				Password string `xml:"password,attr"`
			} `xml:"Sip"`
		} `xml:"Dial"`
	}
	s.Require().NoError(xml.Unmarshal([]byte(body), &parsed))
	s.Equal("sip:"+s.e164+"@trunk.sip.example.test", parsed.Dial.Sip.URI)
	s.Equal("trunk-user", parsed.Dial.Sip.Username)
	s.Equal("trunk-pa$$<word>", parsed.Dial.Sip.Password)
	s.NotContains(s.logged.String(), "trunk-pa$$", "the password is never logged")
}

func (s *TelnyxHookSuite) TestAnUnsignedCallIsRefused() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")

	status, _, body := s.deliver(s.call(s.e164), false)

	s.Equal(http.StatusUnauthorized, status)
	s.NotContains(body, "trunk-password")
	s.NotContains(body, "signed")
}

func (s *TelnyxHookSuite) TestACallToANumberNobodyHoldsIsHungUp() {
	number := s.utils.number()
	status, contentType, body := s.deliver(s.call(number), true)

	s.Equal(http.StatusOK, status)
	s.Equal("application/xml", contentType)
	s.Contains(body, "<Response><Hangup/></Response>")
	s.True(s.loggedLine(number, "unknown, released or not attached"))
	s.False(s.loggedLine(number, "attach the number again"), "there is nothing to attach again")
}

// Dev and hosted routers share one Telnyx account and so one signing key: a call for the
// other router's TeXML application is signed as well as one for this router's.
func (s *TelnyxHookSuite) TestACallOnAnotherTeXMLApplicationIsHungUp() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	form := s.call(s.e164)
	form.Set("ConnectionId", "9999999999")

	status, _, body := s.deliver(form, true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "<Response><Hangup/></Response>")
	s.NotContains(body, "trunk-password")
	s.True(s.loggedLine("another TeXML application", "9999999999"))
}

// With no application configured, a signed call that names none must not match "".
func (s *TelnyxHookSuite) TestWithoutAConfiguredTeXMLApplicationNoCallIsAnswered() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	configured := s.router.telnyxConnectionID
	s.router.telnyxConnectionID = ""
	defer func() { s.router.telnyxConnectionID = configured }()
	form := s.call(s.e164)
	form.Del("ConnectionId")

	status, _, body := s.deliver(form, true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "<Response><Hangup/></Response>")
	s.NotContains(body, "trunk-password")
	s.True(s.loggedLine("TELNYX_CONNECTION_ID is not set"))
}

func (s *TelnyxHookSuite) TestWithoutAKeyringACallSaysWhatToSet() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	sealed := s.router.phone
	s.router.phone = s.telephonySealedBy(nil, slog.New(slog.NewTextHandler(s.logged, nil)))
	defer func() { s.router.phone = sealed }()

	status, _, body := s.deliver(s.call(s.e164), true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "<Response><Hangup/></Response>")
	s.True(s.loggedLine(s.e164, "no key to open trunk passwords", "ROUTER_AUTH_KEK"))
	s.False(s.loggedLine(s.e164, "attached before its trunk was kept"), "the key is what is missing")
}

func (s *TelnyxHookSuite) TestANumberAttachedBeforeItsTrunkWasKeptIsHungUpAndSaysWhatToDo() {
	s.attach(s.e164, "", "", "")

	status, _, body := s.deliver(s.call(s.e164), true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "<Hangup/>")
	s.True(s.loggedLine(s.e164, "attach the number again"), "the log says what to do about this number")
}

// loggedLine reports whether one line of the log holds every part.
func (s *TelnyxHookSuite) loggedLine(parts ...string) bool {
	for _, line := range strings.Split(s.logged.String(), "\n") {
		found := true
		for _, part := range parts {
			found = found && strings.Contains(line, part)
		}
		if found {
			return true
		}
	}
	return false
}

func (s *TelnyxHookSuite) TestACallThatIsNotInboundIsHungUp() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	form := s.call(s.e164)
	form.Set("Direction", "outbound-api")

	status, _, body := s.deliver(form, true)

	s.Equal(http.StatusOK, status)
	s.Contains(body, "<Hangup/>")
	s.NotContains(body, "trunk-password")
}

func (s *TelnyxHookSuite) TestTheCallersNumberIsNotLogged() {
	s.deliver(s.call(s.utils.number()), true)

	s.NotContains(s.logged.String(), "+15550001111")
}

// A hook is not a customer. Credentials and a claimed tenant on a signed call are not
// read, so the call still reaches the number's own trunk and nobody is recorded under
// the app or joined to its organization.
func (s *TelnyxHookSuite) TestASignedCallCarryingCredentialsRecordsNoUserAndJoinsNoPolicy() {
	s.attach(s.e164, "sip:"+s.e164+"@trunk.sip.example.test", "trunk-user", "trunk-password")
	userID := "user-" + s.utils.uuid()
	claimed := "claimed-" + s.utils.uuid()

	body := s.call(s.e164).Encode()
	request, err := http.NewRequest(http.MethodPost, s.server.URL+telnyxCallHookPath, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	signature := ed25519.Sign(s.private, []byte(timestamp+"|"+body))
	request.Header.Set("Telnyx-Timestamp", timestamp)
	request.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	request.Header.Set(auth.UserHeader, userID)
	request.Header.Set(auth.CustomerHeader, claimed)

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answered, err := io.ReadAll(response.Body)
	s.Require().NoError(err)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Contains(string(answered), "sip:"+s.e164+"@trunk.sip.example.test")
	s.Contains(string(answered), "trunk-user")
	s.NotContains(string(answered), claimed)

	ctx := context.Background()
	recorded, err := s.store.DB().NewSelect().Model((*store.User)(nil)).
		Where("customer_id = ?", s.customerID()).
		Where("id = ?", userID).
		Count(ctx)
	s.Require().NoError(err)
	s.Zero(recorded)

	s.Never(func() bool {
		under, err := s.store.OrganizationOf(ctx, s.customerID())
		return err == nil && under != ""
	}, time.Second, 20*time.Millisecond)
	under, err := s.store.OrganizationOf(ctx, claimed)
	s.Require().NoError(err)
	s.Empty(under)
}

func (s *TelnyxHookSuite) call(to string) url.Values {
	return telnyxRinger{&s.RouterSuite, s.private}.call(to)
}

func (s *TelnyxHookSuite) deliver(form url.Values, signed bool) (int, string, string) {
	return telnyxRinger{&s.RouterSuite, s.private}.deliver(form, signed)
}

func (s *TelnyxHookSuite) attach(e164, uri, username, password string) {
	telnyxRinger{&s.RouterSuite, s.private}.attach(e164, uri, username, password)
}

// telnyxRinger is Telnyx calling the suite's router about a number on its TeXML application,
// signing what it posts with private.
type telnyxRinger struct {
	suite   *RouterSuite
	private ed25519.PrivateKey
}

// call is the form Telnyx posts when someone rings a number, as seen live on 2026-10-13.
func (l telnyxRinger) call(to string) url.Values {
	return url.Values{
		"AccountSid":       {"00000000-0000-4000-8000-000000000001"},
		"CallSid":          {"v3:call-" + l.suite.utils.uuid()},
		"CallSessionId":    {l.suite.utils.uuid()},
		"CallStatus":       {"in-progress"},
		"CallerId":         {"+15550001111"},
		"CallingPartyType": {"pstn"},
		"ConnectionId":     {"1234567890"},
		"Direction":        {"inbound"},
		"From":             {"+15550001111"},
		"To":               {to},
	}
}

// deliver posts the form as Telnyx does, signed or not, and answers with what came back.
func (l telnyxRinger) deliver(form url.Values, signed bool) (int, string, string) {
	body := form.Encode()
	request, err := http.NewRequest(http.MethodPost, l.suite.server.URL+"/v1/phone/hooks/telnyx", strings.NewReader(body))
	l.suite.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if signed {
		timestamp := strconv.FormatInt(time.Now().Unix(), 10)
		signature := ed25519.Sign(l.private, []byte(timestamp+"|"+body))
		request.Header.Set("Telnyx-Timestamp", timestamp)
		request.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	}
	response, err := http.DefaultClient.Do(request)
	l.suite.Require().NoError(err)
	defer response.Body.Close()
	answered, err := io.ReadAll(response.Body)
	l.suite.Require().NoError(err)
	return response.StatusCode, response.Header.Get("Content-Type"), string(answered)
}

// attach holds the number at Telnyx for the suite's app and attaches it as the phone
// service does, its trunk password sealed under the service's key. An empty uri is a
// number attached before trunks were kept.
func (l telnyxRinger) attach(e164, uri, username, password string) {
	s := l.suite
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: e164, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	attachment := store.NumberAttachment{TrunkID: "trunk-" + s.utils.uuid(), CallType: "default"}
	if uri != "" {
		sealer, err := auth.NewSealer(suiteTrunkKEK)
		s.Require().NoError(err)
		sealed, err := sealer.SealWithAAD(password, []byte("phone_numbers/"+s.customerID()+"/"+e164))
		s.Require().NoError(err)
		attachment.BridgeURI, attachment.BridgeUsername = uri, username
		attachment.BridgePasswordSealed, attachment.BridgePasswordKEKVersion = sealed, sealer.CurrentVersion()
	}
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), e164, attachment))
}

// TelnyxHookUnconfiguredSuite is a deployment with no Telnyx public key.
type TelnyxHookUnconfiguredSuite struct {
	RouterSuite
}

func TestTelnyxHookUnconfiguredSuite(t *testing.T) {
	runSuite(t, new(TelnyxHookUnconfiguredSuite))
}

func (s *TelnyxHookUnconfiguredSuite) TestWithoutAKeyNoCallIsAnswered() {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+"/v1/phone/hooks/telnyx",
		strings.NewReader("Direction=inbound&To=%2B15125550100"))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)

	s.Equal(http.StatusNotFound, response.StatusCode)
	s.Contains(string(body), "not configured")
}
