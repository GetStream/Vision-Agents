//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Kanat, 2026-10-10: every way an inbound message ends with no reply is logged at info, with
// a stable message and the ids that find it (connector, provider app, customer, connection,
// config), and never the message itself, so a staging log says what happened. The answers stay
// as they were.

// ids is how a drop line names the test's provider app, customer, connection and config.
func (s *SlackChannelSuite) ids() string {
	return "connector=slack_bot provider_app=" + s.app.ProviderAppID + " customer=" + s.customerID() +
		" connection=" + s.bot.ConnectionID + " config=" + s.config.ID
}

// since is what the router logged after before.
func (s *SlackChannelSuite) since(before int) string {
	return s.logged.String()[before:]
}

func (s *SlackChannelSuite) TestAChannelMessageNotToTheBotIsLoggedAtInfoAsDropped() {
	before := len(s.logged.String())

	status, _ := s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000ALICE","text":"synthetic lunch text",`+
		`"ts":"1759740000.004100","channel_type":"channel"}`), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
	logged := s.since(before)
	s.Contains(logged, `level=INFO msg="dropped an inbound message that is not addressed to the connection on a thread nobody linked" `+
		s.ids()+" waiting=false")
	s.NotContains(logged, "synthetic lunch text")
}

// A reply in a thread nobody linked is kept for the mention that links it (AI-990 F31a): the
// line says it waits.
func (s *SlackChannelSuite) TestAReplyNotToTheBotOnAThreadNobodyLinkedIsLoggedAsWaiting() {
	before := len(s.logged.String())

	status, _ := s.deliver(s.message("U0000ALICE", "synthetic reply text", "1759740000.004300", "1759740000.004200"), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
	logged := s.since(before)
	s.Contains(logged, `level=INFO msg="dropped an inbound message that is not addressed to the connection on a thread nobody linked" `+
		s.ids()+" waiting=true")
	s.NotContains(logged, "synthetic reply text")
}

func (s *SlackChannelSuite) TestARetriedMessageIsLoggedAtInfoAsDropped() {
	event := s.message("U0000ALICE", "synthetic retried text", "1759740000.004400", "")
	s.deliver(event, 0)
	channel := s.threadChannel("C0000CHAN:1759740000.004400")
	s.written(channel, 1)
	before := len(s.logged.String())

	status, _ := s.deliver(event, 1)

	s.Equal(http.StatusOK, status, "a retry is taken, so Slack stops retrying")
	logged := s.since(before)
	s.Contains(logged, `level=INFO msg="dropped a retried inbound message" `+s.ids()+" channel="+channel)
	s.NotContains(logged, "synthetic retried text")
}

// A message the bridge answers logs none of the drop lines: the control.
func (s *SlackChannelSuite) TestAMessageTheBridgeAnswersLogsNoDrop() {
	before := len(s.logged.String())

	s.messaged("U0000ALICE", "synthetic answered text", "1759740000.004500", "")

	logged := s.since(before)
	s.NotContains(logged, "dropped")
	s.NotContains(logged, "carried no message")
}

func (s *SlackChannelSuite) TestAnEventForAProviderAppNobodyHoldsIsLoggedAtInfo() {
	before := len(s.logged.String())
	unknown := "A0000UNKNOWN" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))[:8]

	status, _ := s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/"+unknown, s.secret,
		s.message("U0000ALICE", "synthetic unknown text", "1759740000.004600", ""), 0)

	s.Equal(http.StatusNotFound, status, "as before")
	logged := s.since(before)
	s.Contains(logged, `level=INFO msg="dropped a provider app's event: no customer holds the provider app with a signing secret" `+
		"connector=slack_bot provider_app="+unknown)
	s.NotContains(logged, "synthetic unknown text")
}

// A provider app of a connector whose channel does not read the app's events (github has no
// channel) takes none: 404 as before, and a line that says why.
func (s *SlackChannelSuite) TestAnEventForAProviderAppOfAConnectorWithoutAChannelIsLoggedAtInfo() {
	appID := "synthetic-github-" + s.utils.uuid()
	secret := "synthetic-github-signing-" + s.utils.uuid()
	record := &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "github", Registration: core.ClientCustomer,
		ClientID: "synthetic-client", ProviderAppID: appID,
	}
	var err error
	record.SigningSecretSealed, err = s.sealer.SealWithAAD(secret, providerAppAAD(record.CustomerID, "github", appID))
	s.Require().NoError(err)
	record.SigningKEKVersion = s.sealer.CurrentVersion()
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), "github", core.ClientCustomer))
	})
	before := len(s.logged.String())

	status, _ := s.slack.Deliver(s.server.URL+providerAppEventsPath+"github/"+appID, secret, []byte(`{"synthetic":"github"}`), 0)

	s.Equal(http.StatusNotFound, status, "as before")
	s.Contains(s.since(before), `level=INFO msg="dropped a provider app's event: the connector has no channel verified with the app's own secret" `+
		"connector=github provider_app="+appID+" customer="+s.customerID())
}

// A message the manifest skips is not "no message": its skip line says why, and the
// no-message line stays out.
func (s *SlackChannelSuite) TestASkippedMessageIsNotAlsoLoggedAsCarryingNoMessage() {
	before := len(s.logged.String())

	status, _ := s.deliver(s.event(`{"type":"message","subtype":"channel_join","channel":"C0000CHAN","user":"U0000ALICE",`+
		`"ts":"1759740000.004700"}`), 0)

	s.Equal(http.StatusOK, status)
	s.NotContains(s.since(before), "carried no message")
}

// A delivery report is a verified event the manifest reads nothing from: no message, none
// skipped, no signal. Answered 200 as before, and said at info.
func (s *WhatsAppChannelSuite) TestADeliveryReportIsLoggedAtInfoAsCarryingNoMessage() {
	statuses := []byte(`{"object":"whatsapp_business_account","entry":[{"id":"1","changes":[{"field":"messages","value":{` +
		`"messaging_product":"whatsapp","metadata":{"display_phone_number":"15550783881","phone_number_id":"` + s.unit + `"},` +
		`"statuses":[{"id":"wamid.out","status":"delivered","timestamp":"1749416383","recipient_id":"16505551234"}]}}]}]}`)
	before := len(s.logged.String())

	s.Equal(http.StatusOK, s.deliver(statuses))

	s.nothingLinked()
	logged := s.logged.String()[before:]
	s.Contains(logged, `level=INFO msg="a connector event carried no message" connector=whatsapp provider_app=`+s.app+" customer="+s.customerID())
	s.NotContains(logged, "16505551234")
}

// A text message is read, so it carries one: the control.
func (s *WhatsAppChannelSuite) TestAWhatsAppMessageIsNotLoggedAsCarryingNoMessage() {
	before := len(s.logged.String())

	s.Equal(http.StatusOK, s.deliver(s.received(s.unit, "16505551234", "synthetic whatsapp text")))

	s.NotContains(s.logged.String()[before:], "carried no message")
}
