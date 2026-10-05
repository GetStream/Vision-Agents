//go:build integration

package store

import (
	"errors"
	"time"
)

// line connects a WhatsApp line for a customer, with sealed credentials standing in for the
// real ones.
func (s *StoreSuite) line(customerID, e164 string) ChannelAccount {
	account := ChannelAccount{
		CustomerID: customerID, Kind: "whatsapp", E164: e164,
		AccountID: "phone-number-id", Token: newID(),
		SecretsSealed: []byte("sealed"), SecretsKEKVersion: 1,
	}
	s.Require().NoError(s.store.SaveChannelAccount(s.ctx, &account))
	return account
}

func (s *StoreSuite) TestAConnectedLineIsFoundByItsNumberAndByItsWebhookToken() {
	customerID := newID()
	account := s.line(customerID, "+1555"+newID()[:7])

	byNumber, err := s.store.ChannelAccount(s.ctx, customerID, "whatsapp", account.E164)
	s.Require().NoError(err)
	byToken, err := s.store.ChannelAccountByToken(s.ctx, account.Token)
	s.Require().NoError(err)

	s.Equal(account.ID, byNumber.ID)
	s.Equal(account.ID, byToken.ID)
}

// Rotating a token must not mean setting the webhook up with the provider all over again.
func (s *StoreSuite) TestReconnectingALineKeepsItsTokenAndReplacesItsCredentials() {
	customerID := newID()
	first := s.line(customerID, "+1555"+newID()[:7])

	again := ChannelAccount{
		CustomerID: customerID, Kind: "whatsapp", E164: first.E164,
		AccountID: "another-phone-number-id", Token: newID(),
		SecretsSealed: []byte("sealed again"), SecretsKEKVersion: 2,
	}
	s.Require().NoError(s.store.SaveChannelAccount(s.ctx, &again))

	held, err := s.store.ChannelAccount(s.ctx, customerID, "whatsapp", first.E164)
	s.Require().NoError(err)
	s.Equal(first.Token, held.Token)
	s.Equal(first.ID, held.ID)
	s.Equal("another-phone-number-id", held.AccountID)
	s.Equal([]byte("sealed again"), held.SecretsSealed)
}

// A line nobody answers on has no use for the credentials.
func (s *StoreSuite) TestDisconnectingALineDropsItsCredentials() {
	customerID := newID()
	account := s.line(customerID, "+1555"+newID()[:7])

	s.Require().NoError(s.store.DeleteChannelAccount(s.ctx, customerID, "whatsapp", account.E164))

	_, err := s.store.ChannelAccountByToken(s.ctx, account.Token)
	s.ErrorIs(err, ErrUnknownChannelAccount)
	s.ErrorIs(s.store.DeleteChannelAccount(s.ctx, customerID, "whatsapp", account.E164),
		ErrUnknownChannelAccount)
}

func (s *StoreSuite) TestAnotherAppsLineIsNotThisOnes() {
	e164 := "+1555" + newID()[:7]
	s.line(newID(), e164)

	_, err := s.store.ChannelAccount(s.ctx, newID(), "whatsapp", e164)

	s.ErrorIs(err, ErrUnknownChannelAccount)
}

// A provider that did not hear back in time sends the same message again.
func (s *StoreSuite) TestAMessageAlreadyTakenIsNotClaimedTwice() {
	account := s.line(newID(), "+1555"+newID()[:7])
	message := "wamid." + newID()

	first, err := s.store.ClaimChannelMessage(s.ctx, account.ID, message)
	s.Require().NoError(err)
	second, err := s.store.ClaimChannelMessage(s.ctx, account.ID, message)
	s.Require().NoError(err)

	s.True(first)
	s.False(second)
}

func (s *StoreSuite) TestTheAgentOnANumberIsTheOneThatNamedIt() {
	customerID := newID()
	e164 := "+1555" + newID()[:7]
	config := &AgentConfig{
		CustomerID: customerID, Name: "channelled-" + newID(),
		Channels: AgentChannels{WhatsApp: &ChannelLine{Number: e164}, Identity: ChannelIdentityPhone},
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))

	found, ok, err := s.store.ConfigOnChannel(s.ctx, customerID, "whatsapp", e164)

	s.Require().NoError(err)
	s.Require().True(ok)
	s.Equal(config.ID, found.ID)
}

func (s *StoreSuite) TestNoAgentIsFoundOnANumberNobodyNamed() {
	_, ok, err := s.store.ConfigOnChannel(s.ctx, newID(), "whatsapp", "+15550000000")

	s.Require().NoError(err)
	s.False(ok)
}

// The lines are kept per channel, so a number that texts and carries WhatsApp is two.
func (s *StoreSuite) TestTheSameNumberOnAnotherChannelIsAnotherLine() {
	customerID := newID()
	e164 := "+1555" + newID()[:7]
	whatsapp := s.line(customerID, e164)
	texting := ChannelAccount{
		CustomerID: customerID, Kind: "sms", E164: e164, Token: newID(),
		SecretsSealed: []byte("sealed"), SecretsKEKVersion: 1,
	}
	s.Require().NoError(s.store.SaveChannelAccount(s.ctx, &texting))

	s.NotEqual(whatsapp.ID, texting.ID)
	s.NotEqual(whatsapp.Token, texting.Token)
}

func (s *StoreSuite) TestWhoANumberIsCanBeRecordedAndItsConversationMovedOn() {
	customerID, configID := newID(), newID()
	address := "+1347" + newID()[:7]
	identity := ChannelIdentity{
		CustomerID: customerID, ConfigID: configID, Kind: "whatsapp",
		Address: address, UserID: "phone:" + address,
	}
	s.Require().NoError(s.store.SaveChannelIdentity(s.ctx, &identity))

	identity.ConversationID = "agent:support-" + newID()
	s.Require().NoError(s.store.SaveChannelIdentity(s.ctx, &identity))

	held, err := s.store.ChannelIdentity(s.ctx, customerID, configID, "whatsapp", address)
	s.Require().NoError(err)
	s.Equal(identity.ConversationID, held.ConversationID)
	s.Equal("phone:"+address, held.UserID)
}

// The same person writing to two agents is two conversations.
func (s *StoreSuite) TestANumberIsIdentifiedPerAgent() {
	customerID, address := newID(), "+1347"+newID()[:7]
	first := ChannelIdentity{
		CustomerID: customerID, ConfigID: newID(), Kind: "whatsapp",
		Address: address, UserID: "someone", ConversationID: "agent:one",
	}
	s.Require().NoError(s.store.SaveChannelIdentity(s.ctx, &first))
	second := first
	second.ID = ""
	second.ConfigID = newID()
	second.ConversationID = "agent:two"
	s.Require().NoError(s.store.SaveChannelIdentity(s.ctx, &second))

	held, err := s.store.ChannelIdentity(s.ctx, customerID, first.ConfigID, "whatsapp", address)
	s.Require().NoError(err)
	s.Equal("agent:one", held.ConversationID)
}

func (s *StoreSuite) TestANumberNobodyHasClaimedBelongsToNobody() {
	_, err := s.store.ChannelIdentity(s.ctx, newID(), newID(), "whatsapp", "+13470000000")

	s.True(errors.Is(err, ErrUnknownChannelIdentity))
}

// A code somebody saw go by must not claim a second number.
func (s *StoreSuite) TestACodeIsSpentOnce() {
	customerID := newID()
	link := ChannelLink{
		Code: newID()[:6], CustomerID: customerID, ConfigID: newID(),
		UserID: "on-call-engineer", ExpiresAt: time.Now().UTC().Add(time.Minute),
	}
	s.Require().NoError(s.store.SaveChannelLink(s.ctx, &link))

	claimed, err := s.store.ClaimChannelLink(s.ctx, customerID, link.Code)
	s.Require().NoError(err)
	s.Equal("on-call-engineer", claimed.UserID)

	_, err = s.store.ClaimChannelLink(s.ctx, customerID, link.Code)
	s.ErrorIs(err, ErrUnknownChannelLink)
}

func (s *StoreSuite) TestACodePastItsTimeIsNoCodeAtAll() {
	customerID := newID()
	link := ChannelLink{
		Code: newID()[:6], CustomerID: customerID, ConfigID: newID(),
		UserID: "on-call-engineer", ExpiresAt: time.Now().UTC().Add(-time.Minute),
	}
	s.Require().NoError(s.store.SaveChannelLink(s.ctx, &link))

	_, err := s.store.ClaimChannelLink(s.ctx, customerID, link.Code)

	s.ErrorIs(err, ErrUnknownChannelLink)
}

func (s *StoreSuite) TestAnotherAppsCodeCannotBeSpentHere() {
	link := ChannelLink{
		Code: newID()[:6], CustomerID: newID(), ConfigID: newID(),
		UserID: "on-call-engineer", ExpiresAt: time.Now().UTC().Add(time.Minute),
	}
	s.Require().NoError(s.store.SaveChannelLink(s.ctx, &link))

	_, err := s.store.ClaimChannelLink(s.ctx, newID(), link.Code)

	s.ErrorIs(err, ErrUnknownChannelLink)
}
