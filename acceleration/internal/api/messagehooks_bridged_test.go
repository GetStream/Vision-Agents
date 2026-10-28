package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"
)

// MessageBridgedSuite reads the mark the channel bridge sets on a message it wrote
// (channelbridge.BridgedField) from a message.new as Stream delivers it: at the top level of
// the message, as every custom field.
type MessageBridgedSuite struct {
	suite.Suite
}

func TestMessageBridgedSuite(t *testing.T) {
	suite.Run(t, new(MessageBridgedSuite))
}

// sent is a message.new in a thread channel whose message has the given fields besides its own,
// as raw JSON members.
func (s *MessageBridgedSuite) sent(fields string) messageEvent {
	var event messageEvent
	s.Require().NoError(json.Unmarshal([]byte(
		`{"channel_type":"agent","channel_id":"thread-1","message":{"id":"m","text":"hi","user":{"id":"u"}`+fields+`}}`), &event))
	return event
}

func (s *MessageBridgedSuite) TestTrueIsTheBridgesMessage() {
	s.True(s.sent(`,"bridged":true`).bridged())
}

func (s *MessageBridgedSuite) TestAMessageWithoutTheFieldIsSomebodyElses() {
	s.False(s.sent(``).bridged())
}

func (s *MessageBridgedSuite) TestFalseIsSomebodyElses() {
	s.False(s.sent(`,"bridged":false`).bridged())
}

func (s *MessageBridgedSuite) TestNullIsSomebodyElses() {
	s.False(s.sent(`,"bridged":null`).bridged())
}

// Anyone writing in the app may set the field to anything: only JSON true is the bridge's, and
// a value of another type still leaves the event readable.
func (s *MessageBridgedSuite) TestTheStringTrueIsSomebodyElses() {
	s.False(s.sent(`,"bridged":"true"`).bridged())
}

func (s *MessageBridgedSuite) TestANumberIsSomebodyElses() {
	s.False(s.sent(`,"bridged":1`).bridged())
}

func (s *MessageBridgedSuite) TestAnObjectIsSomebodyElses() {
	s.False(s.sent(`,"bridged":{}`).bridged())
}

// The field is the message's, not the event's: one beside the message marks nothing.
func (s *MessageBridgedSuite) TestTheFieldOutsideTheMessageMarksNothing() {
	var event messageEvent
	s.Require().NoError(json.Unmarshal([]byte(
		`{"channel_type":"agent","channel_id":"thread-1","bridged":true,"message":{"id":"m","text":"hi","user":{"id":"u"}}}`), &event))

	s.False(event.bridged())
}
