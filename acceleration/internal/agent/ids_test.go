package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestAnIDNothingGaveTheModelIsNotPassedOn(t *testing.T) {
	tools := []string{"verify_identity", "lookup_appointment", "reschedule_appointment", "create_ticket", "dispatch_tech"}
	history := []llm.Message{
		{Role: llm.User, Content: "Maya Chen, member A B C one two three four five six, gateway error eight forty"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "verify_identity"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":true,"patient_id":"p1"}`},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "lookup_appointment"}}},
		{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"appointments":[{"id":"a1","date":"Thursday"}]}`},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c3", Name: "create_ticket"}}},
		{Role: llm.ToolResult, ToolCallID: "c3", Content: `{"ticket_id":"T-1842"}`},
	}
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "some_tool", Arguments: arguments} }

	key, value, known, unfound := unfoundID(history, call(`{"appointment_id":"p1","new_date":"Tuesday"}`), tools)
	assert.True(t, unfound, "a patient's ID is not an appointment's")
	assert.Equal(t, "appointment_id", key)
	assert.Equal(t, "p1", value)
	assert.Contains(t, known, "a1 (id)")
	assert.Contains(t, unfoundIDResult(key, value, known), "a1 (id)")

	_, _, _, unfound = unfoundID(history, call(`{"appointment_id":"Thursday 2pm with doctor Chen"}`), tools)
	assert.True(t, unfound)
	_, _, _, unfound = unfoundID(history[:5], call(`{"ticket_id":"840"}`), tools)
	assert.True(t, unfound, "a ticket the caller's error code stands in for was never opened")
	key, value, known, unfound = unfoundID(history[:5], call(`{"ticket_id":"","window":"am"}`), tools)
	assert.True(t, unfound, "no ticket at all is not a ticket")
	assert.Contains(t, unfoundIDResult(key, value, known), "ticket_id is empty")

	for _, arguments := range []string{
		`{"appointment_id":"a1"}`,
		`{"ticket_id":"T1842","window":"am"}`,
		`{"member_id":"ABC123456"}`,
		`{"order_id":"12"}`,
		`{"name":"Maya Chen"}`,
	} {
		_, _, _, unfound = unfoundID(history, call(arguments), tools)
		assert.False(t, unfound, arguments)
	}

	told := []llm.Message{{Role: llm.User, Content: "my ticket number is eighteen forty two"}}
	_, _, _, unfound = unfoundID(told, call(`{"ticket_id":"1842"}`), tools)
	assert.False(t, unfound, "a ticket the caller gives the number of is theirs to give")

	assert.True(t, failedResult(unfoundIDResult("ticket_id", "840", nil)))
}

func TestALastFourThatIsNotFourDigitsIsNotSent(t *testing.T) {
	for arguments, misshapen := range map[string]bool{
		`{"pin":"4471","last4":"9821","address":"14 Cedar Lane"}`: false,
		`{"pin":"4471","last4":"98214"}`:                          true,
		`{"last4":"98t1"}`:                                        true,
		`{"last_four":"129821"}`:                                  true,
		`{"pin":"4471"}`:                                          false,
		`{"last4":""}`:                                            false,
		`{"last4":"9821","address":"98214 Cedar Lane"}`:           true,
		`{"last4":"9821","address":"1400 Cedar Lane"}`:            false,
	} {
		_, _, got := misshapenLastFour(llm.ToolCall{Name: "verify_account", Arguments: arguments})
		assert.Equal(t, misshapen, got, arguments)
	}
	assert.True(t, failedResult(misshapenResult("last4", "98214", nil)))

	heard := []llm.Message{{Role: llm.User, Content: "PIN 4471. Last four. 553017 Oak Road. Is there an outage?"}}
	assert.Contains(t, misshapenResult("last4", "55317", heard), "the last four 5530 followed by the address 17 Oak Road")
}

func TestANameIsBookedAsTheCallerSpelledIt(t *testing.T) {
	heard := []llm.Message{{Role: llm.User, Content: "Sorry about that. The name is Alkafor, o k a f o r. Mom, mom, look at this."}}
	for arguments, misspelled := range map[string]bool{
		`{"name":"Alkafor","party_size":3}`:  true,
		`{"name":"Alcophor","party_size":3}`: true,
		`{"name":"Okafor","party_size":3}`:   false,
		`{"name":"okafor","party_size":3}`:   false,
		`{"name":"Chen","party_size":3}`:     false,
	} {
		_, _, got := misspelledName(heard, llm.ToolCall{Name: "create_reservation", Arguments: arguments})
		assert.Equal(t, misspelled, got, arguments)
	}
	_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: "Name is Alkafor. PIN 4471."}},
		llm.ToolCall{Name: "create_reservation", Arguments: `{"name":"Alkafor"}`})
	assert.False(t, got, "nothing was spelled")
	assert.Contains(t, misspelledResult("Alkafor", "okafor"), "o k a f o r, so it is Okafor")
}

func TestAStandInNameIsNotBookedUnder(t *testing.T) {
	for arguments, stand := range map[string]bool{
		`{"name":"Caller","party_size":3}`: true,
		`{"name":"John Doe"}`:              true,
		`{"name":"Okafor"}`:                false,
		`{"name":"Maria de la Cruz"}`:      false,
		`{"party_size":3}`:                 false,
		`{"name":"Carry on with the caller from what is in your instructions."}`: true,
	} {
		_, got := placeholderName(llm.ToolCall{Name: "create_reservation", Arguments: arguments})
		assert.Equal(t, stand, got, arguments)
	}
	assert.True(t, failedResult(placeholderResult("Caller")))
}
