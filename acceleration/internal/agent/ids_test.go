package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
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

func TestAHandedOutIDLeftOutIsAskedFor(t *testing.T) {
	tools := harness.Tools{Tools: []harness.Tool{
		{Name: "create_ticket"},
		{Name: "dispatch_tech", Parameters: map[string]any{"properties": map[string]any{
			"window": map[string]any{"type": "string"}, "ticket_id": map[string]any{"type": "string"},
		}}},
	}}
	dispatch := llm.ToolCall{Name: "dispatch_tech", Arguments: `{"window":"am"}`}
	_, _, missing := missingID(nil, dispatch, tools)
	assert.False(t, missing, "no ticket has been opened yet")

	ticketed := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "create_ticket"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"id":"T-1842","address":"14 Cedar Lane"}`},
	}
	key, known, missing := missingID(ticketed, dispatch, tools)
	assert.True(t, missing)
	assert.Equal(t, "ticket_id", key)
	assert.Equal(t, []string{"T-1842 (id)"}, known)

	_, _, missing = missingID(ticketed, llm.ToolCall{Name: "dispatch_tech", Arguments: `{"window":"am","ticket_id":"T-1842"}`}, tools)
	assert.False(t, missing)
}

func TestANumberNobodyGaveIsNotSent(t *testing.T) {
	history := []llm.Message{
		{Role: llm.User, Content: "PIN four four seven one, last four, nine eight two one, fourteen Cedar Lane."},
		{Role: llm.User, Content: "Phone is five one two, five five five, oh one four two."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "create_ticket"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"id":"1842"}`},
	}
	for arguments, unheard := range map[string]bool{
		`{"pin":"4471","last4":"9821","address":"14 Cedar Lane"}`: false,
		`{"pin":"4471","last4":"1982"}`:                           true,
		`{"phone":"5125550142"}`:                                  false,
		`{"phone":"512550142"}`:                                   true,
		`{"phone":"512-555-0142"}`:                                false,
		`{"ticket_id":"1842"}`:                                    false,
		`{"dob":"1987-03-04"}`:                                    false,
		`{"window":"am","amount":"50"}`:                           false,
	} {
		_, _, _, got := unheardNumber(history, llm.ToolCall{Name: "verify_account", Arguments: arguments})
		assert.Equal(t, unheard, got, arguments)
	}
	refused := append(history,
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "verify_account"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: misshapenResult("last4", "19821", nil)})
	_, _, _, got := unheardNumber(refused, llm.ToolCall{Name: "verify_account", Arguments: `{"last4":"1982"}`})
	assert.True(t, got, "a refusal quoting 19821 does not make 1982 heard")

	_, _, heard, _ := unheardNumber(history, llm.ToolCall{Name: "verify_account", Arguments: `{"last4":"1982"}`})
	assert.Contains(t, unheardResult("last4", "1982", heard), "The numbers the caller said are: ")
	assert.True(t, failedResult(unheardResult("last4", "1982", heard)))
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
	_, _, got = misspelledName([]llm.Message{{Role: llm.User, Content: "The name for the booking is Alvarez, that's a l v a r e z, Will be six people."}},
		llm.ToolCall{Name: "create_reservation", Arguments: `{"name":"Alvarez"}`})
	assert.False(t, got, "the s of that's is not a spelled letter")
	assert.Contains(t, misspelledResult("Alkafor", "okafor"), "o k a f o r, so it is Okafor")
}

func TestAnArgumentTheToolAsksForIsNotSentEmpty(t *testing.T) {
	strings := func(names ...string) map[string]any {
		properties := map[string]any{}
		for _, name := range names {
			properties[name] = map[string]any{"type": "string"}
		}
		return map[string]any{"type": "object", "properties": properties}
	}
	tools := harness.Tools{Tools: []harness.Tool{
		{Name: "verify_account", Description: "Verify the account with PIN, last four of the card or account, and service address.",
			Parameters: strings("pin", "last4", "address")},
		{Name: "verify_identity", Description: "Verify a patient with name, DOB, and member ID or phone.",
			Parameters: strings("name", "dob", "member_id", "phone")},
	}}
	key, empty := emptyNamed(llm.ToolCall{Name: "verify_account", Arguments: `{"address":"","last4":"9821","pin":"4471"}`}, tools)
	assert.True(t, empty)
	assert.Equal(t, "address", key)
	key, empty = emptyNamed(llm.ToolCall{Name: "verify_account", Arguments: `{"pin":"4471","last4":"9821"}`}, tools)
	assert.True(t, empty, "left out is empty too")
	assert.Equal(t, "address", key)
	_, empty = emptyNamed(llm.ToolCall{Name: "verify_identity", Arguments: `{"name":"Maya Chen","dob":"1987-03-04","member_id":"ABC123456"}`}, tools)
	assert.False(t, empty, "phone left out beside a member ID")
	_, empty = emptyNamed(llm.ToolCall{Name: "verify_account", Arguments: `{"address":"14 Cedar Lane","last4":"9821","pin":"4471"}`}, tools)
	assert.False(t, empty)
	_, empty = emptyNamed(llm.ToolCall{Name: "verify_identity", Arguments: `{"name":"Maya Chen","dob":"1987-03-04","member_id":"ABC123456","phone":""}`}, tools)
	assert.False(t, empty, "phone is the alternative to the member ID")
	_, empty = emptyNamed(llm.ToolCall{Name: "verify_identity", Arguments: `{"name":"Maya Chen","dob":"1987-03-04","member_id":"","phone":"5125550142"}`}, tools)
	assert.False(t, empty, "the member ID is the alternative to the phone")
	assert.True(t, failedResult(emptyResult("verify_account", "address")))
}

func TestAStandInNameIsNotBookedUnder(t *testing.T) {
	for arguments, stand := range map[string]bool{
		`{"name":"Caller","party_size":3}`: true,
		`{"name":"John Doe"}`:              true,
		`{"name":"Okafor"}`:                false,
		`{"name":"Maria de la Cruz"}`:      false,
		`{"party_size":3}`:                 false,
		`{"name":"Carry on with the caller from what is in your instructions."}`: true,
		`{"name":"Let Alvarez"}`:  true,
		`{"name":"Wing Wait"}`:    true,
		`{"name":"Will Alvarez"}`: false,
	} {
		_, got := placeholderName(llm.ToolCall{Name: "create_reservation", Arguments: arguments})
		assert.Equal(t, stand, got, arguments)
	}
	assert.True(t, failedResult(placeholderResult("Caller")))
}
