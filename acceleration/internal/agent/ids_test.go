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
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: didNotWork + `"19821" is not four digits.`})
	_, _, _, got := unheardNumber(refused, llm.ToolCall{Name: "verify_account", Arguments: `{"last4":"1982"}`})
	assert.True(t, got, "a refusal quoting 19821 does not make 1982 heard")

	_, _, heard, _ := unheardNumber(history, llm.ToolCall{Name: "verify_account", Arguments: `{"last4":"1982"}`})
	assert.Contains(t, unheardResult("last4", "1982", heard), "The numbers the caller said are: ")
	assert.True(t, failedResult(unheardResult("last4", "1982", heard)))
}

func TestANameIsBookedAsTheCallerSpelledIt(t *testing.T) {
	heard := []llm.Message{{Role: llm.User, Content: "Sorry about that. The name is Aldenor, o d e n o r. Mom, mom, look at this."}}
	for arguments, misspelled := range map[string]bool{
		`{"name":"Aldenor","party_size":3}`: true,
		`{"name":"Aldinor","party_size":3}`: true,
		`{"name":"Odenor","party_size":3}`:  false,
		`{"name":"odenor","party_size":3}`:  false,
		`{"name":"Chen","party_size":3}`:    false,
	} {
		_, _, got := misspelledName(heard, llm.ToolCall{Name: "create_reservation", Arguments: arguments})
		assert.Equal(t, misspelled, got, arguments)
	}
	_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: "Name is Aldenor. PIN 4471."}},
		llm.ToolCall{Name: "create_reservation", Arguments: `{"name":"Aldenor"}`})
	assert.False(t, got, "nothing was spelled")
	_, _, got = misspelledName([]llm.Message{{Role: llm.User, Content: "The name for the booking is Almeida, that's a l m e i d a, Will be six people."}},
		llm.ToolCall{Name: "create_reservation", Arguments: `{"name":"Almeida"}`})
	assert.False(t, got, "the s of that's is not a spelled letter")
	assert.Contains(t, misspelledResult("Aldenor", "odenor"), "o d e n o r, so it is Odenor")
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
		`{"name":"Odenor"}`:                false,
		`{"name":"Maria de la Cruz"}`:      false,
		`{"party_size":3}`:                 false,
		`{"name":"Carry on with the caller from what is in your instructions."}`: true,
	} {
		_, got := placeholderName(nil, llm.ToolCall{Name: "create_reservation", Arguments: arguments})
		assert.Equal(t, stand, got, arguments)
	}
	assert.True(t, failedResult(placeholderResult("Caller")))
}

func TestAnIDATextResultGaveIsPassedOn(t *testing.T) {
	// A lookup by name and ZIP answers with the bare ID, which is not JSON.
	tools := []string{"find_user_id_by_name_zip", "get_user_details"}
	history := []llm.Message{
		{Role: llm.User, Content: "First name, l e n a. Last name, r u i z. ZIP, nine eight one eight seven."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "find_user_id_by_name_zip"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: "lena_ruiz_3850"},
	}
	call := func(arguments string) llm.ToolCall {
		return llm.ToolCall{Name: "get_user_details", Arguments: arguments}
	}

	_, _, _, unfound := unfoundID(history, call(`{"user_id":"lena_ruiz_3850"}`), tools)
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call(`{"user_id":"lena_ruiz_3851"}`), tools)
	assert.True(t, unfound, "a different ID is still one nothing gave")

	quoted := []llm.Message{
		history[0],
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "get_user_details"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: unfoundIDResult("user_id", "lena_ruiz_3850", nil)},
	}
	_, _, _, unfound = unfoundID(quoted, call(`{"user_id":"lena_ruiz_3850"}`), tools)
	assert.True(t, unfound, "a refusal quoting the ID does not make it returned")
}

func TestAnIDListedUnderAPluralKeyIsPassedOn(t *testing.T) {
	// A profile lists a person's orders and reservations as bare IDs under "orders" and
	// "reservations", with no order_id or reservation_id key beside them.
	tools := []string{"get_profile", "get_order", "get_reservation"}
	history := []llm.Message{
		{Role: llm.User, Content: "I'd like to check on my order."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "get_profile"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"user_id":"u-77",` +
			`"orders":["A-1001","A-1002"],"reservations":["R-55"],` +
			`"payment_methods":{"pm-9":{"id":"pm-9"}}}`},
	}
	call := func(name, arguments string) llm.ToolCall {
		return llm.ToolCall{Name: name, Arguments: arguments}
	}

	_, _, _, unfound := unfoundID(history, call("get_order", `{"order_id":"A-1001"}`), tools)
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("get_reservation", `{"reservation_id":"R-55"}`), tools)
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("get_order", `{"order_id":"A-9999"}`), tools)
	assert.True(t, unfound, "an order the list does not hold is still one nothing gave")
	_, _, _, unfound = unfoundID(history, call("get_order", `{"order_id":"R-55"}`), tools)
	assert.True(t, unfound, "a reservation is not an order")
}

func TestAnIDThatKeysARecordIsPassedOn(t *testing.T) {
	// A collection keyed by its records' IDs gives each ID only as a key: no field of the
	// vehicle says "van-2", yet that is the vehicle a reservation names.
	tools := []string{"list_vehicles", "reserve_vehicle", "list_rooms", "book_room"}
	history := []llm.Message{
		{Role: llm.User, Content: "Reserve the van and book the room on the third floor."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "list_vehicles"}, {ID: "c2", Name: "list_rooms"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"vehicles":{"van-2":{"seats":8},"car-5":{"seats":4}}}`},
		{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"rooms":{"R-12":{"floor":3}},"owner":{"name":"Ana","tier":"gold"}}`},
	}
	call := func(name, arguments string) llm.ToolCall {
		return llm.ToolCall{Name: name, Arguments: arguments}
	}

	_, _, _, unfound := unfoundID(history, call("reserve_vehicle", `{"vehicle_id":"van-2"}`), tools)
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("book_room", `{"room_id":"R-12"}`), tools)
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("reserve_vehicle", `{"vehicle_id":"truck-9"}`), tools)
	assert.True(t, unfound, "a vehicle the collection does not hold is still one nothing gave")
	_, _, _, unfound = unfoundID(history, call("book_room", `{"room_id":"name"}`), tools)
	assert.True(t, unfound, "the fields of one record are not IDs")
}

func TestOnlyACollectionsKeysAreItsRecordsIDs(t *testing.T) {
	// A key is an ID of the records its collection is named for, and of nothing else; a map
	// nobody named as a collection keys nothing.
	tools := []string{"list_vehicles", "reserve_vehicle", "get_order", "get_hours", "book_slot"}
	history := []llm.Message{
		{Role: llm.User, Content: "Reserve a vehicle and book me in."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "list_vehicles"}, {ID: "c2", Name: "get_hours"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"vehicles":{"van-2":{"seats":8}}}`},
		{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"monday":{"open":"9:00"},"tuesday":{"open":"10:00"}}`},
	}
	call := func(name, arguments string) llm.ToolCall {
		return llm.ToolCall{Name: name, Arguments: arguments}
	}

	_, _, _, unfound := unfoundID(history, call("get_order", `{"order_id":"van-2"}`), tools)
	assert.True(t, unfound, "a vehicle's key is not an order")
	_, _, _, unfound = unfoundID(history, call("book_slot", `{"slot_id":"monday"}`), tools)
	assert.True(t, unfound, "days nobody named as a collection are not IDs")
}

func TestAnIDThatKeysARecordAndIsLeftOutIsAskedFor(t *testing.T) {
	tools := harness.Tools{Tools: []harness.Tool{
		{Name: "list_vehicles"},
		{Name: "reserve_vehicle", Parameters: map[string]any{"properties": map[string]any{
			"vehicle_id": map[string]any{"type": "string"}, "days": map[string]any{"type": "number"}}}},
	}}
	history := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "list_vehicles"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"vehicles":{"van-2":{"seats":8}}}`},
	}

	key, known, missing := missingID(history, llm.ToolCall{Name: "reserve_vehicle", Arguments: `{"days":2}`}, tools)

	assert.True(t, missing)
	assert.Equal(t, "vehicle_id", key)
	assert.Equal(t, []string{"van-2 (vehicle_id)"}, known)
}

func TestAListUnderAPluralKeyHoldsThatKeysIDs(t *testing.T) {
	assert.Equal(t, "order_id", listedID("orders"))
	assert.Equal(t, "order_id", listedID("order_ids"))
	assert.Equal(t, "reservation_id", listedID("reservations"))
	assert.Equal(t, "", listedID("address"))
	assert.Equal(t, "", listedID("s"))
}

func TestAnIDTheCallerReadOutIsPassedOn(t *testing.T) {
	// Callers read a user ID out, separators by name, as the transcriber wrote them.
	tools := []string{"get_user_details", "get_reservation_details"}
	call := func(arguments string) llm.ToolCall {
		return llm.ToolCall{Name: "get_user_details", Arguments: arguments}
	}
	for said, id := range map[string]string{
		"User ID is kara underscore lind underscore three eight one seven.":                           "kara_lind_3817",
		"Sure. User ID is t o m a s underscore b e r g underscore eight five five five.":              "tomas_berg_8555",
		"Customer I n e s underscore m o r e t t i underscore nine zero six five. That's my user ID.": "ines_moretti_9065",
	} {
		history := []llm.Message{{Role: llm.User, Content: said}}
		_, _, _, unfound := unfoundID(history, call(`{"user_id":"`+id+`"}`), tools)
		assert.False(t, unfound, said)
	}
	history := []llm.Message{{Role: llm.User, Content: "User ID is kara underscore lind underscore three eight one seven."}}
	_, _, _, unfound := unfoundID(history, call(`{"user_id":"kara_lind_3818"}`), tools)
	assert.True(t, unfound, "a digit the caller did not say is still not theirs")
}

func TestANumberReadOutInDashedGroupsIsOneNumber(t *testing.T) {
	// A caller gives a line in dashed groups, as 555-123-2002.
	history := []llm.Message{{Role: llm.User, Content: "Yeah. It's five five five dash one two three dash two zero zero two."}}
	for arguments, unheard := range map[string]bool{
		`{"phone_number":"5551232002"}`:   false,
		`{"phone_number":"555-123-2002"}`: false,
		`{"phone_number":"5551232003"}`:   true,
		`{"phone_number":"1232002555"}`:   true,
	} {
		_, _, _, got := unheardNumber(history, llm.ToolCall{Name: "get_customer_by_phone", Arguments: arguments})
		assert.Equal(t, unheard, got, arguments)
	}
}

func TestASpellingEndsWhereTheCallerStoppedSpelling(t *testing.T) {
	for said, spelled := range map[string][]string{
		"Sorry about that. The name is Aldenor, o d e n o r. Mom, mom, look at this.": {"odenor"},
		"Hi. This is Dana Fisher. That's f i s c h e r. I need to book a flight":      {"fischer"},
		"The name for the booking is Almeida, that's a l m e i d a, Will be six":      {"almeida"},
		"It's P-A-T-E-L.":                          {"patel"},
		"J. O. H. N. Smith":                        {"john"},
		"First name, l e n a. Last name, r u i z.": {"lena", "ruiz"},
		"Lay Wilson. That's l e i w i l s o n.":    {"leiwilson", "lei", "wilson"},
		"Name is Aldenor. PIN 4471.":               nil,
		"Okay. It's j o eight n s m i t h. Uh":     {"nsmith"},
	} {
		assert.Equal(t, spelled, spelledNames(said), said)
	}
}

func TestANameTheCallerSpelledIsBookedThoughAnotherSpellingIsClose(t *testing.T) {
	// Each is how a transcriber writes a caller spelling a name it first heard otherwise.
	for said, arguments := range map[string]string{
		"Hi. This is Dana Fisher. That's f i s c h e r. I need to book a flight to New York":          `{"passenger_name":"Dana Fischer"}`,
		"Lay Wilson. That's l e i w i l s o n. ZIP is three two two five five":                        `{"first_name":"Lei","last_name":"Wilson"}`,
		"L l e n a. First name, l e n a. Last name, r u i z.":                                         `{"first_name":"Lena","last_name":"Ruiz"}`,
		"Lena Ruiz. That's l e n n last Last name, r u i z. First name, l e n a. Last name, r u i z.": `{"first_name":"Lena","last_name":"Ruiz"}`,
		"Sure. First First name i i v a n Last name, h e r n a n d e z. No, just i v a n":             `{"first_name":"Ivan","last_name":"Hernandez"}`,
		"It's j o eight n s m i t h. Sorry, it's j o h n. That's John. And then s m i t h":            `{"full_name":"John Smith"}`,
	} {
		_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: said}},
			llm.ToolCall{Name: "lookup", Arguments: arguments})
		assert.False(t, got, said)
	}
	for said, arguments := range map[string]string{
		"That's f i s c h e r. I need a flight.": `{"passenger_name":"Dana Fisher"}`,
		"The name is Odenor, o d e n o r.":       `{"name":"Denor"}`,
		"Okay. It's j o eight n s m i t h. Uh":   `{"full_name":"John Smith"}`,
	} {
		_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: said}},
			llm.ToolCall{Name: "lookup", Arguments: arguments})
		assert.True(t, got, "%s: a name no spelling holds exactly is still refused", said)
	}
}
