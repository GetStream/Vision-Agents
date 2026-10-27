package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestAnIDNothingGaveTheModelIsNotPassedOn(t *testing.T) {
	tools := []string{"confirm_guest", "find_booking", "move_booking", "open_case", "send_courier"}
	history := []llm.Message{
		{Role: llm.User, Content: "Rosa Lind, card K T R five eight three zero two seven, boiler error seven twenty"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "confirm_guest"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":true,"guest_id":"g1"}`},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "find_booking"}}},
		{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"bookings":[{"id":"a1","date":"Thursday"}]}`},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c3", Name: "open_case"}}},
		{Role: llm.ToolResult, ToolCallID: "c3", Content: `{"case_id":"C-6631"}`},
	}
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "some_tool", Arguments: arguments} }

	key, value, known, unfound := unfoundID(history, call(`{"booking_id":"g1","date":"Tuesday"}`), tools, "")
	assert.True(t, unfound, "a guest's ID is not a booking's")
	assert.Equal(t, "booking_id", key)
	assert.Equal(t, "g1", value)
	assert.Contains(t, known, "a1 (id)")
	assert.Contains(t, unfoundIDResult(key, value, known), "a1 (id)")

	_, _, _, unfound = unfoundID(history, call(`{"booking_id":"Friday 3pm in the garden room"}`), tools, "")
	assert.True(t, unfound)
	_, _, _, unfound = unfoundID(history[:5], call(`{"case_id":"720"}`), tools, "")
	assert.True(t, unfound, "a case the caller's error code stands in for was never opened")
	key, value, known, unfound = unfoundID(history[:5], call(`{"case_id":"","slot":"am"}`), tools, "")
	assert.True(t, unfound, "no case at all is not a case")
	assert.Contains(t, unfoundIDResult(key, value, known), "case_id is empty")

	for _, arguments := range []string{
		`{"booking_id":"a1"}`,
		`{"case_id":"C6631","slot":"am"}`,
		`{"card_id":"KTR583027"}`,
		`{"order_id":"12"}`,
		`{"name":"Rosa Lind"}`,
	} {
		_, _, _, unfound = unfoundID(history, call(arguments), tools, "")
		assert.False(t, unfound, arguments)
	}

	told := []llm.Message{{Role: llm.User, Content: "my case number is sixty six thirty one"}}
	_, _, _, unfound = unfoundID(told, call(`{"case_id":"6631"}`), tools, "")
	assert.False(t, unfound, "a case the caller gives the number of is theirs to give")

	assert.True(t, failedResult(unfoundIDResult("case_id", "915", nil)))
}

func TestAnIDTheCallerWasAskedForOrTheOperatorGaveIsPassedOn(t *testing.T) {
	tools := []string{"open_ticket", "get_ticket", "get_account"}
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "get_ticket", Arguments: arguments} }
	asked := []llm.Message{
		{Role: llm.Assistant, Content: "Sure. What's your ticket number?"},
		{Role: llm.User, Content: "It's forty four ten."},
	}
	_, _, _, unfound := unfoundID(asked, call(`{"ticket_id":"4410"}`), tools, "")
	assert.False(t, unfound, "the caller said it in answer to being asked for the ticket")

	_, _, _, unfound = unfoundID(asked[1:], call(`{"ticket_id":"4410"}`), tools, "")
	assert.True(t, unfound, "said with nothing to say what it is, it is not a ticket")
	later := []llm.Message{
		asked[0],
		{Role: llm.User, Content: "Hold on."},
		{Role: llm.Assistant, Content: "Take your time."},
		{Role: llm.User, Content: "Forty four ten."},
	}
	_, _, _, unfound = unfoundID(later, call(`{"ticket_id":"4410"}`), tools, "")
	assert.True(t, unfound, "only the answer to the question that asked for it")

	account := llm.ToolCall{Name: "get_account", Arguments: `{"account_id":"ACC-7781"}`}
	_, _, _, unfound = unfoundID(nil, account, tools, "You are the help line. The caller's account_id is ACC-7781.")
	assert.False(t, unfound, "the operator gave it")
	_, _, _, unfound = unfoundID(nil, account, tools, "")
	assert.True(t, unfound)
	_, _, _, unfound = unfoundID(nil, account, tools, "Accounts look like ACC-77810.")
	assert.True(t, unfound, "only an ID the instructions give whole")
}

func TestAHandedOutIDLeftOutIsAskedFor(t *testing.T) {
	tools := harness.Tools{Tools: []harness.Tool{
		{Name: "open_case"},
		{Name: "send_courier", Parameters: map[string]any{"properties": map[string]any{
			"slot": map[string]any{"type": "string"}, "case_id": map[string]any{"type": "string"},
		}}},
	}}
	courier := llm.ToolCall{Name: "send_courier", Arguments: `{"slot":"am"}`}
	_, _, missing := missingID(nil, courier, tools)
	assert.False(t, missing, "no case has been opened yet")

	opened := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "open_case"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"id":"C-6631","address":"14 Birch Road"}`},
	}
	key, known, missing := missingID(opened, courier, tools)
	assert.True(t, missing)
	assert.Equal(t, "case_id", key)
	assert.Equal(t, []string{"C-6631 (id)"}, known)

	_, _, missing = missingID(opened, llm.ToolCall{Name: "send_courier", Arguments: `{"slot":"am","case_id":"C-6631"}`}, tools)
	assert.False(t, missing)
}

func TestANumberNobodyGaveIsNotSent(t *testing.T) {
	history := []llm.Message{
		{Role: llm.User, Content: "PIN two seven four five, last four, six three zero five, fourteen Birch Road."},
		{Role: llm.User, Content: "Phone is four zero eight, five five five, oh three nine one."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "open_case"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"id":"6631"}`},
	}
	for arguments, unheard := range map[string]bool{
		`{"pin":"2745","last4":"6305","address":"14 Birch Road"}`: false,
		`{"pin":"2745","last4":"5630"}`:                           true,
		`{"phone":"4085550391"}`:                                  false,
		`{"phone":"408550391"}`:                                   true,
		`{"phone":"408-555-0391"}`:                                false,
		`{"case_id":"6631"}`:                                      false,
		`{"dob":"1990-11-21"}`:                                    false,
		`{"slot":"am","amount":"50"}`:                             false,
	} {
		_, _, _, got := unheardNumber(history, llm.ToolCall{Name: "confirm_member", Arguments: arguments}, "")
		assert.Equal(t, unheard, got, arguments)
	}
	refused := append(history,
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "confirm_member"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: didNotWork + `"56305" is not four digits.`})
	_, _, _, got := unheardNumber(refused, llm.ToolCall{Name: "confirm_member", Arguments: `{"last4":"5630"}`}, "")
	assert.True(t, got, "a refusal quoting 56305 does not make 5630 heard")

	_, _, heard, _ := unheardNumber(history, llm.ToolCall{Name: "confirm_member", Arguments: `{"last4":"5630"}`}, "")
	assert.Contains(t, unheardResult("last4", "5630", heard), "The numbers the caller said are: ")
	assert.True(t, failedResult(unheardResult("last4", "5630", heard)))
}

func TestOnlyANearMissOfANumberTheCallerSaidIsRefused(t *testing.T) {
	history := []llm.Message{
		{Role: llm.User, Content: "Same trip as this year, two thousand twenty six, but for next year. My member number is four one seven three."},
	}
	for arguments, unheard := range map[string]bool{
		`{"year":"2027"}`:          false,
		`{"member_number":"4173"}`: false,
		`{"member_number":"1473"}`: true,
		`{"member_number":"413"}`:  false,
		`{"store":"8812"}`:         false,
	} {
		_, _, _, got := unheardNumber(history, llm.ToolCall{Name: "book_trip", Arguments: arguments}, "")
		assert.Equal(t, unheard, got, arguments)
	}
	tooShort := []llm.Message{{Role: llm.User, Content: "Code four one seven three two."}}
	_, _, _, got := unheardNumber(tooShort, llm.ToolCall{Name: "redeem", Arguments: `{"code":"4729"}`}, "")
	assert.False(t, got, "a number nothing like any the caller said")
	_, _, _, got = unheardNumber(tooShort, llm.ToolCall{Name: "redeem", Arguments: `{"code":"4132"}`}, "")
	assert.True(t, got, "one digit dropped")

	_, _, _, got = unheardNumber([]llm.Message{{Role: llm.User, Content: "Store eight one two eight, please."}},
		llm.ToolCall{Name: "check_stock", Arguments: `{"store":"8812"}`}, "Our main store is 8812.")
	assert.False(t, got, "the operator gave it")
}

func TestANameIsBookedAsTheCallerSpelledIt(t *testing.T) {
	heard := []llm.Message{{Role: llm.User, Content: "Sorry about that. The name is Aldenor, o d e n o r. Hold on, the dog is barking."}}
	for arguments, misspelled := range map[string]bool{
		`{"name":"Aldenor","guests":3}`: true,
		`{"name":"Aldinor","guests":3}`: false,
		`{"name":"Odenor","guests":3}`:  false,
		`{"name":"odenor","guests":3}`:  false,
		`{"name":"Moreau","guests":3}`:  false,
	} {
		_, _, got := misspelledName(heard, llm.ToolCall{Name: "book_room", Arguments: arguments})
		assert.Equal(t, misspelled, got, arguments)
	}
	_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: "Name is Aldenor. PIN 2745."}},
		llm.ToolCall{Name: "book_room", Arguments: `{"name":"Aldenor"}`})
	assert.False(t, got, "nothing was spelled")
	_, _, got = misspelledName([]llm.Message{{Role: llm.User, Content: "The name for the booking is Almeida, that's a l m e i d a, for four of us."}},
		llm.ToolCall{Name: "book_room", Arguments: `{"name":"Almeida"}`})
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
		{Name: "confirm_member", Description: "Confirm the member with PIN, last four of the card or phone, and home address.",
			Parameters: strings("pin", "last4", "address")},
		{Name: "confirm_guest", Description: "Confirm a guest with name, DOB, and card ID or phone.",
			Parameters: strings("name", "dob", "card_id", "phone")},
	}}
	key, empty := emptyNamed(llm.ToolCall{Name: "confirm_member", Arguments: `{"address":"","last4":"6305","pin":"2745"}`}, tools)
	assert.True(t, empty)
	assert.Equal(t, "address", key)
	key, empty = emptyNamed(llm.ToolCall{Name: "confirm_member", Arguments: `{"pin":"2745","last4":"6305"}`}, tools)
	assert.True(t, empty, "left out is empty too")
	assert.Equal(t, "address", key)
	_, empty = emptyNamed(llm.ToolCall{Name: "confirm_guest", Arguments: `{"name":"Rosa Lind","dob":"1990-11-21","card_id":"KTR583027"}`}, tools)
	assert.False(t, empty, "phone left out beside a card ID")
	_, empty = emptyNamed(llm.ToolCall{Name: "confirm_member", Arguments: `{"address":"14 Birch Road","last4":"6305","pin":"2745"}`}, tools)
	assert.False(t, empty)
	_, empty = emptyNamed(llm.ToolCall{Name: "confirm_guest", Arguments: `{"name":"Rosa Lind","dob":"1990-11-21","card_id":"KTR583027","phone":""}`}, tools)
	assert.False(t, empty, "phone is the alternative to the card ID")
	_, empty = emptyNamed(llm.ToolCall{Name: "confirm_guest", Arguments: `{"name":"Rosa Lind","dob":"1990-11-21","card_id":"","phone":"4085550391"}`}, tools)
	assert.False(t, empty, "the card ID is the alternative to the phone")
	assert.True(t, failedResult(emptyResult("confirm_member", "address")))
}

func TestAStandInNameIsNotBookedUnder(t *testing.T) {
	for arguments, stand := range map[string]bool{
		`{"name":"Caller","guests":3}`: true,
		`{"name":"John Doe"}`:          true,
		`{"name":"Odenor"}`:            false,
		`{"name":"Maria de la Cruz"}`:  false,
		`{"guests":3}`:                 false,
		`{"name":"Carry on with the caller from what is in your instructions."}`:           true,
		`{"name":"what a tool or your colleague"}`:                                         true,
		`{"name":"one short sentence"}`:                                                    true,
		`{"event_name":"Northside Pottery Club Spring Open Day"}`:                          false,
		`{"file_name":"minutes of the annual general meeting 2026"}`:                       false,
		`{"name":"Carry Lind"}`:                                                            false,
		`{"event_name":"On With The Show"}`:                                                false,
		`{"name":"Booked by: Carry on with the caller from what is in your instructions"}`: true,
	} {
		_, got := placeholderName(nil, llm.ToolCall{Name: "book_room", Arguments: arguments})
		assert.Equal(t, stand, got, arguments)
	}
	assert.True(t, failedResult(placeholderResult("Caller")))
}

func TestAnIDATextResultGaveIsPassedOn(t *testing.T) {
	// A lookup by name and ZIP answers with the bare ID, which is not JSON.
	tools := []string{"find_user_by_name_zip", "get_user_profile"}
	history := []llm.Message{
		{Role: llm.User, Content: "First name, l e n a. Last name, r u i z. ZIP, nine eight one eight seven."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "find_user_by_name_zip"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: "lena_ruiz_4162"},
	}
	call := func(arguments string) llm.ToolCall {
		return llm.ToolCall{Name: "get_user_profile", Arguments: arguments}
	}

	_, _, _, unfound := unfoundID(history, call(`{"user_id":"lena_ruiz_4162"}`), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call(`{"user_id":"lena_ruiz_4163"}`), tools, "")
	assert.True(t, unfound, "a different ID is still one nothing gave")

	quoted := []llm.Message{
		history[0],
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "get_user_profile"}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: unfoundIDResult("user_id", "lena_ruiz_4162", nil)},
	}
	_, _, _, unfound = unfoundID(quoted, call(`{"user_id":"lena_ruiz_4162"}`), tools, "")
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

	_, _, _, unfound := unfoundID(history, call("get_order", `{"order_id":"A-1001"}`), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("get_reservation", `{"reservation_id":"R-55"}`), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("get_order", `{"order_id":"A-9999"}`), tools, "")
	assert.True(t, unfound, "an order the list does not hold is still one nothing gave")
	_, _, _, unfound = unfoundID(history, call("get_order", `{"order_id":"R-55"}`), tools, "")
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

	_, _, _, unfound := unfoundID(history, call("reserve_vehicle", `{"vehicle_id":"van-2"}`), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("book_room", `{"room_id":"R-12"}`), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history, call("reserve_vehicle", `{"vehicle_id":"truck-9"}`), tools, "")
	assert.True(t, unfound, "a vehicle the collection does not hold is still one nothing gave")
	_, _, _, unfound = unfoundID(history, call("book_room", `{"room_id":"name"}`), tools, "")
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

	_, _, _, unfound := unfoundID(history, call("get_order", `{"order_id":"van-2"}`), tools, "")
	assert.True(t, unfound, "a vehicle's key is not an order")
	_, _, _, unfound = unfoundID(history, call("book_slot", `{"slot_id":"monday"}`), tools, "")
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
	tools := []string{"get_user_profile", "get_booking"}
	call := func(arguments string) llm.ToolCall {
		return llm.ToolCall{Name: "get_user_profile", Arguments: arguments}
	}
	for said, id := range map[string]string{
		"User ID is kara underscore lind underscore two nine four three.":                              "kara_lind_2943",
		"Sure. User ID is t o m a s underscore b e r g underscore seven one zero six.":                 "tomas_berg_7106",
		"Customer I n e s underscore m o r e t t i underscore five two seven four. That's my user ID.": "ines_moretti_5274",
	} {
		history := []llm.Message{{Role: llm.User, Content: said}}
		_, _, _, unfound := unfoundID(history, call(`{"user_id":"`+id+`"}`), tools, "")
		assert.False(t, unfound, said)
	}
	history := []llm.Message{{Role: llm.User, Content: "User ID is kara underscore lind underscore two nine four three."}}
	_, _, _, unfound := unfoundID(history, call(`{"user_id":"kara_lind_2944"}`), tools, "")
	assert.True(t, unfound, "a digit the caller did not say is still not theirs")
}

func TestANumberReadOutInDashedGroupsIsOneNumber(t *testing.T) {
	// A caller gives a line in dashed groups, as 555-246-8013.
	history := []llm.Message{{Role: llm.User, Content: "Yeah. It's five five five dash two four six dash eight zero one three."}}
	for arguments, unheard := range map[string]bool{
		`{"phone_number":"5552468013"}`:   false,
		`{"phone_number":"555-246-8013"}`: false,
		`{"phone_number":"2468013555"}`:   true,
		`{"phone_number":"555246813"}`:    true,
	} {
		_, _, _, got := unheardNumber(history, llm.ToolCall{Name: "find_customer_by_phone", Arguments: arguments}, "")
		assert.Equal(t, unheard, got, arguments)
	}
}

func TestASpellingEndsWhereTheCallerStoppedSpelling(t *testing.T) {
	for said, spelled := range map[string][]string{
		"Sorry about that. The name is Aldenor, o d e n o r. Hold on, the dog is barking.": {"odenor"},
		"Hi. This is Mara Kline. That's k l e i n. I need to book a flight":                {"klein"},
		"The name for the booking is Almeida, that's a l m e i d a, for four of us":        {"almeida"},
		"It's K-O-V-A-C.":                          {"kovac"},
		"J. O. H. N. Smith":                        {"john"},
		"First name, l e n a. Last name, r u i z.": {"lena", "ruiz"},
		"Kay Jensen. That's k a i j e n s e n.":    {"kaijensen", "kai", "jensen"},
		"Name is Aldenor. PIN 2745.":               nil,
		"Okay. It's j o eight n s m i t h. Uh":     {"nsmith"},
	} {
		assert.Equal(t, spelled, spelledNames(said), said)
	}
}

func TestANameTheCallerSpelledIsBookedThoughAnotherSpellingIsClose(t *testing.T) {
	// Each is how a transcriber writes a caller spelling a name it first heard otherwise.
	for said, arguments := range map[string]string{
		"Hi. This is Mara Kline. That's k l e i n. I need to book a flight to New York":               `{"passenger_name":"Mara Klein"}`,
		"Kay Jensen. That's k a i j e n s e n. ZIP is three two two five five":                        `{"first_name":"Kai","last_name":"Jensen"}`,
		"L l e n a. First name, l e n a. Last name, r u i z.":                                         `{"first_name":"Lena","last_name":"Ruiz"}`,
		"Lena Ruiz. That's l e n n last Last name, r u i z. First name, l e n a. Last name, r u i z.": `{"first_name":"Lena","last_name":"Ruiz"}`,
		"Sure. First First name e e l i a s Last name, m o r e n o. No, just e l i a s":               `{"first_name":"Elias","last_name":"Moreno"}`,
		"It's j o eight n s m i t h. Sorry, it's j o h n. That's John. And then s m i t h":            `{"full_name":"John Smith"}`,
	} {
		_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: said}},
			llm.ToolCall{Name: "lookup", Arguments: arguments})
		assert.False(t, got, said)
	}
	for said, arguments := range map[string]string{
		"That's k l e i n. I need a flight.":   `{"passenger_name":"Mara Kline"}`,
		"The name is Odenor, o d e n o r.":     `{"name":"Denor"}`,
		"Okay. It's j o eight n s m i t h. Uh": `{"full_name":"John Smith"}`,
	} {
		_, _, got := misspelledName([]llm.Message{{Role: llm.User, Content: said}},
			llm.ToolCall{Name: "lookup", Arguments: arguments})
		assert.False(t, got, "%s: no word the transcriber wrote beside the spelling", said)
	}
}

func TestOnlyTheWordWrittenBesideASpellingIsRefused(t *testing.T) {
	said := []llm.Message{{Role: llm.User, Content: "Surname Jonas, j o n e s. And my son Joseph is coming too."}}
	for arguments, misspelled := range map[string]bool{
		`{"guest_name":"Jonas"}`:                       true,
		`{"guest_name":"Ann Jonas"}`:                   true,
		`{"guest_name":"Jones"}`:                       false,
		`{"guest_name":"Joseph Jones"}`:                false,
		`{"name":"Jones","companion_name":"Joseph"}`:   false,
		`{"name":"Jones","restaurant_name":"Jonah's"}`: false,
	} {
		_, _, got := misspelledName(said, llm.ToolCall{Name: "add_guest", Arguments: arguments})
		assert.Equal(t, misspelled, got, arguments)
	}
	after := []llm.Message{{Role: llm.User, Content: "It's j o n e s, Jonas."}}
	_, spelled, got := misspelledName(after, llm.ToolCall{Name: "add_guest", Arguments: `{"guest_name":"Jonas"}`})
	assert.True(t, got, "the word right after the spelling")
	assert.Equal(t, "jones", spelled)
}
