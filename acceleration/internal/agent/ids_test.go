package agent

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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

func TestAKindWordInsideAnotherWordIsNotTheCallerNamingTheID(t *testing.T) {
	tools := []string{"open_case", "get_case"}
	call := llm.ToolCall{Name: "get_case", Arguments: `{"case_id":"6631"}`}
	for _, c := range []struct {
		name    string
		history []llm.Message
		unfound bool
	}{
		{"said beside its kind", []llm.Message{{Role: llm.User, Content: "My case number is 6631."}}, false},
		{"said beside its plural", []llm.Message{{Role: llm.User, Content: "Cases 6631 and 6632."}}, false},
		{"said joined to its kind by #", []llm.Message{{Role: llm.User, Content: "It is case#6631."}}, false},
		{"said joined to its kind by a dash", []llm.Message{{Role: llm.User, Content: "My ticket-4521 and case-6631."}}, false},
		{"said beside a kind joined to another word by a dash", []llm.Message{{Role: llm.User, Content: "My case-number is 6631."}}, false},
		{"said beside a word that holds its kind", []llm.Message{{Role: llm.User, Content: "The showcase number is 6631."}}, true},
		{"said beside a word that begins with its kind", []llm.Message{{Role: llm.User, Content: "My casework is 6631."}}, true},
		{"answered to its kind asked", []llm.Message{{Role: llm.Assistant, Content: "What's your case number?"}, {Role: llm.User, Content: "6631"}}, false},
		{"answered to a kind joined to another word asked", []llm.Message{{Role: llm.Assistant, Content: "What's your case-number?"}, {Role: llm.User, Content: "6631"}}, false},
		{"answered to a kind joined by # asked", []llm.Message{{Role: llm.Assistant, Content: "What's your case#?"}, {Role: llm.User, Content: "6631"}}, false},
		{"answered to a word that holds its kind asked", []llm.Message{{Role: llm.Assistant, Content: "What's your suitcase number?"}, {Role: llm.User, Content: "6631"}}, true},
	} {
		_, _, _, unfound := unfoundID(c.history, call, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
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

func TestAStandInNameIsNotBookedUnder(t *testing.T) {
	for arguments, stand := range map[string]bool{
		`{"name":"Caller","guests":3}`: true,
		`{"name":"John Doe"}`:          true,
		`{"name":"Odenor"}`:            false,
		`{"name":"Maria de la Cruz"}`:  false,
		`{"guests":3}`:                 false,
		`{"name":"Carry on with the caller from the note that follows."}`:           true,
		`{"name":"what a tool or your colleague"}`:                                  true,
		`{"name":"one short sentence"}`:                                             true,
		`{"event_name":"Northside Pottery Club Spring Open Day"}`:                   false,
		`{"file_name":"minutes of the annual general meeting 2026"}`:                false,
		`{"name":"Carry Lind"}`:                                                     false,
		`{"event_name":"On With The Show"}`:                                         false,
		`{"name":"Booked by: Carry on with the caller from the note that follows"}`: true,
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

func TestAnIDPastedInALinkIsPassedOn(t *testing.T) {
	// A person pastes a link to what they want read, and the tools take the link's parts as
	// IDs: the channel is one segment of its path, or one value of its query. Nothing names
	// the channel in words.
	tools := []string{"search_channels", "read_channel", "read_message"}
	user := func(text string) []llm.Message {
		return []llm.Message{{Role: llm.User, Content: text}}
	}
	channel := llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"C0EX4MPL1"}`}
	message := llm.ToolCall{Name: "read_message", Arguments: `{"message_id":"1793120000000100"}`}
	cases := []struct {
		name    string
		history []llm.Message
		call    llm.ToolCall
		unfound bool
	}{
		{"a path segment", user("Can you read this? https://team.example.com/archives/C0EX4MPL1/p1793120000000100"), channel, false},
		{"a query value", user("what does https://team.example.com/view?thread=1793120000.000100&cid=C0EX4MPL1 say"), channel, false},
		{"a link in angle brackets with a label", user("<https://team.example.com/archives/C0EX4MPL1|the thread>"), channel, false},
		{"a link in quotes", user(`{"link":"https://team.example.com/archives/C0EX4MPL1","note":"the thread"}`), channel, false},
		{"a scheme in capitals", user("HTTPS://TEAM.EXAMPLE.COM/ARCHIVES/C0EX4MPL1"), channel, false},
		{"a percent-escaped segment", user("https://team.example.com/archives/C0EX%34MPL1"), channel, false},
		{"the ID alone, with no link and no word for it", user("read C0EX4MPL1 please"), channel, true},
		{"the ID inside a longer segment", user("https://team.example.com/archives/C0EX4MPL1/p1793120000000100"), message, true},
		{"the ID inside a longer query value", user("https://team.example.com/view?ref=xC0EX4MPL1"), channel, true},
		{"the ID as a query key", user("https://team.example.com/view?C0EX4MPL1=1"), channel, true},
		{"the ID as the host", user("https://C0EX4MPL1/archives"), channel, true},
		{"the ID in the fragment", user("https://team.example.com/archives#C0EX4MPL1"), channel, true},
		{"the ID in a link with no scheme", user("team.example.com/archives/C0EX4MPL1"), channel, true},
		{"the ID in a link that is not a URI", user("https://team.example.com/%zz/C0EX4MPL1"), channel, true},
		{"the ID in a link the agent wrote", []llm.Message{
			{Role: llm.Assistant, Content: "Is it https://team.example.com/archives/C0EX4MPL1 you mean?"},
			{Role: llm.User, Content: "yes, read it"},
		}, channel, true},
		{"an escaped slash inside a segment", user("https://team.example.com/archives/C0X%2FC0EX4MPL1"), channel, true},
		{"the ID in a link a failed tool call wrote", []llm.Message{
			{Role: llm.User, Content: "read the thread I starred"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "search_channels"}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"ok":false,"error":"no access to https://team.example.com/archives/C0EX4MPL1"}`},
		}, channel, true},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(c.history, c.call, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestAnIDAJSONResultWritesOutIsPassedOn(t *testing.T) {
	// A search answers with an object whose "results" is a list written for people, the
	// channels' IDs in brackets, under no key that names them.
	tools := []string{"search_channels", "read_channel"}
	listed := `{"results":"# Channels for: help\n\n## Channels (2 results)\n` +
		`1. #help (C0HELP002) - public_channel - Purpose: questions\n` +
		`2. #general (C0GEN0001) - public_channel","pagination_info":"no more results"}`
	history := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "What's new in the help channel?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "search_channels"}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	read := func(id string) llm.ToolCall {
		return llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"` + id + `"}`}
	}

	_, _, _, unfound := unfoundID(history(listed), read("C0HELP002"), tools, "")
	assert.False(t, unfound)
	_, _, _, unfound = unfoundID(history(listed), read("C0HELP003"), tools, "")
	assert.True(t, unfound, "a channel the list does not hold is still one nothing gave")
	_, _, _, unfound = unfoundID(history(`[{"lines":["1. #help (C0HELP002) - public_channel"]}]`), read("C0HELP002"), tools, "")
	assert.False(t, unfound, "text in a list inside a list is read too")
	_, _, _, unfound = unfoundID(history(`{"note":"channel C0HELP002"}`), read("C0HELP002"), tools, "")
	assert.False(t, unfound, "two words are written out")
	_, _, _, unfound = unfoundID(history(`{"user_id":"u-77","channel":"C0HELP002"}`), read("C0HELP002"), tools, "")
	assert.True(t, unfound, "one word under a key that names no ID is not written out")
	_, _, _, unfound = unfoundID(history(`{"ok":false,"error":"channel C0HELP002 was not found"}`), read("C0HELP002"), tools, "")
	assert.True(t, unfound, "a failure quoting the ID does not make it returned")
}

func TestAnIDAJSONResultEchoesOrWritesBesideAnotherWordIsNotPassedOn(t *testing.T) {
	// A result's text writes back what the model asked for, or a value of another kind: the
	// model must not take either as an ID a tool returned.
	tools := []string{"search_channels", "read_channel", "find_booking", "open_case"}
	answered := func(arguments, result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "Can you help?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "look_up", Arguments: arguments}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "some_tool", Arguments: arguments} }
	cases := []struct {
		name      string
		history   []llm.Message
		arguments string
	}{
		{
			"a search writing back its query under a heading for channels",
			answered(`{"query":"C0FAKE01"}`, `{"results":"# Channels for: C0FAKE01\n\n## Channels (0 results)","pagination_info":"no more results"}`),
			`{"channel_id":"C0FAKE01"}`,
		},
		{
			"a search writing back its query in a sentence about channels",
			answered(`{"query":"C0FAKE01"}`, `{"results":"No channels match C0FAKE01"}`),
			`{"channel_id":"C0FAKE01"}`,
		},
		{
			"a lookup writing back the time it was asked about",
			answered(`{"name":"Ana","time":"7pm"}`, `{"bookings":[],"message":"No bookings found for Ana at 7pm"}`),
			`{"booking_id":"7pm"}`,
		},
		{
			"an error code in a sentence that names no case",
			answered(`{}`, `{"status":"declined","detail":"processor answered with code 720 today"}`),
			`{"case_id":"720"}`,
		},
		{
			"a card in a note that names no case",
			answered(`{}`, `{"profile":{"note":"Card KTR583027 on file"}}`),
			`{"case_id":"KTR583027"}`,
		},
		{
			"a search joining words of its query the tool writes back apart",
			answered(`{"query":"C0FAKE01-general"}`, `{"results":"Channels matching C0FAKE01 general"}`),
			`{"channel_id":"C0FAKE01"}`,
		},
		{
			"a search whose query spaces the ID the tool writes back whole",
			answered(`{"query":"C0 FAKE 01"}`, `{"results":"Channels for C0FAKE01"}`),
			`{"channel_id":"C0FAKE01"}`,
		},
		{
			"a channel on a line that names none, under one that does",
			answered(`{}`, `{"results":"## Channels (0 results)\nRequested by U0ASKER1"}`),
			`{"channel_id":"U0ASKER1"}`,
		},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(c.history, call(c.arguments), tools, "")
		assert.True(t, unfound, c.name)
	}

	_, _, _, unfound := unfoundID(answered(`{"query":"help"}`, `{"results":"1. #help (C0HELP002) - public_channel"}`), call(`{"channel_id":"C0HELP002"}`), tools, "")
	assert.False(t, unfound, "a channel the search was not sent and lists as one is returned")
}

func TestOnlyALinkPartOfLettersAndDigitsIsAnIDOfAnyKind(t *testing.T) {
	// A number alone in a link is as often a page, a year or a size as an ID, and a word alone
	// is a name or a section: neither is an ID of a kind the link does not name.
	tools := []string{"read_channel", "open_case"}
	user := func(text string) []llm.Message { return []llm.Message{{Role: llm.User, Content: text}} }
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "some_tool", Arguments: arguments} }
	cases := []struct {
		name      string
		history   []llm.Message
		arguments string
		unfound   bool
	}{
		{"a page number in the query", user("see https://team.example.com/list?page=1 thanks"), `{"case_id":"1"}`, true},
		{"a year in the path", user("https://team.example.com/docs/2024/intro"), `{"case_id":"2024"}`, true},
		{"a word in the path", user("https://team.example.com/archives/general"), `{"channel_id":"general"}`, true},
		{"a number in a link whose words name its kind", user("https://team.example.com/cases/48213"), `{"case_id":"48213"}`, false},
		{"letters and digits in the path", user("https://team.example.com/archives/C0EX4MPL1"), `{"channel_id":"C0EX4MPL1"}`, false},
		{"letters and digits in the query", user("https://team.example.com/view?cid=C0EX4MPL1"), `{"channel_id":"C0EX4MPL1"}`, false},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(c.history, call(c.arguments), tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestALinkIsReadAsThePersonWroteIt(t *testing.T) {
	// A shared channel passes the person's text on inside a JSON envelope, which writes & as
	// \u0026, and a chat may send it escaped for HTML, & as &amp;: every value of the link's
	// query is still one the person gave.
	tools := []string{"read_channel"}
	channel := llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"C0EX4MPL1"}`}
	link := "https://team.example.com/view?thread=1793120000.000100&cid=C0EX4MPL1"
	envelope, err := json.Marshal(struct {
		Author struct {
			ID string `json:"user_id"`
		} `json:"author"`
		Text string `json:"text"`
	}{Text: "what does " + link + " say"})
	require.NoError(t, err)
	require.Contains(t, string(envelope), `\u0026cid=`, "the envelope escapes the link's &")
	user := func(text string) []llm.Message { return []llm.Message{{Role: llm.User, Content: text}} }
	cases := []struct {
		name    string
		history []llm.Message
		unfound bool
	}{
		{"a link in a JSON envelope", user(string(envelope)), false},
		{"a link escaped for HTML", user("what does https://team.example.com/view?thread=1793120000.000100&amp;cid=C0EX4MPL1 say"), false},
		{"a link in HTML-escaped angle brackets", user("&lt;https://team.example.com/archives/C0EX4MPL1&gt;"), false},
		{"a link under a key of a JSON object", user(`{"https://team.example.com/archives/C0EX4MPL1":"the thread"}`), false},
		{"a message that starts with a number, then a link", user("2 things in https://team.example.com/archives/C0EX4MPL1"), false},
		{"a link in an envelope that escapes / around text escaped for HTML", user(`{"text":"she said &quot;read https:\/\/team.example.com\/archives\/C0EX4MPL1&quot;"}`), false},
		{"a JSON envelope with no link", user(`{"author":{"user_id":"C0EX4MPL1"},"text":"read C0EX4MPL1"}`), true},
		{"an HTML-escaped link the agent wrote", []llm.Message{
			{Role: llm.Assistant, Content: "Is it https://team.example.com/view?a=1&amp;cid=C0EX4MPL1 you mean?"},
			{Role: llm.User, Content: "yes"},
		}, true},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(c.history, channel, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestAWrittenLineGivesOnlyTheIDBesideTheWordForItsKind(t *testing.T) {
	// A line written for people names a thing by its kind, its name and its ID side by side.
	// Another ID on the line, the creator's or the sender's, is not one of that kind, nor is
	// one after a role a label ends in or a named thing's role (AI-2884).
	tools := []string{"search_channels", "read_channel", "lookup_phone_number"}
	answered := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "What's new?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "search_channels", Arguments: `{"query":"help"}`}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	call := func(arguments string) llm.ToolCall { return llm.ToolCall{Name: "some_tool", Arguments: arguments} }
	cases := []struct {
		name      string
		result    string
		arguments string
		unfound   bool
	}{
		{"a listed channel with its creator", `{"results":"1. #help (C0HELP002) - public_channel - Creator: Ana Lind - 2026-01-02 - Not Archived"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its kind and name", `{"messages":"Channel: #help (C0HELP002)\nhello"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its kind, name and the word ID", `{"messages":"Channel: #help (ID: C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel named by several words after its label", `{"messages":"Channel: Group DM (ID: C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a tool line naming a channel by several words after its label", `{"messages":"Channel: Team ops-room west (ID: C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel named by several words after a date and its label", `{"messages":"Created (today) Channel: Group DM (ID: C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a kind of two words named by several words after its label", `{"results":"Phone number: Main line (ID: P0NUM001)"}`, `{"phone_number_id":"P0NUM001"}`, false},
		{"a channel mentioned in the text", `{"messages":"see channel <#C0HELP002|help>"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a person bracketed after another label", `{"results":"1. #help (C0HELP002) - public_channel - Creator: Ana Lind (<@U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after a label of another kind", `{"messages":"Channel: #help (C0HELP002) Creator: Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after another bracketed ID", `{"messages":"Channel: #help (C0HELP002) (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after a dash", `{"messages":"Channel: #help - Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after a label with no bracket before it", `{"messages":"Channel: Group DM Creator: Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after a name that is no label", `{"messages":"Channel #x by Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a channel after a name and a dash", `{"messages":"Channel #help - C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its name and the word ID", `{"messages":"Channel #help ID: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a kind of two words", `{"results":"Phone number P0NUM001 - active"}`, `{"phone_number_id":"P0NUM001"}`, false},
		{"a person by a channel's name", `{"messages":"Channel #x by U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a creator at the end of a listed channel", `{"results":"1. #help (C0HELP002) - public_channel - created by U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a channel's name and a person's ID after it", `{"messages":"Channel #help Ana U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"the word ID and a person's ID after a name", `{"messages":"Channel #help by ID U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person who joined a channel", `{"messages":"U0ANA001 joined channel #help"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person who joined a channel, the word ID between", `{"messages":"U0ANA001 id joined channel #help"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person before a channel's name and kind", `{"messages":"U0ANA001 wrote in the #help channel"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a channel beside the plural of its kind", `{"results":"Channels: #help (C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel beside its key", `{"results":"Posted. channel_id: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a person by a name that ends in the kind's word", `{"messages":"#ops-channel by U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person by a name that holds the kind's word after _", `{"messages":"#team_channel by U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a label that ends in the kind's word", `{"messages":"Ops-channel: Ana Lind (ID: U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a label that opens with the kind's word", `{"messages":"Channel owner: Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a listed channel's type", `{"results":"1. #help (C0HELP002) - public_channel"}`, `{"channel_id":"public_channel"}`, true},
		{"a person after a channel's name and a word holding \"id\"", `{"results":"Channel #onboarding guide: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person labelled as a channel and as a sender", `{"results":"Channel: Ana Lind (ID: U0ANA001)\nFrom: Ana Lind <a@example.com> (ID: U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person labelled as a channel's owner and as a participant", `{"results":"Channel: #help owner (U0ANA001)\nParticipants: Ana Lind (ID: U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a group DM whose participants and sender are labelled", `{"results":"Channel: Group DM (ID: C0GRP0001)\nParticipants: Ana Lind (ID: U0ANA001), Bob Ray (ID: U0BOB0001)\nFrom: Ana Lind <a@example.com> (ID: U0ANA001)"}`, `{"channel_id":"C0GRP0001"}`, false},
		{"a person before a phrase that names the kind's owner", `{"messages":"Ana Lind (U0ANA001) - channel owner"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person before the kind and a role", `{"messages":"Owner U0ANA001 channel admin"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person before the plural of the kind and a role", `{"messages":"Ana U0ANA001 channels moderator since May"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person before a key whose last word is a role", `{"messages":"U0ANA001 - public_channel_owner"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a channel before its kind in brackets", `{"results":"#help C0HELP002 (channel)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel before its type at the end of the line", `{"results":"#help (C0HELP002) private_channel"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel before its type and a comma", `{"results":"#help (C0HELP002) - private_channel, archived"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel before its kind and a bar", `{"results":"C0HELP002 channel | 4 members"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel in a text table row, columns two spaces apart", `{"results":"ID  TYPE  MEMBERS\nC0HELP002  public_channel  12"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel in a text table row, columns a tab apart", `{"results":"C0HELP002\tchannel\t12"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel in a text table row, a name before it", `{"results":"#help  C0HELP002  channel  12"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a person in a text table row of channels, the role column after the kind", `{"results":"Ana Lind  U0ANA001  channel owner"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a kind of two words after the ID", `{"results":"Main line P0NUM001 phone number"}`, `{"phone_number_id":"P0NUM001"}`, false},
		{"a kind of two words after the ID, then a role", `{"results":"Ana P0NUM001 phone number owner"}`, `{"phone_number_id":"P0NUM001"}`, true},
		{"a person after a label that ends in a role", `{"results":"Channel owner: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a lowercase label that ends in another role", `{"messages":"channel moderator: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a label of the plural kind and a role", `{"results":"Channels admin: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a label of a role and the word ID", `{"results":"Channel owner ID: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after a label of a kind of two words and a role", `{"results":"Phone number owner: P0NUM001"}`, `{"phone_number_id":"P0NUM001"}`, true},
		{"a person after a named channel's role", `{"messages":"#help channel admin U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person bracketed after a named channel's role", `{"messages":"#help channel moderator (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a person after an owner label of a named channel", `{"messages":"Owner of #help: U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a channel after its kind and a name with no mark", `{"results":"channel help (C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its kind, a name with no mark and a dash", `{"results":"Channel help - C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its label and a name with no mark", `{"results":"Channel: help C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a label of its kind, a name and the word ID", `{"results":"Channel: help, ID: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a label of its kind, a name and a bar", `{"results":"Channel: help | ID: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a label of its kind, a name and a dash", `{"results":"Channel: help - ID: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a phone number after a label of its kind, a name and the word ID", `{"results":"Phone number: Main, ID: P0NUM001"}`, `{"phone_number_id":"P0NUM001"}`, false},
		{"a channel after its kind, a name with no mark and a date after a colon", `{"results":"channel help (C0HELP002) - Created: 2026-11-05"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel in a numbered list", `{"results":"1. Channel help (C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel and a link after it", `{"results":"channel help (C0HELP002), docs at https://docs.example.com/help"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a sentence that holds another channel's marked name", `{"results":"Moved from #random to channel help (C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after another channel's marked name and a comma", `{"results":"#random, channel help (C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a label that ends in its marked name", `{"results":"Channel #help: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after a sentence that ends in its marked name", `{"results":"Posted to channel #help: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its name and kind", `{"results":"#help channel C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a channel after its name and a label of its kind", `{"results":"#help channel: C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(answered(c.result), call(c.arguments), tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestALineLabellingAnIDAsAnotherKindOnlyDoubtsALabel(t *testing.T) {
	// Another line's label doubts a label that gives the ID (AI-2823), not a line that has the
	// ID beside its own kind, and not a line a call echoes back the ID it was sent in.
	tools := []string{"list_channels", "send_message", "read_channel"}
	read := llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"C0HELP002"}`}
	history := func(listed, echoed, sent string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "What's new?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "list_channels", Arguments: `{"query":"help"}`}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"results":"` + listed + `"}`},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "send_message", Arguments: sent}}},
			{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"results":"` + echoed + `"}`},
		}
	}
	list := "1. #help (C0HELP002) - public_channel"
	for _, c := range []struct {
		name           string
		listed, echoed string
		sent           string
		unfound        bool
	}{
		{"a list row and a send's echo", list, "Message posted to: #help (C0HELP002)", `{"channel_id":"C0HELP002","text":"hi"}`, false},
		{"a list row and a name line", list, "Name: #help (C0HELP002)", `{"channel_id":"C0HELP002","text":"hi"}`, false},
		{"a list row and a channel name line", list, "Channel name: #help (C0HELP002)", `{"channel_id":"C0HELP002","text":"hi"}`, false},
		{"a list row and a line whose name holds a colon", list, "Channel: Standup 10:30 (ID: C0HELP002)", `{"channel_id":"C0HELP002","text":"hi"}`, false},
		{"a label line and a send's echo", "Channel: Group DM (ID: C0HELP002)", "Posted to: Group DM (ID: C0HELP002)", `{"channel_id":"C0HELP002","text":"hi"}`, false},
		{"a list row and a name line a call not sent the ID wrote", list, "Name: #help (C0HELP002)", `{"text":"hi"}`, false},
		{"a label line and a label a call not sent the ID wrote", "Channel: Group DM (ID: C0HELP002)", "From: Ana Lind (ID: C0HELP002)", `{"text":"hi"}`, true},
	} {
		_, _, _, unfound := unfoundID(history(c.listed, c.echoed, c.sent), read, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestAnIDInALinkAToolReturnedIsPassedOn(t *testing.T) {
	// A tool's record may give a channel's ID only in its link: the link names what each of
	// its parts is by where it stands, as one the person pastes does.
	tools := []string{"search_channels", "read_channel"}
	answered := func(arguments, result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "What's new in the help channel?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "search_channels", Arguments: arguments}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	channel := llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"C0HELP002"}`}
	cases := []struct {
		name    string
		history []llm.Message
		unfound bool
	}{
		{"a link under a key", answered(`{"query":"help"}`, `{"permalink":"https://team.example.com/archives/C0HELP002"}`), false},
		{"a record whose link alone gives the ID", answered(`{"query":"help"}`, `{"results":"### Result 1 of 2\nName: #help\nCreator: Ana Lind (<@U0ANA001)\nPermalink: [link](https://team.example.com/archives/C0HELP002)\nChannel Type: public_channel"}`), false},
		{"a link whose slashes are escaped for JSON", answered(`{"query":"help"}`, `{"permalink":"https:\/\/team.example.com\/archives\/C0HELP002"}`), false},
		{"a link whose & is escaped for HTML", answered(`{"query":"help"}`, `{"results":"see https://team.example.com/view?thread=1793120000.000100&amp;cid=C0HELP002"}`), false},
		{"a link in text that is escaped for HTML inside JSON", answered(`{"query":"help"}`, `{"results":"she said &quot;read https:\/\/team.example.com\/archives\/C0HELP002&quot;"}`), false},
		{"a link that holds what the call was sent", answered(`{"query":"C0HELP002"}`, `{"results":"Nothing found: https://team.example.com/search?q=C0HELP002"}`), true},
		{"a link that holds another ID", answered(`{"query":"help"}`, `{"permalink":"https://team.example.com/archives/C0HELP003"}`), true},
		{"a link a failed call wrote", answered(`{"query":"help"}`, `{"ok":false,"error":"no access to https://team.example.com/archives/C0HELP002"}`), true},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(c.history, channel, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestALinkAToolReturnedGivesOnlyAnIDOfTheKindItsWordsName(t *testing.T) {
	// A link names its parts by where they stand, but not what kind of thing each is: the
	// words around it do, those of its line, of the keys it stands under or of the tool's
	// name (AI-2852). Its own host and path say where it points, not what kind (AI-2881).
	tools := []string{"search", "get_user", "search_channels", "read_channel", "open_case"}
	answered := func(tool, result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "What's new?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: tool, Arguments: `{"query":"help"}`}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	cases := []struct {
		name, tool, result, arguments string
		unfound                       bool
	}{
		{"a person's profile link given as a channel", "get_user", `{"profile":"https://team.example.com/team/U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a message's link given as a case", "search_channels", `{"permalink":"https://team.example.com/archives/C0HELP002/p1789925000740629"}`, `{"case_id":"p1789925000740629"}`, true},
		{"a link no word names the kind of", "search", `{"results":"see https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, true},
		{"a link under a key that names the kind", "search", `{"channel":{"url":"https://team.example.com/archives/C0HELP002"}}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link under a key of two words that names the kind", "search", `{"channel_link":"https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link on a line that names the kind", "search", `{"results":"Channel link: https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link on a line that names a case", "search", `{"results":"Case opened: https://desk.example.com/view/CS0A6631"}`, `{"case_id":"CS0A6631"}`, false},
		{"a link on another line than the one that names the kind", "search", `{"results":"Channels found\nsee https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, true},
		{"a link from a tool whose name names the kind", "read_channel", `{"results":"see https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link in a result that answers no call", "", `{"next":"https://team.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, true},
		{"a link whose host names the kind", "search", `{"results":"see https://channels.example.com/u/U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose subdomain names the kind", "search", `{"results":"https://channel.example.com/U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose path names the kind", "search", `{"results":"see https://team.example.com/channels/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, true},
		{"a link whose path names the kind just before the ID", "search", `{"results":"see https://team.example.com/channel/U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose last word names the kind, after the ID", "search", `{"results":"see https://team.example.com/U0ANA001/channel"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose query key names the kind", "search", `{"results":"see https://team.example.com/view?channel=U0ANA001"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose path names a case", "search", `{"results":"Opened https://desk.example.com/cases/CS0A6631"}`, `{"case_id":"CS0A6631"}`, true},
		{"a link whose path is a label", "search", `{"results":"https://team.example.com/channel: Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link before the kind's word in brackets", "search", `{"results":"https://team.example.com/archives/C0HELP002 (channel)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link whose text names the kind", "search", `{"results":"[channel](https://team.example.com/archives/C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link whose host names the kind, under a key that names it", "search", `{"channel":"https://channels.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link written again without its scheme, in a chat's form", "search", `{"results":"<https://team.example.com/channel/U0ANA001|team.example.com/channel/U0ANA001>"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link written again without its scheme, as markdown text", "search", `{"results":"[channels.example.com/u/U0ANA001](https://channels.example.com/u/U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link written again without its scheme, the copy alone after the first", "search", `{"results":"see https://team.example.com/view/U0ANA001 (team.example.com/view/U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link written again without its scheme, on a line that names the kind", "search", `{"results":"Channel link: <https://team.example.com/archives/C0HELP002|team.example.com/archives/C0HELP002>"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link written again without its scheme, under a key that names the kind", "search", `{"channel":"[team.example.com/archives/C0HELP002](https://team.example.com/archives/C0HELP002)"}`, `{"channel_id":"C0HELP002"}`, false},
		{"a link written again without its scheme, after a word, in a chat's form", "search", `{"results":"see <https://team.example.com/channel/U0ANA001|team.example.com/channel/U0ANA001>"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link written again without its scheme, as a label's text", "search", `{"results":"<https://team.example.com/channel|team.example.com/channel>: Ana Lind (U0ANA001)"}`, `{"channel_id":"U0ANA001"}`, true},
		{"a link whose host names the kind, from a tool whose name names it", "read_channel", `{"results":"see https://channels.example.com/archives/C0HELP002"}`, `{"channel_id":"C0HELP002"}`, false},
	}
	for _, c := range cases {
		history := answered(c.tool, c.result)
		if c.tool == "" {
			history[2].ToolCallID = "c9"
		}
		_, _, _, unfound := unfoundID(history, llm.ToolCall{Name: "some_tool", Arguments: c.arguments}, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestTextTheAgentWroteIsNoSourceOfAnID(t *testing.T) {
	// A tool that reads a channel back returns what the agent posted there or said: an ID the
	// agent made up and wrote is still one nothing gave it (AI-2853).
	tools := []string{"search_channels", "read_channel", "send_message"}
	said := func(text string) llm.Message { return llm.Message{Role: llm.Assistant, Content: text} }
	posted := func(text string) []llm.Message {
		arguments, err := json.Marshal(map[string]string{"channel_id": "C0HELP002", "text": text})
		require.NoError(t, err)
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "send_message", Arguments: string(arguments)}}},
			{Role: llm.ToolResult, ToolCallID: "s1", Content: `{"ok":true}`},
		}
	}
	read := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_channel", Arguments: `{"channel_id":"C0HELP002"}`}}},
			{Role: llm.ToolResult, ToolCallID: "r1", Content: result},
		}
	}
	listed := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "l1", Name: "search_channels", Arguments: `{"query":"help"}`}}},
		{Role: llm.ToolResult, ToolCallID: "l1", Content: `{"results":"1. #help (C0HELP002) - public_channel"}`},
	}
	named := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "l2", Name: "search_channels", Arguments: `{"query":"help"}`}}},
		{Role: llm.ToolResult, ToolCallID: "l2", Content: `{"results":"Channel: #help (C0HELP002)"}`},
	}
	cases := []struct {
		name    string
		history [][]llm.Message
		id      string
		unfound bool
	}{
		{"a link the agent posted, read back", [][]llm.Message{posted("see https://team.example.com/archives/C0FAKE01"), read(`{"messages":"Agent: see https://team.example.com/archives/C0FAKE01\nAna: thanks"}`)}, "C0FAKE01", true},
		{"a link the agent posted, read back in brackets", [][]llm.Message{posted("Moved to https://team.example.com/archives/C0FAKE01 today"), read(`{"messages":[{"user":"bot","text":"Moved to <https://team.example.com/archives/C0FAKE01> today"}]}`)}, "C0FAKE01", true},
		{"a link the agent said, read back", [][]llm.Message{{said("It is at https://team.example.com/archives/C0FAKE01")}, read(`{"messages":"Bot: It is at https://team.example.com/archives/C0FAKE01\nAna: ok"}`)}, "C0FAKE01", true},
		{"a line the agent said, read back", [][]llm.Message{{said("Channel: #x (C0FAKE01)")}, read(`{"messages":"Bot: Channel: #x (C0FAKE01)\nAna: ok"}`)}, "C0FAKE01", true},
		{"a line the agent posted, read back escaped for HTML", [][]llm.Message{posted("Channel: Q&A room (C0FAKE01)"), read(`{"messages":"Bot: Channel: Q&amp;A room (C0FAKE01)"}`)}, "C0FAKE01", true},
		{"a line the agent posted, read back as text", [][]llm.Message{posted("Use channel C0FAKE01 now"), read("[bot] Use channel C0FAKE01 now\n[ana] ok")}, "C0FAKE01", true},
		{"a person repeating, in text, the channel the agent posted", [][]llm.Message{posted("Use channel C0FAKE01 now"), read("[bot] Use channel C0FAKE01 now\n[ana] C0FAKE01 is right")}, "C0FAKE01", false},
		{"a link another person wrote after the agent's", [][]llm.Message{{said("It is at https://team.example.com/archives/C0FAKE01")}, read(`{"messages":"Ana: no, it is https://team.example.com/archives/C0REAL001"}`)}, "C0REAL001", false},
		{"a channel the agent named before a search listed it", [][]llm.Message{{said("Let me look for C0HELP002 in the list.")}, listed}, "C0HELP002", false},
		{"a channel the agent read before a search listed it", [][]llm.Message{read(`{"ok":false,"error":"not_in_channel"}`), listed}, "C0HELP002", false},
		{"a line a tool returned before the agent wrote it", [][]llm.Message{named, {said("Channel: #help (C0HELP002)")}}, "C0HELP002", false},
		{"a listed channel the agent repeated, read back", [][]llm.Message{listed, {said("Channel: #help (C0HELP002)")}, read(`{"messages":"Bot: Channel: #help (C0HELP002)"}`)}, "C0HELP002", false},
		// A message read back keeps the channel the agent named in it as a field of its own (AI-2883).
		{"a channel the agent posted, read back in a field", [][]llm.Message{posted("use <#C0FAKE01>"), read(`{"messages":[{"text":"use <#C0FAKE01>","blocks":[{"type":"channel","channel_id":"C0FAKE01"}]}]}`)}, "C0FAKE01", true},
		{"a channel the agent said, read back in a nested field", [][]llm.Message{{said("Moved to <#C0FAKE01|help> today")}, read(`{"messages":[{"text":"Moved to <#C0FAKE01|help> today","blocks":[{"elements":[{"elements":[{"type":"channel","channel_id":"C0FAKE01"}]}]}]}]}`)}, "C0FAKE01", true},
		{"a channel the agent posted, read back as an id", [][]llm.Message{posted("Ask in C0FAKE01 please"), read(`{"messages":[{"text":"Ask in C0FAKE01 please","attachments":[{"id":"C0FAKE01"}]}]}`)}, "C0FAKE01", true},
		{"a channel the agent posted, read back escaped for HTML in a field", [][]llm.Message{posted("Q&A is in <#C0FAKE01>"), read(`{"messages":[{"text":"Q&amp;A is in <#C0FAKE01>","channel_id":"C0FAKE01"}]}`)}, "C0FAKE01", true},
		{"a channel another result gave, read back in a field", [][]llm.Message{posted("use <#C0FAKE01>"), read(`{"messages":[{"text":"use <#C0FAKE01>","blocks":[{"channel_id":"C0FAKE01"}]}]}`), read(`{"channel":{"id":"C0FAKE01","name":"help"}}`)}, "C0FAKE01", false},
		{"a channel a result gave before the agent posted it", [][]llm.Message{read(`{"channel":{"id":"C0FAKE01","name":"help"}}`), posted("use <#C0FAKE01>"), read(`{"messages":[{"text":"use <#C0FAKE01>","blocks":[{"channel_id":"C0FAKE01"}]}]}`)}, "C0FAKE01", false},
		{"a field holding a channel the agent's read-back line does not name", [][]llm.Message{posted("use <#C0FAKE01>"), read(`{"messages":[{"text":"use <#C0FAKE01>","channel_id":"C0REAL001"}]}`)}, "C0REAL001", false},
		{"a field read back with a line the agent wrote after it", [][]llm.Message{read(`{"messages":[{"text":"use <#C0FAKE01>","channel_id":"C0FAKE01"}]}`), {said("use <#C0FAKE01>")}}, "C0FAKE01", false},
	}
	for _, c := range cases {
		history := []llm.Message{{Role: llm.User, Content: "What's new?"}}
		for _, part := range c.history {
			history = append(history, part...)
		}
		_, _, _, unfound := unfoundID(history, llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"` + c.id + `"}`}, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
}

func TestAnIDAResultReadsBackIsNotListedAsReturned(t *testing.T) {
	history := []llm.Message{
		{Role: llm.User, Content: "What's new?"},
		{Role: llm.Assistant, Content: "use <#C0FAKE01>"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_channel", Arguments: `{"channel_id":"C0HELP002"}`}}},
		{Role: llm.ToolResult, ToolCallID: "r1", Content: `{"channel_id":"C0HELP002","messages":[{"text":"use <#C0FAKE01>","blocks":[{"channel_id":"C0FAKE01"}]}]}`},
	}
	_, _, known, unfound := unfoundID(history, llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"C0FAKE01"}`}, []string{"read_channel"}, "")
	require.True(t, unfound)
	assert.Equal(t, []string{"C0HELP002 (channel_id)"}, known, "the field read back is not offered as returned; the channel read is")
	_, _, known, unfound = unfoundID(history, llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":""}`}, []string{"read_channel"}, "")
	require.True(t, unfound)
	assert.ElementsMatch(t, []string{"C0HELP002 (channel_id)", "C0FAKE01 (channel_id)"}, known, "an empty ID is answered with every ID returned")
}

func TestAnArgumentSentEscapedIsStillAnEcho(t *testing.T) {
	// The model may write an argument with JSON escapes; a result writing it back still only
	// echoes it.
	tools := []string{"search_channels", "read_channel"}
	answered := func(arguments, result string) []llm.Message {
		return []llm.Message{
			{Role: llm.User, Content: "Can you help?"},
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "search_channels", Arguments: arguments}}},
			{Role: llm.ToolResult, ToolCallID: "c1", Content: result},
		}
	}
	listed := func(id string) string { return `{"results":"1. #fake (` + id + `) - public_channel"}` }
	cases := []struct {
		name      string
		arguments string
		id        string
		unfound   bool
	}{
		{"a query written in unicode escapes", `{"query":"\u0043\u0030FAKE01"}`, "C0FAKE01", true},
		{"a query that is a number", `{"query":12345678}`, "12345678", true},
		{"a query in a list", `{"queries":["help","\u0043\u0030FAKE01"]}`, "C0FAKE01", true},
		{"arguments that are not JSON", `C0FAKE01`, "C0FAKE01", true},
		{"a query that does not hold the ID", `{"query":"\u0068elp"}`, "C0FAKE01", false},
	}
	for _, c := range cases {
		_, _, _, unfound := unfoundID(answered(c.arguments, listed(c.id)), llm.ToolCall{Name: "read_channel", Arguments: `{"channel_id":"` + c.id + `"}`}, tools, "")
		assert.Equal(t, c.unfound, unfound, c.name)
	}
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
		`{"guest_name":"Jonas"}`:                     true,
		`{"guest_name":"Ann Jonas"}`:                 true,
		`{"guest_name":"Jones"}`:                     false,
		`{"guest_name":"Joseph Jones"}`:              false,
		`{"name":"Jones","companion_name":"Joseph"}`: false,
		`{"name":"Jones","shop_name":"Joles"}`:       false,
	} {
		_, _, got := misspelledName(said, llm.ToolCall{Name: "add_guest", Arguments: arguments})
		assert.Equal(t, misspelled, got, arguments)
	}
	after := []llm.Message{{Role: llm.User, Content: "It's j o n e s, Jonas."}}
	_, spelled, got := misspelledName(after, llm.ToolCall{Name: "add_guest", Arguments: `{"guest_name":"Jonas"}`})
	assert.True(t, got, "the word right after the spelling")
	assert.Equal(t, "jones", spelled)
}

func TestUnsourcedTransfer(t *testing.T) {
	call := llm.ToolCall{Name: "transfer", Arguments: `{"to":"+15550001111"}`}
	for told, accepted := range map[string]bool{
		"Hand off to +15550001111.":                        true,
		"Hand off to +1 555 000 1111":                      true,
		"Hand off to +1 (555) 000-1111.":                   true,
		"Hand off to +1.555.000.1111":                      true,
		"Hand off to 1-555-000-1111":                       true,
		"Hand off to (555) 000-1111":                       true,
		"1. Sales: +15550001111\n2. Billing: +15550002222": true,
		"Transfer to +1 555 000 1111 (9am-5pm)":            true,
		"Sales is +15550001111. 24/7 support":              true,
		"Billing: (555) 000-1111 (9am-5pm)":                true,
		"Billing: (555) 000-1111. 24/7":                    true,
		"Hand off to +15550001112":                         false,
		"Hand off to +1 555 000 1112":                      false,
		"Hand off to +155500011112":                        false,
		"Hand off to (555) 000-1112":                       false,
		"Hand off to +44 555 000 1111":                     false,
		"Hand off to 555 000 1111 2222":                    false,
		"Hand off to +44 (555) 000-1111":                   false,
		"Hand off to +1 555 000 1111 2222":                 false,
		"Hand off to 555 000 1111 (ext. 22)":               true,
		"Hand off to 2. 555-000-1111":                      true,
	} {
		_, unsourced := unsourcedTransfer(nil, call, told)
		assert.Equal(t, accepted, !unsourced, told)
	}
	german := llm.ToolCall{Name: "transfer", Arguments: `{"to":"+13012345678"}`}
	_, unsourced := unsourcedTransfer(nil, german, "Hand off to +49 (30) 1234 5678")
	assert.True(t, unsourced, "a bracketed area code after another country code")
}

func TestUnsourcedTransferIgnoresANumberQuotedByAFailedTool(t *testing.T) {
	call := llm.ToolCall{Name: "transfer", Arguments: `{"to":"+15550001111"}`}
	for _, content := range []string{
		didNotWork + "no agent is on duty at +15550001111.",
		`{"error":"line +15550001111 is busy"}`,
	} {
		_, unsourced := unsourcedTransfer([]llm.Message{{Role: llm.ToolResult, Content: content}}, call, "")
		assert.True(t, unsourced, content)
	}
}

func TestANumberTheAgentWroteIsNoSourceOfANumber(t *testing.T) {
	// A tool that reads back a note or a message the agent sent returns the number the agent wrote:
	// a near miss of the caller's that the agent sent is still one nobody gave (AI-2886). What the
	// agent said on the call is not counted: it reads a number back to confirm it, and a tool may
	// well list it.
	said := func(text string) []llm.Message { return []llm.Message{{Role: llm.Assistant, Content: text}} }
	posted := func(text string) []llm.Message {
		arguments, err := json.Marshal(map[string]string{"channel_id": "C0HELP002", "text": text})
		require.NoError(t, err)
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "send_message", Arguments: string(arguments)}}},
			{Role: llm.ToolResult, ToolCallID: "s1", Content: `{"ok":true}`},
		}
	}
	read := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_notes", Arguments: `{}`}}},
			{Role: llm.ToolResult, ToolCallID: "r1", Content: result},
		}
	}
	lookup := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "l1", Name: "find_member", Arguments: `{"last4":"5630"}`}}},
		{Role: llm.ToolResult, ToolCallID: "l1", Content: `{"member":"Ana Lind","last4":"5630"}`},
	}
	cases := []struct {
		name    string
		history [][]llm.Message
		told    string
		unheard bool
	}{
		{"a number the agent posted, read back", [][]llm.Message{posted("Your last four are 5630, I have noted that."), read(`{"notes":"Agent: Your last four are 5630, I have noted that."}`)}, "", true},
		{"a number the agent saved in a note, read back with a record", [][]llm.Message{posted("Card ends 5630"), read("Ana Lind | member | notes: Card ends 5630")}, "", true},
		{"a number the agent said, listed by a tool", [][]llm.Message{said("Your last four are 5630, I have noted that."), read(`{"notes":"Agent: Your last four are 5630, I have noted that."}`)}, "", false},
		{"a number the agent said to confirm it, listed as text", [][]llm.Message{said("Card ending 5630?"), read("Card ending 5630: Ana Lind")}, "", false},
		{"a number the agent saved in a note, read back with the tool's own", [][]llm.Message{posted("Card ends 5630"), read("Ana Lind | card 5630 | notes: Card ends 5630")}, "", false},
		{"a number the agent posted, read back in a list", [][]llm.Message{posted("Member's last four: 5630"), read(`{"messages":[{"user":"bot","text":"Member's last four: 5630"}]}`)}, "", true},
		{"a number the agent posted, read back as text", [][]llm.Message{posted("Last four 5630 confirmed"), read("[bot] Last four 5630 confirmed\n[ana] ok")}, "", true},
		{"a number the agent posted, read back escaped for HTML", [][]llm.Message{posted("Q&A: last four 5630"), read(`{"notes":"Q&amp;A: last four 5630"}`)}, "", true},
		{"a number the agent posted, read back on another line than a tool's", [][]llm.Message{posted("Your last four are 5630, I have noted that."), read(`{"notes":"Agent: Your last four are 5630, I have noted that.\nCard on file ends 5630"}`)}, "", false},
		{"a number a tool returned before the agent posted it", [][]llm.Message{lookup, posted("Your last four are 5630, I have noted that."), read(`{"notes":"Agent: Your last four are 5630, I have noted that."}`)}, "", false},
		{"a number a tool returned before the agent copied its line into a note", [][]llm.Message{posted("Caller said last four 5630"), read("Card on file ends 5630 for Ana"), posted("Card on file ends 5630 for Ana")}, "", false},
		{"half a number either side of the agent's line, read back", [][]llm.Message{posted("Note 5630 here"), read("Ref 56Note 5630 here30")}, "", true},
		{"letters that lowercase to another length, before the agent's line", [][]llm.Message{posted("Note 5630"), read("\u212A\u212A\u212A Note 5630")}, "", true},
		{"a number the operator gave", [][]llm.Message{posted("Your last four are 5630, I have noted that."), read(`{"notes":"Agent: Your last four are 5630, I have noted that."}`)}, "Test members end in 5630.", false},
		{"a number the agent sent alone, returned", [][]llm.Message{lookup}, "", false},
	}
	for _, c := range cases {
		history := []llm.Message{{Role: llm.User, Content: "My last four are six three zero five."}}
		for _, part := range c.history {
			history = append(history, part...)
		}
		_, _, _, unheard := unheardNumber(history, llm.ToolCall{Name: "confirm_member", Arguments: `{"last4":"5630"}`}, c.told)
		assert.Equal(t, c.unheard, unheard, c.name)
	}
	heard := []llm.Message{
		{Role: llm.User, Content: "My last four are five six three zero."},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "save_note", Arguments: `{"text":"Your last four are 5630, I have noted that."}`}}},
		{Role: llm.ToolResult, ToolCallID: "s1", Content: `{"ok":true}`},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_notes", Arguments: `{}`}}},
		{Role: llm.ToolResult, ToolCallID: "r1", Content: `{"notes":"Agent: Your last four are 5630, I have noted that."}`},
	}
	_, _, _, unheard := unheardNumber(heard, llm.ToolCall{Name: "confirm_member", Arguments: `{"last4":"5630"}`}, "")
	assert.False(t, unheard, "the caller said it")
}

func TestATransferNumberTheAgentWroteIsNoSource(t *testing.T) {
	// A tool that reads back a note or a message the agent sent returns the number the agent wrote,
	// in whatever form it wrote it: the agent is no source of a number to put a caller through to
	// (AI-2886). What the agent said on the call is not counted: it reads a number back to confirm
	// it, and a tool may well list it.
	said := func(text string) []llm.Message { return []llm.Message{{Role: llm.Assistant, Content: text}} }
	posted := func(text string) []llm.Message {
		arguments, err := json.Marshal(map[string]string{"channel_id": "C0HELP002", "text": text})
		require.NoError(t, err)
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "send_message", Arguments: string(arguments)}}},
			{Role: llm.ToolResult, ToolCallID: "s1", Content: `{"ok":true}`},
		}
	}
	read := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "r1", Name: "read_channel", Arguments: `{"channel_id":"C0HELP002"}`}}},
			{Role: llm.ToolResult, ToolCallID: "r1", Content: result},
		}
	}
	directory := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "d1", Name: "find_department", Arguments: `{"name":"sales"}`}}},
		{Role: llm.ToolResult, ToolCallID: "d1", Content: `{"department":"Sales","phone":"+1 (555) 010-0199"}`},
	}
	checked := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "check_line", Arguments: `{"number":"+15550100199"}`}}},
		{Role: llm.ToolResult, ToolCallID: "c1", Content: `{"status":"Sales +1 555 010 0199 is open"}`},
	}
	noted := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "u1", Name: "update_contact", Arguments: `{"id":"42","notes":"Call Ana back at +1 555 010 0199"}`}}},
		{Role: llm.ToolResult, ToolCallID: "u1", Content: `{"ok":true}`},
	}
	departments := func(result string) []llm.Message {
		return []llm.Message{
			{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "d2", Name: "list_departments", Arguments: `{}`}}},
			{Role: llm.ToolResult, ToolCallID: "d2", Content: result},
		}
	}
	checkedLocal := []llm.Message{
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "check_line", Arguments: `{"number":"(555) 010-0199"}`}}},
		{Role: llm.ToolResult, ToolCallID: "c2", Content: `{"status":"Sales +1 555 010 0199 is open"}`},
	}
	cases := []struct {
		name      string
		history   [][]llm.Message
		to, told  string
		unsourced bool
	}{
		{"a number the agent posted, read back", [][]llm.Message{posted("I'll put you through to +1 (555) 010-0199 now."), read(`{"messages":"Bot: I'll put you through to +1 (555) 010-0199 now."}`)}, "+15550100199", "", true},
		{"a number the agent saved in a note, read back with a record", [][]llm.Message{noted, read("Ana | customer | notes: Call Ana back at +1 555 010 0199")}, "+15550100199", "", true},
		{"a number the agent said, listed by a tool", [][]llm.Message{said("I'll put you through to +1 (555) 010-0199 now."), read(`{"messages":"Bot: I'll put you through to +1 (555) 010-0199 now."}`)}, "+15550100199", "", false},
		{"a number the agent said to confirm it, listed", [][]llm.Message{{{Role: llm.User, Content: "Billing please, 555 010 0199"}}, said("Billing: 555-010-0199?"), {{Role: llm.User, Content: "yes"}}, departments("Billing: 555-010-0199\nSales: 555-010-0200")}, "+15550100199", "", false},
		{"a number the agent asked about, listed in other words", [][]llm.Message{said("So that's billing on 555 010 0199, right?"), departments(`{"departments":["So that's billing on 555 010 0199, right? Yes: Billing (Mon-Fri)"]}`)}, "+15550100199", "", false},
		{"a number the agent saved in a note, read back with the record's own", [][]llm.Message{noted, read("Ana | phone +1 555 010 0199 | notes: Call Ana back at +1 555 010 0199")}, "+15550100199", "", false},
		{"a number the agent saved in a note, read back with the record's own after it", [][]llm.Message{noted, read("Ana | notes: Call Ana back at +1 555 010 0199 | mobile (555) 010-0199")}, "+15550100199", "", false},
		{"a number the agent saved in a note, read back with a number running on from it", [][]llm.Message{noted, read("Ana | notes: Call Ana back at +1 555 010 0199 99")}, "+15550100199", "", true},
		{"a number the agent saved in a note, read back twice", [][]llm.Message{noted, read("Ana | notes: Call Ana back at +1 555 010 0199 | last note: Call Ana back at +1 555 010 0199")}, "+15550100199", "", true},
		{"a number the agent saved in a note, read back with digits either side of it", [][]llm.Message{noted, read("Ana | ext 1-555Call Ana back at +1 555 010 0199. 010-0199")}, "+15550100199", "", true},
		{"a number the agent posted, read back in another form", [][]llm.Message{posted("Call us at +15550100199"), read(`{"messages":[{"text":"Call us at +1 555 010 0199"}]}`)}, "+15550100199", "", true},
		{"a number the agent posted without its country code, read back", [][]llm.Message{posted("Sales is at (555) 010-0199"), read(`{"messages":[{"text":"Sales is at (555) 010-0199"}]}`)}, "+15550100199", "", true},
		{"a number the agent posted, read back as text, sent in another form", [][]llm.Message{posted("Sales: +15550100199"), read("[bot] Sales: +15550100199\n[ana] thanks")}, "+1 (555) 010-0199", "", true},
		{"a number the agent posted, read back on another line than a person's", [][]llm.Message{posted("I'll put you through to +1 (555) 010-0199 now."), read(`{"messages":"Bot: I'll put you through to +1 (555) 010-0199 now.\nAna: yes, sales is +1 555 010 0199"}`)}, "+15550100199", "", false},
		{"a number a tool returned before the agent posted it", [][]llm.Message{directory, posted("I'll put you through to +1 (555) 010-0199 now."), read(`{"messages":"Bot: I'll put you through to +1 (555) 010-0199 now."}`)}, "+15550100199", "", false},
		{"a number a tool returned, sent in another form", [][]llm.Message{directory}, "+15550100199", "", false},
		{"a number the operator gave", [][]llm.Message{posted("I'll put you through to +1 (555) 010-0199 now."), read(`{"messages":"Bot: I'll put you through to +1 (555) 010-0199 now."}`)}, "+15550100199", "Sales: (555) 010-0199", false},
		{"a number the agent sent alone, returned", [][]llm.Message{checked}, "+15550100199", "", false},
		{"a number the agent sent alone without its country code, returned", [][]llm.Message{checkedLocal}, "+1 (555) 010-0199", "", false},
		{"an extension on a line with an agent's line holding its last digits", [][]llm.Message{posted("Room 234 is ready."), read(`{"messages":"Bot: Room 234 is ready. Front desk is 1234"}`)}, "1234", "", false},
		{"a number the agent posted, read back with its digits split by a space and a line break", [][]llm.Message{posted("Put them through to 5550100199"), read("Put them through to 555 010 0199\nRef 555Put them through to 5550100199 0100199")}, "+15550100199", "", true},
	}
	for _, c := range cases {
		history := []llm.Message{{Role: llm.User, Content: "Can you put me through to sales?"}}
		for _, part := range c.history {
			history = append(history, part...)
		}
		arguments, err := json.Marshal(map[string]string{"to": c.to})
		require.NoError(t, err)
		_, unsourced := unsourcedTransfer(history, llm.ToolCall{Name: toolTransfer, Arguments: string(arguments)}, c.told)
		assert.Equal(t, c.unsourced, unsourced, c.name)
	}
}
