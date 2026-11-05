package agent

import (
	"encoding/json"
	"fmt"
	"html"
	"maps"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// unfoundID reports an ID argument whose value no tool has returned, under that key or as an
// "id", with the IDs the tools have returned so far. Models pass made-up values as IDs: a
// person's own identifier as a booking's, a time and a name as a booking ID, or an error
// code as a case that was never opened.
//
// Only an ID one of the tools hands out is checked: "case_id" when there is an open_case,
// "booking_id" when there is a find_booking. Any other ID, such as a card_id, is the
// caller's to give, and so is one the caller said along with what it is ("my case is
// 6631"), or said in answer to the agent asking for it ("What's your case number?" "6631"),
// or pasted inside a link, as a whole part of its path or a whole value of its query (linkParts):
// a link names what each of its parts is by where it stands. The error code "seven twenty", said where no case was mentioned, is not a case. An ID
// written in told, the instructions the operator gave the agent, is the operator's to give.
func unfoundID(history []llm.Message, call llm.ToolCall, tools []string, told string) (key, value string, known []string, found bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", nil, false
	}
	var said strings.Builder
	var results []any
	// answers are what the caller said straight after the agent asked them something, with
	// the question.
	type answer struct{ question, said string }
	var answers []answer
	// pasted are the parts of the links the caller wrote.
	var pasted []string
	// sent are the arguments of each call, by its ID, their escapes undone: a result that
	// writes one of them back echoes the model, it does not return it, however the model
	// escaped it ("\u0043" for "C").
	sent := map[string]string{}
	// called are the names of the tools called, by the call's ID.
	called := map[string]string{}
	// wrote are the lines the agent wrote, in its replies and in the arguments of its calls
	// (sent), each alphanumeric. A tool that reads back a channel or a thread returns them too.
	var wrote []string
	write := func(text string) {
		for _, own := range strings.Split(text, "\n") {
			wrote = append(wrote, alphanumeric(own))
		}
	}
	// lines are the lines a JSON result writes out for people, with the arguments of the
	// call it answers and how many lines the agent had written by then.
	type line struct {
		text, sent string
		wrote      int
	}
	var lines []line
	// texts are results that are not JSON objects or lists: a lookup may answer with the
	// bare ID, such as jane_doe_17, under no key to look it up by.
	var texts []line
	// linked are the parts of the links the tools returned (linkParts), but for the parts the
	// call each answers was sent: a link that holds them writes back what the model asked for.
	// Each comes with its line and the words that may name its kind: its line's, those of the
	// keys it stands under and the tool's name.
	type toolLink struct {
		part  string
		in    line
		named []string
	}
	var linked []toolLink
	var question string
	for _, message := range history {
		switch message.Role {
		case llm.Assistant:
			if strings.TrimSpace(message.Content) != "" {
				question = message.Content
			}
			write(message.Content)
			for _, made := range message.ToolCalls {
				sent[made.ID] = asWritten(made.Arguments)
				called[made.ID] = made.Name
				write(sent[made.ID])
			}
		case llm.User:
			heard := message.Content + " " + withSpokenSeparators(digitsFor(message.Content))
			said.WriteString(" " + heard)
			// A link is read as the person wrote it: a shared channel passes their text on in
			// a JSON envelope, which writes & as \u0026, and a chat may pass it on escaped for
			// HTML, & as &amp;. Either way a link would end, or its query split, at the escape.
			pasted = append(pasted, linkParts(html.UnescapeString(asWritten(message.Content)))...)
			if question != "" {
				answers = append(answers, answer{strings.ToLower(question), alphanumeric(heard)})
			}
			question = ""
		case llm.ToolResult:
			if failedResult(message.Content) {
				// A refusal quotes the value it refused, which does not make it returned.
				continue
			}
			// A link a tool returns names what each of its parts is by where it stands, as the
			// person's do, and is read the same way: "Permalink: <link to the channel>" gives
			// the channel. What kind of thing the part is, only the words around the link say:
			// a person's profile link a search for channels returns gives no channel.
			writtenFields(message.Content, func(under []string, value string) {
				for _, text := range strings.Split(html.UnescapeString(value), "\n") {
					for _, part := range linkParts(text) {
						if !strings.Contains(alphanumeric(sent[message.ToolCallID]), part) {
							named := slices.Concat(lineWords(text), lineWords(strings.Join(under, " ")), lineWords(called[message.ToolCallID]))
							linked = append(linked, toolLink{part, line{text, sent[message.ToolCallID], len(wrote)}, named})
						}
					}
				}
			})
			var result any
			if json.Unmarshal([]byte(message.Content), &result) != nil {
				texts = append(texts, line{message.Content, "", len(wrote)})
			} else if text, bare := result.(string); bare {
				texts = append(texts, line{text, "", len(wrote)})
			} else {
				results = append(results, result)
				// A result may write its answer out under a key of its own: a list of channels
				// as "1. #general (C0123) - public_channel".
				for _, text := range prose(result) {
					for _, written := range strings.Split(text, "\n") {
						lines = append(lines, line{written, sent[message.ToolCallID], len(wrote)})
					}
				}
			}
		}
	}
	heard := alphanumeric(said.String())
	keys := make([]string, 0, len(arguments))
	for name := range arguments {
		keys = append(keys, name)
	}
	slices.Sort(keys)
	for _, name := range keys {
		text, ok := arguments[name].(string)
		wanted := alphanumeric(text)
		if !ok || !handedOut(name, tools) {
			continue
		}
		if wanted == "" {
			return name, text, collectIDs(results), true
		}
		// A line a tool returned that holds a line the agent wrote before it, with the ID in it,
		// is the agent's own text read back: a channel's history holds what the agent posted
		// there, and what the agent wrote is no source of an ID. A line of the agent's that is
		// the ID alone, an argument's value, is not counted: every line that gives the ID holds it.
		agents := func(returned line) bool {
			text := alphanumeric(html.UnescapeString(returned.text))
			return slices.ContainsFunc(wrote[:returned.wrote], func(own string) bool {
				return own != wanted && strings.Contains(own, wanted) && strings.Contains(text, own)
			})
		}
		if strings.Contains(heard, wanted) && holdsKind(plainWords(said.String()), entityOf(name), strings.Contains) || slices.Contains(pasted, wanted) ||
			slices.ContainsFunc(linked, func(link toolLink) bool {
				return link.part == wanted && holdsKind(link.named, entityOf(name), strings.Contains) && !agents(link.in)
			}) {
			continue
		}
		if slices.ContainsFunc(answers, func(given answer) bool {
			return holdsKind(plainWords(given.question), entityOf(name), strings.Contains) && strings.Contains(given.said, wanted)
		}) || holdsValue(told, wanted) {
			continue
		}
		returned := false
		var ids []string
		for _, result := range results {
			walkIDs(result, "", func(at, id string) {
				if (at == name || at == "id") && alphanumeric(id) == wanted {
					returned = true
				}
				if entry := fmt.Sprintf("%s (%s)", id, at); !slices.Contains(ids, entry) {
					ids = append(ids, entry)
				}
			})
		}
		// A line written out gives an ID only beside the word for what it is (besideKind), and
		// only one the call it answers was not sent: "No bookings found for Ana at 7pm" does
		// not return the 7pm the model asked about, and "Card KTR583027 on file" returns no case.
		// An ID is the ID of one thing, so no line reads a label as giving it where another
		// line labels it as another kind (labelledAs): after "From: Ana Lind (ID: U0…)",
		// "Channel: Ana Lind (ID: U0…)" names the sender, not a channel. Only a label is
		// doubted: a line beside the kind still gives it, and a line the call was sent the ID in
		// labels nothing. "Channel: Group DM (ID: C0…)" still gives one, where no line labels
		// C0… as anything else.
		other := slices.ContainsFunc(lines, func(written line) bool {
			return !strings.Contains(alphanumeric(written.sent), wanted) && slices.ContainsFunc(labelledAs(written.text, wanted), func(label []string) bool { return !labelNames(label, entityOf(name)) })
		})
		returned = returned || slices.ContainsFunc(lines, func(written line) bool {
			return besideKind(written.text, wanted, entityOf(name), !other) && !strings.Contains(alphanumeric(written.sent), wanted) && !agents(written)
		})
		if !returned && !slices.ContainsFunc(texts, func(result line) bool {
			return slices.ContainsFunc(strings.Split(result.text, "\n"), func(text string) bool {
				return holdsValue(text, wanted) && !agents(line{text, "", result.wrote})
			})
		}) {
			return name, text, ids, true
		}
	}
	return "", "", nil, false
}

// placeholderName reports a name argument filled with a stand-in for one the caller never
// gave, such as "Caller" or "John Doe", or with words of a note the router wrote into the
// conversation itself (fromNote), which is nobody's name.
func placeholderName(history []llm.Message, call llm.ToolCall) (string, bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", false
	}
	for key, value := range arguments {
		text, ok := value.(string)
		if !ok || (key != "name" && !strings.HasSuffix(key, "_name")) {
			continue
		}
		if (slices.Contains(placeholders, strings.ToLower(strings.TrimSpace(text))) || fromNote(text)) &&
			!nameWasGiven(history, text) {
			return text, true
		}
	}
	return "", false
}

// formatVerb is a fmt verb, where a note written as a format string has words of its own.
var formatVerb = regexp.MustCompile(`%[-+# 0-9.]*[a-zA-Z]`)

// noteSentence ends a sentence of a note.
var noteSentence = regexp.MustCompile(`[.?!;:]`)

// fromNote reports whether text was taken from a note the router writes into the
// conversation itself, the harness's (harness.Notes) and the agent's own (composeNote,
// approvalSilence): it is three or more words lying wholly inside one of them, or it holds one
// of their sentences whole. A name that only shares a few words with a note, "On With The
// Show" with "Carry on with the caller", is not taken from it.
func fromNote(text string) bool {
	words := func(text string) string {
		return " " + strings.Join(strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}), " ") + " "
	}
	said := words(text)
	for _, note := range append(harness.Notes(), composeNote, approvalSilence) {
		for _, part := range formatVerb.Split(note, -1) {
			if len(strings.Fields(said)) >= 3 && strings.Contains(words(part), said) {
				return true
			}
			for _, sentence := range noteSentence.Split(part, -1) {
				if sentence := words(sentence); len(strings.Fields(sentence)) >= 3 && strings.Contains(said, sentence) {
					return true
				}
			}
		}
	}
	return false
}

// A name field can identify a place or business, not just a person. The lexical
// heuristics must not reject a name the caller or a successful tool already gave.
func nameWasGiven(history []llm.Message, name string) bool {
	words := func(text string) string {
		return " " + strings.ToLower(strings.Join(strings.FieldsFunc(text, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}), " ")) + " "
	}
	wanted := words(name)
	if wanted == "  " {
		return false
	}
	var contains func(any) bool
	contains = func(value any) bool {
		switch value := value.(type) {
		case string:
			return strings.Contains(words(value), wanted)
		case []any:
			return slices.ContainsFunc(value, contains)
		case map[string]any:
			for _, child := range value {
				if contains(child) {
					return true
				}
			}
		}
		return false
	}
	for _, message := range history {
		switch message.Role {
		case llm.User:
			if contains(message.Content) {
				return true
			}
		case llm.ToolResult:
			if failedResult(message.Content) {
				continue
			}
			var result any
			if json.Unmarshal([]byte(message.Content), &result) != nil {
				result = message.Content
			}
			if contains(result) {
				return true
			}
		}
	}
	return false
}

// unheardNumber reports an argument of four or more digits that neither the caller, a tool
// nor the operator's instructions (told) ever gave, and that is a near miss of a number the
// caller said (nearMiss): digits reordered from what the caller said ("six three zero five"
// sent as 5630), or a phone number with a digit dropped. A number that is no near miss is the
// model's to work out, such as a year from "next year", and is let through.
func unheardNumber(history []llm.Message, call llm.ToolCall, told string) (key, value string, heard []string, unheard bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", nil, false
	}
	sources := []string{told}
	// spoken are the numbers the caller said, each as its digits alone.
	var spoken []string
	for _, message := range history {
		switch message.Role {
		case llm.User:
			sources = append(sources, message.Content, withSpokenSeparators(digitsFor(message.Content)))
			for _, number := range digitRun.FindAllString(digitsFor(message.Content), -1) {
				if len(number) >= 3 && !slices.Contains(heard, number) {
					heard = append(heard, number)
				}
			}
			spoken = append(spoken, digitGroups(message.Content, withSpokenSeparators(digitsFor(message.Content)))...)
		case llm.ToolResult:
			// A refusal quotes the value refused: "56305" refused let 5630 through.
			if !failedResult(message.Content) {
				sources = append(sources, message.Content)
			}
		}
	}
	spoken = append(spoken, heard...)
	digits := digitGroups(sources...)
	keys := make([]string, 0, len(arguments))
	for name := range arguments {
		keys = append(keys, name)
	}
	slices.Sort(keys)
	for _, name := range keys {
		text, ok := arguments[name].(string)
		if !ok || len(text) < 4 || !allDigits(text) {
			continue
		}
		if slices.ContainsFunc(digits, func(said string) bool { return strings.Contains(said, text) }) {
			continue
		}
		if slices.ContainsFunc(spoken, func(said string) bool { return nearMiss(text, said) }) {
			return name, text, heard, true
		}
	}
	return "", "", nil, false
}

// unsourcedTransfer reports the number of a transfer that neither the operator's instructions
// (told) nor a successful tool gave. Unlike unheardNumber it does not count the caller as a
// source: anyone can read out a number and be put through to it.
func unsourcedTransfer(history []llm.Message, call llm.ToolCall, told string) (to string, unsourced bool) {
	var arguments transferArguments
	if call.Name != toolTransfer || json.Unmarshal([]byte(call.Arguments), &arguments) != nil || strings.TrimSpace(arguments.To) == "" {
		return "", false
	}
	sources := []string{told}
	for _, message := range history {
		if message.Role == llm.ToolResult && !failedResult(message.Content) {
			sources = append(sources, message.Content)
		}
	}
	wanted := strings.Join(digitGroups(arguments.To), "")
	for _, source := range sources {
		for _, run := range writtenNumber.FindAllString(source, -1) {
			groups := digitGroups(run)
			spans := numberGroup.FindAllStringIndex(run, -1)
			// Digits run on from a number unless a mark sets them off: before it only a closing
			// bracket or a full stop ("1. 555", "+44 (555)" is not one), after it also an opening
			// bracket, as in "1111 (9am" or "1111. 24/7".
			before := func(i int) bool {
				if i == 0 {
					return true
				}
				gap := strings.TrimSpace(run[spans[i-1][1]:spans[i][0]])
				return gap != "" && strings.Trim(gap, ").") == ""
			}
			after := func(i int) bool {
				return i == len(groups) || strings.ContainsAny(run[spans[i-1][1]:spans[i][0]], "().")
			}
			// Digits beside the number, "(9am-5pm)" or "24/7", are not part of it.
			for start := range groups {
				for end := start + 1; end <= len(groups); end++ {
					digits := strings.Join(groups[start:end], "")
					if digits == wanted && before(start) && after(end) {
						return "", false
					}
					// A US number written without its country code is the same line, unless
					// digits run on from it unmarked, as a country code or a second number do.
					if len(digits) == 10 && "1"+digits == wanted &&
						before(start) && after(end) {
						return "", false
					}
				}
			}
		}
	}
	return arguments.To, true
}

// unsourcedTransferResult is what a transfer to a number only the caller gave is answered with.
func unsourcedTransferResult(to string) string {
	return fmt.Sprintf(didNotWork+"%q was not given by the operator or a tool, so it is not a number "+
		"to transfer to. Do not transfer. Tell the caller you cannot put them through to that number.", to)
}

// digitGroups returns every number written in texts (numberGroup) as its digits alone.
func digitGroups(texts ...string) []string {
	var digits []string
	for _, text := range texts {
		for _, group := range numberGroup.FindAllString(text, -1) {
			digits = append(digits, strings.Map(func(r rune) rune {
				if unicode.IsDigit(r) {
					return r
				}
				return -1
			}, group))
		}
	}
	return digits
}

// nearMiss reports whether value is a number the caller said, written wrongly in the way a
// number is misheard or miscopied: the same digits in another order, or one digit dropped. A
// single digit changed is not counted: it is also what a number worked out from one said
// looks like, such as next year's from this year's.
//
// Numbers said one after another can run together in the transcript ("six three zero five,
// fourteen" is written 630514), so the number may also be the start or the end of what the
// caller said.
func nearMiss(value, said string) bool {
	for _, length := range []int{len(value), len(value) + 1} {
		if length > len(said) {
			continue
		}
		for _, part := range []string{said[:length], said[len(said)-length:]} {
			if part == value {
				continue
			}
			if length == len(value) {
				a, b := []byte(value), []byte(part)
				slices.Sort(a)
				slices.Sort(b)
				if string(a) == string(b) {
					return true
				}
				continue
			}
			for at := range part {
				if part[:at]+part[at+1:] == value {
					return true
				}
			}
		}
	}
	return false
}

var digitRun = regexp.MustCompile(`\d+`)

// numberGroup is one number as written, its dashes and brackets included: "512-555-0178".
// Numbers apart are kept apart, so "2745, last four 6305" holds no 5630.
var numberGroup = regexp.MustCompile(`\d(?:[\d().-]*\d)?`)

// writtenNumber is a number as people write it, its spaces, dots, dashes and brackets
// included: "+1 (555) 000-1111". A line break ends it, and so do more than two separators in a row.
var writtenNumber = regexp.MustCompile(`\d(?:[ ().-]{0,2}\d)*`)

func allDigits(text string) bool {
	return strings.IndexFunc(text, func(r rune) bool { return !unicode.IsDigit(r) }) < 0
}

// unheardResult is what a call with a number nobody gave is answered with.
func unheardResult(key, value string, heard []string) string {
	result := fmt.Sprintf(didNotWork+"the caller never said %q, and no tool returned it, so it cannot be "+
		"the %s. ", value, key)
	if len(heard) > 0 {
		result += "The numbers the caller said are: " + strings.Join(heard, ", ") + ". "
	}
	return result + "Call again with the one they gave, digit for digit, or read back what you heard and ask."
}

// misspelledName reports a name argument holding the word the transcriber wrote beside a
// spelling the caller gave, where that word is not what they spelled. Suppose a caller says
// "The name is Odenor, o d e n o r" and the transcriber writes "The name is Aldenor, o d e n
// o r": the letters are what the caller said, and Aldenor is the transcriber's guess at the
// name.
//
// Only that word is refused, and only when it is close to the spelling (closeTo): another
// name the caller gives, however like the spelled one, is theirs. A name the caller spelled
// exactly at some point in the call is let through even where it is also the word beside
// another spelling: the transcriber hears the same letters differently from one spelling to
// the next ("n o a h", then "n o a a").
func misspelledName(history []llm.Message, call llm.ToolCall) (name, spelled string, misspelled bool) {
	var arguments map[string]any
	if json.Unmarshal([]byte(call.Arguments), &arguments) != nil {
		return "", "", false
	}
	var spellings []string
	// misheard maps the word written beside a spelling to the spelling.
	misheard := map[string]string{}
	for _, message := range history {
		if message.Role != llm.User {
			continue
		}
		spellings = append(spellings, spelledNames(message.Content)...)
		for word, spelling := range writtenBeside(message.Content) {
			misheard[word] = spelling
		}
	}
	if len(misheard) == 0 {
		return "", "", false
	}
	keys := make([]string, 0, len(arguments))
	for key := range arguments {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		text, ok := arguments[key].(string)
		if !ok || (key != "name" && !strings.HasSuffix(key, "_name")) {
			continue
		}
		for _, word := range strings.Fields(text) {
			word = alphanumeric(word)
			if spelling, beside := misheard[word]; beside && !slices.Contains(spellings, word) {
				return text, spelling, true
			}
		}
	}
	return "", "", false
}

// writtenBeside maps the word written right before or right after each spelling in a message
// to the spelling, where that word is close to it (closeTo) and not the same.
func writtenBeside(text string) map[string]string {
	runes := []rune(strings.ToLower(text))
	isLetter := func(at int) bool { return at >= 0 && at < len(runes) && unicode.IsLetter(runes[at]) }
	// before is the word that ends at or before at, and after the word that starts at or after it.
	before := func(at int) string {
		for at >= 0 && !isLetter(at) {
			at--
		}
		end := at + 1
		for isLetter(at) {
			at--
		}
		return string(runes[at+1 : end])
	}
	after := func(at int) string {
		for at < len(runes) && !isLetter(at) {
			at++
		}
		start := at
		for isLetter(at) {
			at++
		}
		return string(runes[start:at])
	}
	beside := map[string]string{}
	for _, run := range spelledRuns(runes) {
		for _, word := range []string{before(run.start - 1), after(run.end)} {
			if len(word) >= 3 && word != run.letters && closeTo(word, run.letters) {
				beside[word] = run.letters
			}
		}
	}
	return beside
}

// spelledNames are the names a caller spelled out in one message, lowercase: three or more
// single letters read out one at a time, "o d e n o r", "K-O-V-A-C", "J. O. H. N.".
//
// A spelling's letters are joined by one kind of separator, spaces, commas, hyphens or
// periods, and a change of separator ends it: in "k l e i n. I need" the period ends the
// sentence, and read as one more letter the "I" made Klein "kleini". A letter only
// counts when it stands alone: in "that's a l m e i d a" the s ends a word, and read as a
// spelled letter it made Almeida "Salmeida".
//
// A caller saying a name and then spelling it with the next one in a single breath ("Kay
// Jensen. That's k a i j e n s e n") gives the transcriber no pause to write. A spelling
// that opens or closes with a name said in the same message is also read as the two
// names, so Kai and Jensen are each spelled as well as kaijensen.
func spelledNames(text string) []string {
	runes := []rune(strings.ToLower(text))
	var runs []string
	for _, run := range spelledRuns(runes) {
		runs = append(runs, run.letters)
	}
	if len(runs) == 0 {
		return nil
	}
	said := strings.FieldsFunc(string(runes), func(r rune) bool { return !unicode.IsLetter(r) })
	spellings := slices.Clone(runs)
	for _, run := range runs {
		for _, word := range said {
			if len(word) < 3 || len(run)-len(word) < 2 {
				continue
			}
			if rest, opens := strings.CutPrefix(run, word); opens {
				spellings = append(spellings, word, rest)
			}
			if rest, closes := strings.CutSuffix(run, word); closes {
				spellings = append(spellings, rest, word)
			}
		}
	}
	return spellings
}

// spelledRun is one spelling in a message: its letters, and where in the message's runes it
// starts and ends.
type spelledRun struct {
	letters    string
	start, end int
}

// spelledRuns finds the spellings in a lowercased message, as spelledNames describes them.
func spelledRuns(runes []rune) []spelledRun {
	alone := func(at int) bool {
		return at < len(runes) && unicode.IsLetter(runes[at]) &&
			(at == 0 || !unicode.IsLetter(runes[at-1]) && runes[at-1] != '\'' && runes[at-1] != '’') &&
			(at+1 == len(runes) || !unicode.IsLetter(runes[at+1]))
	}
	var runs []spelledRun
	for at := 0; at < len(runes); at++ {
		if !alone(at) {
			continue
		}
		start := at
		letters := []rune{runes[at]}
		kind := rune(0)
		for {
			next := at + 1
			for next < len(runes) && strings.ContainsRune(" \t\n,.-", runes[next]) {
				next++
			}
			if next == at+1 || !alone(next) {
				break
			}
			separator := separatorKind(runes[at+1 : next])
			if kind == 0 {
				kind = separator
			} else if separator != kind {
				break
			}
			letters = append(letters, runes[next])
			at = next
		}
		if len(letters) >= 3 {
			runs = append(runs, spelledRun{string(letters), start, at + 1})
		}
	}
	return runs
}

// separatorKind names what joins two spelled letters: a comma, a period or a hyphen when
// there is one, and a space when there is only space.
func separatorKind(between []rune) rune {
	for _, kind := range []rune{',', '.', '-'} {
		if slices.Contains(between, kind) {
			return kind
		}
	}
	return ' '
}

// closeTo reports whether two words are near enough to be one name heard two ways: the same
// first or last two letters, and lengths within three.
func closeTo(a, b string) bool {
	if d := len(a) - len(b); d > 3 || d < -3 || len(a) < 2 || len(b) < 2 {
		return false
	}
	return a[len(a)-2:] == b[len(b)-2:] || a[:2] == b[:2]
}

// misspelledResult is what a call with a misspelled name is answered with.
func misspelledResult(name, spelled string) string {
	return fmt.Sprintf(didNotWork+"the caller spelled the name out as %s, so it is %s, not %q. Call "+
		"again with it.", strings.Join(strings.Split(spelled, ""), " "),
		strings.ToUpper(spelled[:1])+spelled[1:], name)
}

var placeholders = []string{"caller", "the caller", "customer", "guest", "user", "john doe", "jane doe", "unknown", "n/a"}

// placeholderResult is what a call with a stand-in name is answered with.
func placeholderResult(name string) string {
	return fmt.Sprintf(didNotWork+"%q is not a name the caller gave. Call again with the name they "+
		"gave, or ask them for it.", name)
}

func toolNames(tools harness.Tools) []string {
	names := make([]string, 0, len(tools.Tools))
	for _, tool := range tools.Tools {
		names = append(names, tool.Name)
	}
	return names
}

func collectIDs(results []any) []string {
	var ids []string
	for _, result := range results {
		walkIDs(result, "", func(at, id string) {
			if entry := fmt.Sprintf("%s (%s)", id, at); !slices.Contains(ids, entry) {
				ids = append(ids, entry)
			}
		})
	}
	return ids
}

// listedID is the ID key a list under key holds: "order_id" for "orders" or "order_ids",
// and "" for a key that is not plural.
func listedID(key string) string {
	if entity, ok := strings.CutSuffix(key, "_ids"); ok && entity != "" {
		return entity + "_id"
	}
	if entity, ok := strings.CutSuffix(key, "s"); ok && entity != "" && !strings.HasSuffix(entity, "s") && !strings.HasSuffix(entity, "_id") {
		return entity + "_id"
	}
	return ""
}

// entityOf is what an ID argument names: "booking" for "booking_id".
func entityOf(key string) string {
	return strings.ReplaceAll(strings.TrimSuffix(key, "_id"), "_", " ")
}

// handedOut reports whether key is the ID of something one of the tools makes or looks up.
func handedOut(key string, tools []string) bool {
	if !strings.HasSuffix(key, "_id") {
		return false
	}
	entity := strings.TrimSuffix(key, "_id")
	return slices.ContainsFunc(tools, func(tool string) bool { return tool != "" && strings.Contains(tool, "_"+entity) })
}

// walkIDs calls found with every string or number held under "id" or a key ending in "_id";
// with every one listed under a plural key as the ID of what that key names: a profile that
// lists "orders": ["A-1001"] gives order_id A-1001; and with every key of a map of records
// under a plural key, which is keyed by the records' IDs: "vehicles": {"van-2": {...}} gives
// vehicle_id van-2, though no field of the vehicle repeats it.
func walkIDs(value any, at string, found func(at, id string)) {
	switch typed := value.(type) {
	case map[string]any:
		if keyed := listedID(at); keyed != "" && keyedByID(typed) {
			for name := range typed {
				found(keyed, name)
			}
		}
		for name, inner := range typed {
			walkIDs(inner, name, found)
		}
	case []any:
		if listed := listedID(at); listed != "" {
			at = listed
		}
		for _, inner := range typed {
			walkIDs(inner, at, found)
		}
	case string:
		if at == "id" || strings.HasSuffix(at, "_id") {
			found(at, typed)
		}
	case float64:
		if at == "id" || strings.HasSuffix(at, "_id") {
			found(at, fmt.Sprint(typed))
		}
	}
}

// keyedByID reports whether a map is a collection of records keyed by their IDs: every value
// is a record, an object that is not itself such a collection. A record's own fields hold
// values of every kind, so this is never true of one record; and a wrapper such as
// {"fleet": {"vehicles": {...}}} holds a collection rather than records, so its key is not
// an ID.
func keyedByID(collection map[string]any) bool {
	if len(collection) == 0 {
		return false
	}
	for _, value := range collection {
		record, isObject := value.(map[string]any)
		if !isObject || allObjects(record) {
			return false
		}
	}
	return true
}

// allObjects reports whether every value of a non-empty map is an object.
func allObjects(fields map[string]any) bool {
	if len(fields) == 0 {
		return false
	}
	for _, value := range fields {
		if _, isObject := value.(map[string]any); !isObject {
			return false
		}
	}
	return true
}

// link is a web address written in a message. It ends at the first character a URI cannot
// hold (RFC 3986 section 2 allows none of space, <, >, ", {, }, |, \, ^ and `), so a link
// written in brackets or quotes ends at them.
var link = regexp.MustCompile("(?i)https?://[^\\s<>\"{}|\\\\^`]+")

// linkParts returns every path segment and query value of the links in text, each
// alphanumeric. Only whole parts are given: the 1793 inside a segment p1793 is not one.
//
// Only a part that mixes letters and digits is given. A number alone is as often a page, a
// year or a size as an ID, and letters alone are a word, a name or a section: neither is an
// ID of a kind the link leaves unnamed. A number the person means as an ID still counts
// where what they wrote names its kind, as "…/cases/48213" does.
func linkParts(text string) []string {
	var parts []string
	add := func(part string) {
		if part = alphanumeric(part); strings.ContainsFunc(part, unicode.IsLetter) && strings.ContainsFunc(part, unicode.IsDigit) {
			parts = append(parts, part)
		}
	}
	for _, written := range link.FindAllString(text, -1) {
		address, err := url.Parse(written)
		if err != nil {
			continue
		}
		// Each segment is unescaped after the split, so an escaped "/" stays inside its segment.
		for _, escaped := range strings.Split(address.EscapedPath(), "/") {
			segment, err := url.PathUnescape(escaped)
			if err != nil {
				continue
			}
			add(segment)
		}
		for _, values := range address.Query() {
			for _, value := range values {
				add(value)
			}
		}
	}
	return parts
}

// asWritten is text with the escapes of JSON undone: the keys and values of a JSON value, one
// per line, each number as it was written. Text that is not JSON is returned as it is.
func asWritten(text string) string {
	var written []string
	writtenFields(text, func(_ []string, value string) { written = append(written, value) })
	return strings.Join(written, "\n")
}

// writtenFields calls visit with each key and value of a JSON text, as asWritten writes them,
// and the keys it stands under. Text that is not JSON is one value under no key.
func writtenFields(text string, visit func(under []string, value string)) {
	var value any
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if !json.Valid([]byte(text)) || decoder.Decode(&value) != nil {
		visit(nil, text)
		return
	}
	var walk func([]string, any)
	walk = func(under []string, value any) {
		switch typed := value.(type) {
		case string:
			visit(under, typed)
		case json.Number:
			visit(under, typed.String())
		case []any:
			for _, inner := range typed {
				walk(under, inner)
			}
		case map[string]any:
			for _, key := range slices.Sorted(maps.Keys(typed)) {
				visit(under, key)
				walk(append(under, key), typed[key])
			}
		}
	}
	walk(nil, value)
}

// prose returns the strings of more than one word a JSON result holds, wherever they are. A
// string of one word under a key that is not an ID's is left out, so that a card listed
// under "cards" does not pass as an order.
func prose(value any) []string {
	switch typed := value.(type) {
	case string:
		if len(strings.Fields(typed)) > 1 {
			return []string{typed}
		}
	case []any:
		var texts []string
		for _, inner := range typed {
			texts = append(texts, prose(inner)...)
		}
		return texts
	case map[string]any:
		var texts []string
		for _, inner := range typed {
			texts = append(texts, prose(inner)...)
		}
		return texts
	}
	return nil
}

// holdsValue reports whether a result written as text gives wanted as one of its words, as
// a bare ID answered alone does. wanted is already alphanumeric.
func holdsValue(result, wanted string) bool {
	return slices.ContainsFunc(idWords(result), func(word string) bool { return alphanumeric(word) == wanted })
}

// idWords are the words of text an ID can be, its _, - and # kept: "T-6631" is one word.
func idWords(text string) []string {
	return idWord.FindAllString(text, -1)
}

// idWord is one of idWords.
var idWord = regexp.MustCompile(`[\p{L}\p{Nd}_#-]+`)

// besideKind reports whether a line written for people gives wanted as the ID of kind. A line
// names a thing by its kind, its name and its ID side by side: "Channel: #help (ID: C0…)",
// "#help (C0…) - public_channel". So between the word for the kind and the ID stands at most
// one word, the thing's name, besides the word "ID". Any more ties the ID to another word: in
// "Channel #x by U0…", U0… is whoever made the channel. A label ends that doubt: in
// "Channel: Group DM (ID: C0…)" the label "Channel:" opens a field whose value is the name
// and the ID in brackets, so a name of any length stands between them (labelledAs). That
// reading holds only where labelled is true: no other line labels the ID as another kind.
//
// The word for the kind is a word of its own (holdsKind), and not the ID's: "#ops-channel" is
// a channel's name, not the word "channel", and "public_channel" listed as a channel's type
// is no channel's ID.
func besideKind(line, wanted, kind string, labelled bool) bool {
	if labelled && slices.ContainsFunc(labelledAs(line, wanted), func(label []string) bool { return labelNames(label, kind) }) {
		return true
	}
	words := lineWords(line)
	ends := phraseEnds(line)
	// A kind may take more than one word: "phone number" for phone_number_id.
	size := len(strings.Fields(kind))
	for at, word := range words {
		if alphanumeric(word) != wanted {
			continue
		}
		for start := 0; start+size <= len(words); start++ {
			if start <= at && at < start+size || !holdsKind(words[start:start+size], kind, strings.Contains) {
				continue
			}
			var between []string
			limit := 1
			if at < start {
				// An ID before the kind is the one the line lists, "#help (C0…) - public_channel":
				// nothing stands between them. Any word does, as in "U0… joined channel #help",
				// where U0… joined the channel. And the kind's word ends its phrase: a word after
				// it is what the phrase names, so in "Ana Lind (U0…) - channel owner" and
				// "public_channel_owner" U0… is an owner.
				if !ends[start+size-1] || !holdsKind(words[start:start+size], kind, strings.HasSuffix) {
					continue
				}
				between, limit = words[at+1:start], 0
			} else if at >= start+size {
				between = words[start+size : at]
			}
			if len(slices.DeleteFunc(slices.Clone(between), func(word string) bool { return alphanumeric(word) == "id" })) <= limit {
				return true
			}
		}
	}
	return false
}

// lineWords are the words of a line, lowercase (idWords), but for a mark alone, the dash in
// "#help - C0…", which is no word.
func lineWords(line string) []string {
	var words []string
	for _, word := range idWords(line) {
		if alphanumeric(word) != "" {
			words = append(words, strings.ToLower(word))
		}
	}
	return words
}

// phraseEnds reports for each of lineWords(line) whether it ends its phrase: the line ends
// after it, or a mark stands between it and the next word, as the dash does in
// "public_channel - Creator", or a tab or a gap of two spaces or more stands there, as in the
// columns of a text table. A word followed by one space alone does not: "channel owner".
func phraseEnds(line string) []bool {
	var spans [][]int
	for _, span := range idWord.FindAllStringIndex(line, -1) {
		if alphanumeric(line[span[0]:span[1]]) != "" {
			spans = append(spans, span)
		}
	}
	ends := make([]bool, len(spans))
	for i, span := range spans {
		if i == len(spans)-1 {
			ends[i] = true
			continue
		}
		gap := line[span[1]:spans[i+1][0]]
		ends[i] = strings.TrimSpace(gap) != "" || strings.ContainsRune(gap, '\t') || strings.Contains(gap, "  ")
	}
	return ends
}

// plainWords are the words of text the person or the model wrote, lowercase, split at every
// mark: "case#6631", "ticket-4521" and "case-number" each hold the word for the kind.
func plainWords(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// holdsKind reports whether words hold the word for kind, or its plural: find (strings.Contains
// or strings.HasSuffix) looks for it in the words joined with a space on each side. A
// key is split into its words where _ joins them: "public_channel" and "channel_id" each hold
// "channel". A name marked with #, "#team_channel", is one word, as it is when - joins it,
// "#ops-channel".
func holdsKind(words []string, kind string, find func(text, word string) bool) bool {
	split := make([]string, len(words))
	for i, word := range words {
		split[i] = word
		if !strings.HasPrefix(word, "#") {
			split[i] = strings.ReplaceAll(word, "_", " ")
		}
	}
	joined := " " + strings.Join(split, " ") + " "
	return find(joined, " "+kind+" ") || find(joined, " "+kind+"s ")
}

// labelNames reports whether a field's label names kind: its last words are the word for it,
// as in "Phone number:" or "1. Channel:". A label is named by its last word, so "Channel
// owner:" labels an owner.
func labelNames(label []string, kind string) bool {
	return holdsKind(label, kind, strings.HasSuffix)
}

// bracketedID is an ID in brackets, with or without the word "ID": "(C0…)", "(ID: C0…)".
var bracketedID = regexp.MustCompile(`\((?:(?i:id):\s*)?([\w-]+)\)`)

// labelledAs returns the label of each field "<label>: <name> (<wanted>)" on line, as its words:
// ["channel"] for "Channel: Group DM (ID: C0…)". The field ends at a " - " or another label
// or bracket, so "Creator: Ana Lind (<@U0…)" and "Channel #x by U0…" have none.
func labelledAs(line, wanted string) [][]string {
	var labels [][]string
	for _, found := range bracketedID.FindAllStringSubmatchIndex(line, -1) {
		if alphanumeric(line[found[2]:found[3]]) != wanted {
			continue
		}
		field := line[:found[0]]
		if cut := strings.LastIndexAny(field, "()"); cut >= 0 {
			field = field[cut+1:]
		}
		if cut := strings.LastIndex(field, " - "); cut >= 0 {
			field = field[cut+3:]
		}
		if colon := strings.LastIndex(field, ":"); colon >= 0 {
			labels = append(labels, lineWords(field[:colon]))
		}
	}
	return labels
}

// alphanumeric lowercases text and drops everything but letters and digits, so "T-6631"
// and "t6631" match.
func alphanumeric(text string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, text)
}

// unfoundIDResult is what a call with an ID nothing gave the model is answered with. It opens
// with didNotWork so that it backs no claim.
func unfoundIDResult(key, value string, known []string) string {
	result := fmt.Sprintf(didNotWork+"no tool has given you %q as a %s, and the caller did not say it. ", value, key)
	if strings.TrimSpace(value) == "" {
		result = fmt.Sprintf(didNotWork+"%s is empty. ", key)
	}
	if len(known) > 0 {
		result += "The IDs your tools have returned are: " + strings.Join(known, ", ") + ". "
	}
	return result + "Call the tool that gives you the right one first, or use one of those."
}
