package agent

import (
	"slices"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// learnTerms tells the transcriber to expect the names the caller has said and the agent has
// just said back. A transcriber that heard a name right once can still hear it as a common
// word the next time, so a name confirmed in the conversation is given to it as a keyterm.
// Only a name both sides have said is learned, so a name the agent got wrong is never
// reinforced.
func (a *Agent) learnTerms(said string, history []llm.Message) {
	names := confirmedNames(said, history)
	if len(names) == 0 {
		return
	}

	a.mu.Lock()
	learned := false
	for _, name := range names {
		if !slices.ContainsFunc(a.learnedTerms, func(known string) bool { return strings.EqualFold(known, name) }) {
			a.learnedTerms = append(a.learnedTerms, name)
			learned = true
		}
	}
	if !learned {
		a.mu.Unlock()
		return
	}
	terms := stt.CleanKeyterms(append(append([]string(nil), a.options.Keyterms...), a.learnedTerms...))
	if len(terms) > stt.MaxKeyterms {
		terms = terms[:stt.MaxKeyterms]
	}
	listeners := make([]*sttrouter.Session, 0, len(a.listeners))
	for _, session := range a.listeners {
		listeners = append(listeners, session)
	}
	a.mu.Unlock()

	for _, listener := range listeners {
		if err := listener.SetKeyterms(terms); err != nil {
			a.logger.Warn("could not tell the transcriber which names to expect", "error", err)
			continue
		}
		a.logger.Debug("the transcriber was told the names said back, if it takes them mid-session", "terms", terms)
	}
}

// confirmedNames are the capitalised words of the agent's reply, other than one opening a
// sentence, that the caller has also said.
func confirmedNames(said string, history []llm.Message) []string {
	caller := map[string]bool{}
	for _, message := range history {
		if message.Role != llm.User {
			continue
		}
		for _, word := range strings.FieldsFunc(message.Content, notNameRune) {
			caller[strings.ToLower(word)] = true
		}
	}
	if len(caller) == 0 {
		return nil
	}

	var names []string
	opening := true
	for _, field := range strings.Fields(said) {
		for _, word := range strings.FieldsFunc(field, notNameRune) {
			runes := []rune(word)
			if !opening && len(runes) > 1 && unicode.IsUpper(runes[0]) && caller[strings.ToLower(word)] &&
				!slices.Contains(names, word) {
				names = append(names, word)
			}
			opening = false
		}
		opening = strings.ContainsAny(field[len(field)-1:], ".!?")
	}
	return names
}

// notNameRune splits words at anything but a letter or a hyphen, so "I'm" is two words and
// neither is a name.
func notNameRune(r rune) bool {
	return !unicode.IsLetter(r) && r != '-'
}
