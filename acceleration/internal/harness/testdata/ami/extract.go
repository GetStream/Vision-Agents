//go:build ignore

// Extract turns the AMI Meeting Corpus into flow cases, so the flow controller is measured
// on what people actually did rather than only on what somebody wrote down.
//
// Every label is a human's own behaviour at that moment: a pause the speaker talked through
// is a wait, a question somebody answered is a respond, a backchannel the speaker talked
// over is a continue, and an interjection the speaker gave way to is a stop.
//
// The corpus is downloaded on first use and kept in the user cache directory, so regenerating
// the set is one command from internal/harness:
//
//	go generate -run ami .
package main

import (
	"archive/zip"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// corpusURL is the manual annotations: words with their timings and dialogue acts, no audio.
const corpusURL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"

// perState is how many cases each state gets. The ones found are spread across meetings, so
// no one group of four people decides a state.
const perState = 40

// The thresholds are the conventional ones from the turn-taking literature rather than
// numbers fitted to this set.
const (
	// pauseSec is a silence long enough that an endpointer would have called the turn over.
	pauseSec = 0.6
	// answerWithinSec is how quickly somebody must take the floor for a question to count
	// as handed over.
	answerWithinSec = 2.0
	// yieldWithinSec is how quickly the speaker must stop for an interjection to count as
	// having taken the floor from them.
	yieldWithinSec = 1.0
	// carriedOnSec is how long the speaker must keep going for them to have held the floor.
	carriedOnSec = 1.0
	// provisionalSec is how much of an interjection the controller hears before it rules,
	// which is the provisional ask made while the agent is still talking.
	provisionalSec = 1.5
)

const contract = "You are one of four colleagues in a product design meeting at Real " +
	"Reactions, a company designing a new television remote control. Everything said in " +
	"the meeting is said to the whole group, you included. Take part the way a colleague " +
	"would: answer a question put to the group or to you, and do not talk over whoever " +
	"holds the floor."

type word struct {
	id         string
	text       string
	start, end float64
	timed      bool
	punc       bool
}

type act struct {
	meeting, speaker, kind string
	words                  []word
	start, end             float64
}

func (a act) text() string { return join(a.words) }

type history struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}

type flowCase struct {
	ID            string    `json:"id"`
	State         string    `json:"state"`
	Contract      string    `json:"contract"`
	Source        string    `json:"source"`
	History       []history `json:"history"`
	Participant   string    `json:"participant"`
	Heard         string    `json:"heard"`
	AgentSpeaking bool      `json:"agent_speaking,omitempty"`
	AgentSaid     string    `json:"agent_said,omitempty"`
	Unfinished    bool      `json:"unfinished,omitempty"`
	Expect        string    `json:"expect"`
}

func main() {
	in := flag.String("in", "", "unpacked AMI manual annotations; downloaded when empty")
	out := flag.String("out", "", "where to write the set; standard output when empty")
	flag.Parse()

	corpus, err := open(*in)
	must(err)
	kinds, err := daTypes(corpus)
	must(err)
	files, err := fs.Glob(corpus, "dialogueActs/*.dialog-act.xml")
	must(err)

	meetings := map[string][]act{}
	for _, file := range files {
		base := path.Base(file)
		meeting, speaker := strings.Split(base, ".")[0], strings.Split(base, ".")[1]
		// The scenario meetings are the ones the contract describes.
		if !regexp.MustCompile(`^(ES|IS|TS)`).MatchString(meeting) {
			continue
		}
		words, err := readWords(corpus, "words/"+meeting+"."+speaker+".words.xml")
		must(err)
		acts, err := readActs(corpus, file, meeting, speaker, words, kinds)
		must(err)
		meetings[meeting] = append(meetings[meeting], acts...)
	}

	found := map[string][]flowCase{}
	names := make([]string, 0, len(meetings))
	for name := range meetings {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		acts := meetings[name]
		sort.Slice(acts, func(i, j int) bool { return acts[i].start < acts[j].start })
		for _, one := range find(acts) {
			found[one.State] = append(found[one.State], one)
		}
	}

	set := struct {
		Version   int               `json:"version"`
		Source    string            `json:"source"`
		Contracts map[string]string `json:"contracts"`
		Cases     []flowCase        `json:"cases"`
	}{
		Version: 1,
		Source: "AMI Meeting Corpus manual annotations 1.6.2, CC BY 4.0, " +
			"https://groups.inf.ed.ac.uk/ami/corpus/",
		Contracts: map[string]string{"meeting": contract},
	}
	for _, state := range []string{"respond", "wait", "stop", "continue-ack"} {
		picked := spread(found[state], perState)
		fmt.Fprintf(os.Stderr, "%s: %d found, %d kept\n", state, len(found[state]), len(picked))
		set.Cases = append(set.Cases, picked...)
	}

	encoded, err := json.MarshalIndent(set, "", "  ")
	must(err)
	encoded = append(encoded, '\n')
	if *out == "" {
		_, err = os.Stdout.Write(encoded)
	} else {
		err = os.WriteFile(*out, encoded, 0o644)
	}
	must(err)
}

// open reads the corpus from dir, or from the archive, downloading it the first time.
func open(dir string) (fs.FS, error) {
	if dir != "" {
		return os.DirFS(dir), nil
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return nil, err
	}
	archive := filepath.Join(cache, "vision-agents", path.Base(corpusURL))
	if _, err := os.Stat(archive); err != nil {
		fmt.Fprintf(os.Stderr, "downloading %s\n", corpusURL)
		if err := download(corpusURL, archive); err != nil {
			return nil, err
		}
	}
	return zip.OpenReader(archive)
}

// download writes url to dst, through a temporary file so an interrupted download is not
// mistaken for a finished one next time.
func download(url, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	response, err := http.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, response.Status)
	}
	partial, err := os.CreateTemp(filepath.Dir(dst), "ami-*.zip")
	if err != nil {
		return err
	}
	defer os.Remove(partial.Name())
	if _, err := io.Copy(partial, response.Body); err != nil {
		partial.Close()
		return err
	}
	if err := partial.Close(); err != nil {
		return err
	}
	return os.Rename(partial.Name(), dst)
}

// find is every case one meeting offers.
func find(acts []act) []flowCase {
	var cases []flowCase
	for i, held := range acts {
		if !substantive(held.kind) {
			continue
		}
		// A pause the speaker talked through, with nobody else starting in it.
		for w := 3; w < len(held.words)-3; w++ {
			before, after := held.words[w], held.words[w+1]
			if !before.timed || !after.timed || before.punc || after.punc ||
				after.start-before.end < pauseSec || sentenceEnd(held.words, w) ||
				speaksBetween(acts, held.speaker, before.end, after.start) {
				continue
			}
			listener := lastOther(acts, i, held.speaker)
			if listener == "" {
				break
			}
			cases = append(cases, flowCase{
				State:   "wait",
				Source:  source(held.meeting, before.end),
				History: context(acts, i, listener),
				Heard:   join(held.words[:w+1]),
				Expect:  "wait",
			})
			break
		}

		// A question somebody else answered, promptly and without the asker going on.
		if strings.HasPrefix(held.kind, "el.") && strings.HasSuffix(held.text(), "?") &&
			len(spoken(held.words)) >= 4 {
			if next := nextOther(acts, i, held.speaker); next != nil &&
				next.start >= held.end && next.start-held.end <= answerWithinSec &&
				!talks(acts, held.speaker, held.end+0.05, next.start+carriedOnSec) {
				cases = append(cases, flowCase{
					State:   "respond",
					Source:  source(held.meeting, held.end),
					History: context(acts, i, next.speaker),
					Heard:   held.text(),
					Expect:  "answer",
				})
			}
		}

		// Somebody spoke over this act. The speaker is the agent.
		if len(spoken(held.words)) < 8 {
			continue
		}
		for _, over := range acts {
			if over.speaker == held.speaker || over.start < held.start+1 ||
				over.start > held.end-0.3 {
				continue
			}
			said := join(before(held.words, over.start))
			if len(spoken(before(held.words, over.start))) < 4 {
				continue
			}
			switch {
			case over.kind == "bck" && held.end-over.end >= carriedOnSec &&
				!caughtBeforeTheModel(over.text()):
				cases = append(cases, flowCase{
					State:         "continue-ack",
					Source:        source(held.meeting, over.start),
					History:       context(acts, i, held.speaker),
					Heard:         over.text(),
					AgentSpeaking: true,
					AgentSaid:     said,
					Expect:        "continue",
				})
			case substantive(over.kind) && over.end-over.start >= carriedOnSec &&
				stoppedBy(acts, held, over):
				// An interjection that opens on a murmur is a backchannel however it goes on.
				heard := before(over.words, over.start+provisionalSec)
				if len(spoken(heard)) < 3 || caughtBeforeTheModel(spoken(heard)[0].text) {
					continue
				}
				cases = append(cases, flowCase{
					State:         "stop",
					Source:        source(held.meeting, over.start),
					History:       context(acts, i, held.speaker),
					Heard:         join(heard),
					AgentSpeaking: true,
					AgentSaid:     said,
					Unfinished:    true,
					Expect:        "interrupt",
				})
			}
		}
	}
	for i := range cases {
		cases[i].ID = "ami-" + cases[i].State + "-" + strings.ReplaceAll(
			strings.ToLower(strings.TrimPrefix(cases[i].Source, "AMI ")), " ", "-")
		cases[i].Contract = "meeting"
		cases[i].Participant = "A colleague"
	}
	return cases
}

// stoppedBy reports whether the speaker of held gave way to over: they fell silent soon after
// it started and said nothing more until it was over.
func stoppedBy(acts []act, held, over act) bool {
	last := held.start
	for _, one := range acts {
		if one.speaker != held.speaker {
			continue
		}
		for _, w := range one.words {
			if w.timed && w.start < over.end && w.end > last {
				last = w.end
			}
		}
	}
	return last >= over.start && last-over.start <= yieldWithinSec
}

func substantive(kind string) bool {
	switch kind {
	case "inf", "sug", "ass", "el.inf", "el.sug", "el.ass", "el.und":
		return true
	}
	return false
}

// sentenceEnd reports whether the word at w closes a sentence, which a pause after is a
// boundary rather than a hesitation.
func sentenceEnd(words []word, w int) bool {
	return w+1 < len(words) && words[w+1].punc && strings.ContainsAny(words[w+1].text, ".?!")
}

// speaksBetween reports whether anybody but speaker says a word between from and to.
func speaksBetween(acts []act, speaker string, from, to float64) bool {
	for _, one := range acts {
		if one.speaker == speaker || one.end < from || one.start > to {
			continue
		}
		for _, w := range one.words {
			if w.timed && !w.punc && w.end > from && w.start < to {
				return true
			}
		}
	}
	return false
}

// talks reports whether speaker says a word between from and to.
func talks(acts []act, speaker string, from, to float64) bool {
	for _, one := range acts {
		if one.speaker != speaker || one.end < from || one.start > to {
			continue
		}
		for _, w := range one.words {
			if w.timed && !w.punc && w.end > from && w.start < to {
				return true
			}
		}
	}
	return false
}

func lastOther(acts []act, i int, speaker string) string {
	for j := i - 1; j >= 0; j-- {
		if acts[j].speaker != speaker && substantive(acts[j].kind) {
			return acts[j].speaker
		}
	}
	return ""
}

func nextOther(acts []act, i int, speaker string) *act {
	for j := i + 1; j < len(acts); j++ {
		if acts[j].speaker != speaker && acts[j].kind != "bck" {
			return &acts[j]
		}
	}
	return nil
}

// context is the four substantive acts before i, told from the agent's side.
func context(acts []act, i int, agent string) []history {
	var said []history
	for j := i - 1; j >= 0 && len(said) < 4; j-- {
		if !substantive(acts[j].kind) || len(spoken(acts[j].words)) > 40 {
			continue
		}
		speaker := "colleague"
		if acts[j].speaker == agent {
			speaker = "agent"
		}
		said = append([]history{{Speaker: speaker, Text: acts[j].text()}}, said...)
	}
	return said
}

// spread takes up to n cases, one meeting at a time, so a talkative meeting cannot fill a
// state on its own.
func spread(cases []flowCase, n int) []flowCase {
	byMeeting := map[string][]flowCase{}
	var order []string
	for _, one := range cases {
		meeting := strings.Fields(one.Source)[1]
		if _, ok := byMeeting[meeting]; !ok {
			order = append(order, meeting)
		}
		byMeeting[meeting] = append(byMeeting[meeting], one)
	}
	var picked []flowCase
	for round := 0; len(picked) < n; round++ {
		added := false
		for _, meeting := range order {
			if round < len(byMeeting[meeting]) && len(picked) < n {
				picked = append(picked, byMeeting[meeting][round])
				added = true
			}
		}
		if !added {
			break
		}
	}
	return picked
}

func before(words []word, at float64) []word {
	var kept []word
	for _, w := range words {
		if w.timed && w.start >= at {
			break
		}
		kept = append(kept, w)
	}
	return kept
}

func spoken(words []word) []word {
	var kept []word
	for _, w := range words {
		if !w.punc {
			kept = append(kept, w)
		}
	}
	return kept
}

func join(words []word) string {
	var out strings.Builder
	for _, w := range words {
		if out.Len() > 0 && !w.punc {
			out.WriteByte(' ')
		}
		out.WriteString(w.text)
	}
	return out.String()
}

func source(meeting string, at float64) string {
	return fmt.Sprintf("AMI %s %.2fs", meeting, at)
}

// caughtBeforeTheModel is the agent's overlapNoise, as flowbench_test.go mirrors it.
func caughtBeforeTheModel(text string) bool {
	var kept strings.Builder
	for _, symbol := range strings.ToLower(text) {
		if unicode.IsLetter(symbol) || unicode.IsDigit(symbol) || unicode.IsSpace(symbol) {
			kept.WriteRune(symbol)
		}
	}
	stripped := strings.Join(strings.Fields(kept.String()), " ")
	if stripped == "" || strings.Contains(stripped, "cough") || strings.Contains(stripped, "ahem") {
		return true
	}
	switch stripped {
	case "huh", "uh", "mm", "hm", "hmm":
		return true
	}
	return false
}

func daTypes(corpus fs.FS) (map[string]string, error) {
	raw, err := fs.ReadFile(corpus, "ontologies/da-types.xml")
	if err != nil {
		return nil, err
	}
	kinds := map[string]string{}
	for _, match := range regexp.MustCompile(`nite:id="(ami_da_\d+)" name="([a-z.]+)"`).
		FindAllStringSubmatch(string(raw), -1) {
		kinds[match[1]] = match[2]
	}
	return kinds, nil
}

func readWords(corpus fs.FS, name string) ([]word, error) {
	file, err := corpus.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := xml.NewDecoder(file)
	decoder.CharsetReader = func(_ string, input io.Reader) (io.Reader, error) { return input, nil }
	var words []word
	for {
		token, err := decoder.Token()
		if err != nil {
			break
		}
		start, ok := token.(xml.StartElement)
		if !ok || start.Name.Local != "w" {
			continue
		}
		var parsed struct {
			ID    string   `xml:"http://nite.sourceforge.net/ id,attr"`
			Start *float64 `xml:"starttime,attr"`
			End   *float64 `xml:"endtime,attr"`
			Punc  string   `xml:"punc,attr"`
			Text  string   `xml:",chardata"`
		}
		if err := decoder.DecodeElement(&parsed, &start); err != nil {
			return nil, err
		}
		w := word{id: parsed.ID, text: parsed.Text, punc: parsed.Punc == "true"}
		if parsed.Start != nil && parsed.End != nil {
			w.start, w.end, w.timed = *parsed.Start, *parsed.End, true
		}
		words = append(words, w)
	}
	return words, nil
}

var (
	childRange = regexp.MustCompile(`id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?`)
	daPointer  = regexp.MustCompile(`da-types\.xml#id\((ami_da_\d+)\)`)
)

func readActs(
	corpus fs.FS, name, meeting, speaker string, words []word, kinds map[string]string,
) ([]act, error) {
	raw, err := fs.ReadFile(corpus, name)
	if err != nil {
		return nil, err
	}
	index := map[string]int{}
	for i, w := range words {
		index[w.id] = i
	}

	var acts []act
	for _, block := range strings.Split(string(raw), "<dact ")[1:] {
		kind := ""
		if match := daPointer.FindStringSubmatch(block); match != nil {
			kind = kinds[match[1]]
		}
		child := regexp.MustCompile(`<nite:child href="[^#]+#([^"]+)"`).FindStringSubmatch(block)
		if child == nil {
			continue
		}
		ids := childRange.FindStringSubmatch(child[1])
		from, ok := index[ids[1]]
		if !ok {
			continue
		}
		to := from
		if ids[2] != "" {
			if to, ok = index[ids[2]]; !ok {
				continue
			}
		}
		one := act{meeting: meeting, speaker: speaker, kind: kind, words: words[from : to+1]}
		timed := false
		for _, w := range one.words {
			if !w.timed {
				continue
			}
			if !timed {
				one.start, timed = w.start, true
			}
			one.end = w.end
		}
		if timed {
			acts = append(acts, one)
		}
	}
	return acts, nil
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
