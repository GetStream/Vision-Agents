//go:build ignore

// Generate writes flow cases for the states the AMI corpus cannot supply, the ones a phone
// agent meets: a caller mid-way through a number, a recorded menu, an ambiguous request, a
// related addition while the agent talks, an echo, a noise, a voice in the caller's room. They
// are training data for a local controller, never a benchmark.
//
// A model writes the words of each case through a CLI (Meta's muse, by default), for a state
// described in words and a business none of the benchmark's scenarios use, so nothing is
// copied from the written set. The label is not the model's: it follows from the state asked
// for, and the fields that decide which options a controller is offered are set here too.
// Cases whose words match a benchmark case are dropped.
//
// Every call's answer is kept in -work, so an interrupted run resumes where it stopped:
//
//	go run ./testdata/synthetic/generate.go -work /tmp/synthetic -out synthetic.json
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

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
	AnotherVoice  bool      `json:"another_voice,omitempty"`
	Unfinished    bool      `json:"unfinished,omitempty"`
	Expect        string    `json:"expect"`
}

// business is a scenario the agent works in. An outbound one has the agent calling a line on
// somebody's behalf, which is where recorded menus are met.
type business struct {
	name, contract string
	outbound       bool
}

// businesses share nothing with the written set's restaurant, clinic, wireless carrier and
// referral call.
var businesses = []business{
	{"dentist", "You are the reception line for Larkspur Dental in Portland. You can book, move and cancel check-ups and hygiene visits, and you note whether a patient is nervous or needs sedation.", false},
	{"bank", "You are the card services line for Harbor Mutual Bank. You can freeze and unfreeze cards, report fraud, read recent transactions and order replacement cards. Never read out a full card number.", false},
	{"airline", "You are the reservations line for Cascadia Air. You can change flights, add bags, pick seats and cancel bookings, and you always confirm the passenger's surname and booking reference first.", false},
	{"hotel", "You take bookings for the Alder House hotel in Savannah. You can book, change and cancel rooms, add breakfast or parking, and note accessibility needs.", false},
	{"pharmacy", "You are the refill line for Mesa Pharmacy. You can refill prescriptions, move a pickup to another store, and say whether an order is ready. You never give medical advice.", false},
	{"insurance", "You take car insurance claims for Northstar Insurance. You collect the policy number, the date and place of the incident, and whether anyone was hurt, and you open a claim.", false},
	{"car-rental", "You are the reservations line for Ridgeway Car Rental. You can book, extend and cancel rentals, add a second driver or a child seat, and quote prices.", false},
	{"utility", "You are customer service for Bluewater Energy. You can take meter readings, set up payment plans, move a supply to a new address and report outages.", false},
	{"vet", "You are the front desk for Pinecrest Veterinary Clinic. You book, move and cancel appointments for pets, and you send emergencies to the on-call vet.", false},
	{"it-helpdesk", "You are the IT helpdesk for Kestrel Logistics staff. You can reset passwords, unlock accounts, and open tickets for broken laptops and printers.", false},
	{"parcels", "You are the support line for Swiftline Parcels. You can track parcels, rearrange a delivery, redirect to a pickup point and open a claim for a lost item.", false},
	{"gym", "You are the membership line for Ironleaf Gyms. You can freeze, upgrade and cancel memberships, book personal training and update payment details.", false},
	{"pizza", "You take phone orders for Nonna's Pizza. You can take orders for delivery or collection, add or remove toppings and quote a delivery time.", false},
	{"salon", "You take bookings for Maple and Thread hair salon. You can book, move and cancel cuts, colours and treatments with a named stylist.", false},
	{"council", "You are the phone line for Brookfield City Council. You can report missed bin collections, book bulky waste pickups and log potholes.", false},
	{"university", "You are the admissions line for Westhaven University. You can say where an application stands, book campus tours and send forms.", false},
	{"plumber", "You take bookings for Clearflow Plumbing. You book call-outs, give time windows, and treat burst pipes and leaks with no water shut-off as emergencies.", false},
	{"cinema", "You take bookings for the Starlight Cinema. You can book and refund tickets, pick seats and say what is showing.", false},
	{"airline-outbound", "You are calling Cascadia Air on behalf of a traveller to move their flight to the next day. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
	{"bank-outbound", "You are calling Harbor Mutual Bank on behalf of a customer to dispute a card charge. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
	{"utility-outbound", "You are calling Bluewater Energy on behalf of a tenant to report a power outage. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
	{"pharmacy-outbound", "You are calling Mesa Pharmacy on behalf of a patient to check whether a prescription is ready. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
	{"insurer-outbound", "You are calling Northstar Insurance on behalf of a driver to ask about an open claim. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
	{"council-outbound", "You are calling Brookfield City Council on behalf of a resident to book a bulky waste pickup. Work through whatever menu answers you and ask for a person if the menu cannot help.", true},
}

// state is one situation, described for the writer, and what the controller should do in it.
type state struct {
	name, expect, describe string
	speaking, outbound     bool
}

const (
	floorFree = "The agent is silent and the floor is free; the words below have just been said to it. "
	talking   = "The agent is in the middle of speaking when the words below arrive. agent_said is what it has said so far, usually cut off mid-sentence. "
	elsewhere = "The words are from a different person in the caller's room, talking to somebody there rather than to the agent: set participant to \"Someone at the caller's microphone\" and another_voice to true. "
)

var states = []state{
	{"respond", "answer", floorFree + "The caller has finished a complete, unambiguous request or question for the agent, one it can act on.", false, false},
	{"wait", "wait", floorFree + "The caller has stopped part way through a sentence: the words end on something that needs more, such as a preposition, an article, a conjunction, \"um\", or a trailing clause. Vary how they trail off.", false, false},
	{"wait-digits", "wait", floorFree + "The agent asked for a number, such as an account, card, policy, booking reference, phone number, date, or time, and the caller has said only the first part of it, spoken as words. The number is plainly not complete yet.", false, false},
	{"wait-menu", "wait", "The agent is on a call it placed, and the other end is a recorded menu. The menu has read some of its options but has not yet asked the caller to choose. Set participant to \"The line\"; history may be empty or hold what the menu said before.", false, true},
	{"clarify", "answer-clarify", floorFree + "The caller asks for something, but what they mean is ambiguous from the conversation: \"it\" or \"that one\" could be either of two things just mentioned, \"the usual\" names nothing the agent knows, or \"the other one\" has no single referent. The history must set up the ambiguity.", false, false},
	{"ignore", "ignore", floorFree + "The words are not for the agent: either a different person in the caller's room talking to somebody there (set participant to \"Someone at the caller's microphone\" and another_voice to true), or the caller turning away to talk to someone else in the room (participant \"The caller\"). Mix the two.", false, false},
	{"stop", "interrupt", talking + "The caller cuts in with a correction, a new request, a question, or a direct interruption such as \"wait\", \"no\", or \"hang on\", so the agent must stop.", true, false},
	{"shorten", "shorten", talking + "The caller adds something related that the agent's current answer must now also cover, such as \"and Saturday too\" or \"and my daughter as well\", so the agent should wrap up and answer both briefly. It is an addition, not a correction.", true, false},
	{"continue-ack", "continue", talking + "The caller only acknowledges, with a backchannel such as \"okay\", \"mm-hmm\", \"right\", \"yep\", or \"got it\". Vary them.", true, false},
	{"continue-noise", "continue", talking + "What arrives is a noise from the caller's side, written in square brackets, such as [coughs], [baby crying], [keyboard typing], [door closes].", true, false},
	{"continue-echo", "continue", talking + "What arrives repeats a few words the agent is saying, as a phone line's echo of its own voice would: heard must be words taken from agent_said.", true, false},
	{"continue-elsewhere", "ignore", talking + elsewhere, true, false},
}

const schema = `Write %d varied cases as JSON, and nothing else:
{"cases": [{"history": [{"speaker": "caller" or "agent", "text": "..."}], "participant": "The caller", "heard": "...",%s "another_voice": false}]}
history is the conversation before the words, zero to four turns, in the business's own details. heard is the words themselves, as a speech recogniser writes them: no quotation marks, and plain spoken English. Make every case different in wording, length and topic, and natural for a phone call.`

func prompt(b business, s state, n, variant int) string {
	extra := ""
	if s.speaking {
		extra = ` "agent_said": "...",`
	}
	return fmt.Sprintf("You write test cases for a voice agent's turn-taking controller.\n\nThe agent has been told: %s\n\nThe situation: %s\n\n%s\n\nThis is batch %d for this business: avoid the obvious first ideas.",
		b.contract, s.describe, fmt.Sprintf(schema, n, extra), variant+1)
}

type job struct {
	b       business
	s       state
	variant int
}

func main() {
	work := flag.String("work", "", "directory keeping each call's answer, so a run resumes")
	out := flag.String("out", "", "where to write the set")
	bench := flag.String("bench", "../flowbench.json", "the written benchmark, whose words are kept out")
	perCall := flag.Int("per-call", 8, "cases a call asks for")
	variants := flag.Int("variants", 1, "calls per state and business; recorded menus get three times as many")
	workers := flag.Int("workers", 16, "calls in flight")
	model := flag.String("model", "muse-spark-1.3-contributor", "the writer")
	effort := flag.String("effort", "max", "the writer's reasoning effort")
	limit := flag.Int("limit", 0, "make only this many calls, for a trial")
	more := flag.String("more", "", "extra calls for some states, such as shorten=3: calls per business for them")
	flag.Parse()
	if *work == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: generate -work DIR -out FILE")
		os.Exit(2)
	}
	must(os.MkdirAll(*work, 0o755))
	empty := filepath.Join(*work, "cwd")
	must(os.MkdirAll(empty, 0o755)) // the CLI roots its tools here, so it has nothing to touch

	var jobs []job
	for _, s := range states {
		for _, b := range businesses {
			if s.outbound != b.outbound {
				continue
			}
			n := *variants
			if s.outbound {
				n *= 3
			}
			for _, extra := range strings.Split(*more, ",") {
				if name, count, ok := strings.Cut(extra, "="); ok && name == s.name {
					fmt.Sscan(count, &n)
				}
			}
			for v := range n {
				jobs = append(jobs, job{b, s, v})
			}
		}
	}

	if *limit > 0 && *limit < len(jobs) {
		jobs = jobs[:*limit]
	}
	queue := make(chan job)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done, failed := 0, 0
	for range *workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range queue {
				path := filepath.Join(*work, fmt.Sprintf("%s-%s-%d.json", j.s.name, j.b.name, j.variant))
				if _, err := os.Stat(path); err == nil {
					continue
				}
				started := time.Now()
				text, err := ask(*model, *effort, empty, strings.TrimSuffix(path, ".json")+".prompt",
					prompt(j.b, j.s, *perCall, j.variant))
				mu.Lock()
				if err == nil {
					err = os.WriteFile(path, []byte(text), 0o644)
				}
				if err != nil {
					failed++
					fmt.Fprintf(os.Stderr, "%s: %v\n", filepath.Base(path), err)
				} else {
					done++
					fmt.Fprintf(os.Stderr, "%d/%d %s in %s\n", done, len(jobs), filepath.Base(path), time.Since(started).Round(time.Second))
				}
				mu.Unlock()
			}
		}()
	}
	for _, j := range jobs {
		queue <- j
	}
	close(queue)
	wg.Wait()

	known := map[string]bool{}
	if raw, err := os.ReadFile(*bench); err == nil {
		var set struct {
			Cases []flowCase `json:"cases"`
		}
		must(json.Unmarshal(raw, &set))
		for _, c := range set.Cases {
			known[identity(c)] = true
		}
	}
	contracts := map[string]string{}
	var cases []flowCase
	counts := map[string]int{}
	for _, j := range jobs {
		raw, err := os.ReadFile(filepath.Join(*work, fmt.Sprintf("%s-%s-%d.json", j.s.name, j.b.name, j.variant)))
		if err != nil {
			continue
		}
		var answer struct {
			Cases []flowCase `json:"cases"`
		}
		if err := json.Unmarshal([]byte(lastObject(string(raw))), &answer); err != nil {
			fmt.Fprintf(os.Stderr, "%s-%s-%d: unreadable: %v\n", j.s.name, j.b.name, j.variant, err)
			continue
		}
		for i, c := range answer.Cases {
			key := identity(c)
			if normal(c.Heard) == "" || known[key] || (j.s.speaking && strings.TrimSpace(c.AgentSaid) == "") {
				continue
			}
			if j.s.name == "continue-echo" && !strings.Contains(normal(c.AgentSaid), normal(c.Heard)) {
				continue // an echo repeats the agent's own words
			}
			known[key] = true
			c.ID = fmt.Sprintf("synthetic-%s-%s-%d-%d", j.s.name, j.b.name, j.variant, i)
			c.State, c.Expect, c.Contract = j.s.name, j.s.expect, j.b.name
			c.Source = "synthetic, written by " + *model
			c.AgentSpeaking = j.s.speaking
			if !c.AgentSpeaking {
				c.AgentSaid = ""
			}
			// Unfinished means the caller's words are still arriving: the provisional ruling
			// made mid-interjection. A backchannel, a noise and an echo are ruled on as they
			// arrive; a stop or a shorten mostly is, sometimes once settled; words for somebody
			// else in the room are ruled on once settled, since mid-utterance an ignore can only
			// mean carrying on. This follows the written set.
			switch j.s.name {
			case "continue-ack", "continue-noise", "continue-echo":
				c.Unfinished = true
			case "stop", "shorten":
				c.Unfinished = fnv32(c.ID)%5 != 0
			}
			switch j.s.name {
			case "wait-menu":
				c.Participant = "The line"
			case "continue-elsewhere":
				c.Participant, c.AnotherVoice = "Someone at the caller's microphone", true
			}
			if c.Participant == "" {
				c.Participant = "The caller"
			}
			contracts[j.b.name] = j.b.contract
			cases = append(cases, c)
			counts[j.s.name]++
		}
	}
	sort.SliceStable(cases, func(a, b int) bool { return cases[a].State < cases[b].State })
	set := struct {
		Version   int               `json:"version"`
		Source    string            `json:"source"`
		Contracts map[string]string `json:"contracts"`
		Cases     []flowCase        `json:"cases"`
	}{1, "synthetic, written by " + *model + " at " + *effort + " effort; labels follow from the state asked for", contracts, cases}
	encoded, err := json.MarshalIndent(set, "", "  ")
	must(err)
	must(os.WriteFile(*out, append(encoded, '\n'), 0o644))
	fmt.Fprintf(os.Stderr, "%d cases, %d calls failed: %v\n", len(cases), failed, counts)
}

// ask runs one prompt, kept at file, through the CLI with no tools and one model step.
func ask(model, effort, dir, file, text string) (string, error) {
	if err := os.WriteFile(file, []byte(text), 0o644); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "muse", "exec", "--model", model, "--reasoning-effort", effort,
		"--max-model-steps", "1", "--prompt-file", file)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%w: %s", err, lastLine(stderr.String()))
	}
	if !strings.Contains(stdout.String(), "}") {
		return "", fmt.Errorf("no JSON in the answer: %s", lastLine(stdout.String()))
	}
	return stdout.String(), nil
}

// lastObject is the outermost JSON object in text, from its first { to its last }.
func lastObject(text string) string {
	start, end := strings.IndexByte(text, '{'), strings.LastIndexByte(text, '}')
	if start < 0 || end < start {
		return text
	}
	return text[start : end+1]
}

// identity is what makes two cases the same case: the words, and what the agent was saying
// when they came. A backchannel's words repeat from case to case; its moment does not.
func identity(c flowCase) string {
	return normal(c.Heard) + "|" + normal(c.AgentSaid)
}

var nonWord = regexp.MustCompile(`[^a-z0-9]+`)

func normal(s string) string {
	return strings.TrimSpace(nonWord.ReplaceAllString(strings.ToLower(s), " "))
}

// fnv32 is FNV-1a, a stable choice per case.
func fnv32(s string) uint32 {
	h := uint32(2166136261)
	for i := range len(s) {
		h = (h ^ uint32(s[i])) * 16777619
	}
	return h
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return lines[len(lines)-1]
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
