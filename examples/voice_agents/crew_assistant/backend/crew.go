package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Person is someone on the team.
type Person struct {
	ID    string   `json:"id"`
	Name  string   `json:"name"`
	Role  string   `json:"role"`
	Roles []string `json:"stations,omitempty"`
	Note  string   `json:"note,omitempty"`
}

// Shift is one shift on the schedule.
type Shift struct {
	ID      string    `json:"id"`
	Person  string    `json:"-"`
	Station string    `json:"station"`
	Start   time.Time `json:"start"`
	End     time.Time `json:"end"`
	Leader  string    `json:"shift_leader"`
}

// Update is an announcement or a piece of team news.
type Update struct {
	Date  string `json:"date"`
	Kind  string `json:"kind"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// Guide is the quick fix for one error code on one asset.
type Guide struct {
	Asset      string   `json:"asset"`
	Code       string   `json:"code"`
	Meaning    string   `json:"meaning"`
	Safety     string   `json:"safety"`
	Steps      []string `json:"steps"`
	Seconds    int      `json:"seconds"`
	EscalateIf string   `json:"escalate_if"`
}

// InboxItem is something the leaders see: an alert, a swap, a ticket or a check-in.
type InboxItem struct {
	ID      string    `json:"id"`
	Kind    string    `json:"kind"`
	From    string    `json:"from"`
	Title   string    `json:"title"`
	Details string    `json:"details"`
	Status  string    `json:"status"`
	At      time.Time `json:"at"`
}

// Reflection is an end-of-shift reflection, kept without who wrote it.
type Reflection struct {
	Energy int       `json:"energy"`
	Themes []string  `json:"themes"`
	Win    string    `json:"win"`
	At     time.Time `json:"at"`
}

// restaurant is Harbor Grill: its people, schedule, handbook, equipment and the leaders' inbox.
type restaurant struct {
	mu          sync.Mutex
	people      map[string]Person
	shifts      []Shift
	handbook    handbook
	updates     []Update
	guides      []Guide
	menu        []map[string]any
	inbox       []InboxItem
	reflections []Reflection
	nextID      int
}

const (
	me          = "you"
	handbookURL = "https://handbook.example.com/"
)

func newRestaurant(now time.Time) *restaurant {
	day := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.Local)
	at := func(days, hour int) time.Time { return day.AddDate(0, 0, days).Add(time.Duration(hour) * time.Hour) }
	date := func(days int) string { return day.AddDate(0, 0, days).Format("2006-01-02") }

	r := &restaurant{people: map[string]Person{}}
	for _, p := range []Person{
		{ID: "maya", Name: "Maya Chen", Role: "Shift leader, front of house", Note: "Runs most morning shifts"},
		{ID: "sam", Name: "Sam Ortiz", Role: "Shift leader, kitchen", Note: "Runs most evening shifts"},
		{ID: "taylor", Name: "Taylor Morgan", Role: "Owner", Note: "Approves repair tickets and schedules"},
		{ID: "priya", Name: "Priya Patel", Role: "Trainer", Note: "Your trainer for the first 30 days"},
		{ID: "dev", Name: "Dev Harris", Role: "Team member", Roles: []string{"Drive-thru", "Register"}, Note: "Your onboarding buddy"},
		{ID: "lena", Name: "Lena Brooks", Role: "Team member", Roles: []string{"Drive-thru", "Kitchen"}},
		{ID: "marco", Name: "Marco Ruiz", Role: "Team member", Roles: []string{"Kitchen"}},
		{ID: "ava", Name: "Ava Kim", Role: "Team member", Roles: []string{"Register", "Drive-thru"}},
	} {
		r.people[p.ID] = p
	}
	r.shifts = []Shift{
		{ID: "s-101", Person: me, Station: "Drive-thru", Start: at(1, 6), End: at(1, 14), Leader: "maya"},
		{ID: "s-102", Person: me, Station: "Kitchen", Start: at(2, 16), End: at(2, 22), Leader: "sam"},
		{ID: "s-103", Person: me, Station: "Register", Start: at(4, 10), End: at(4, 16), Leader: "maya"},
		{ID: "s-201", Person: "lena", Station: "Kitchen", Start: at(1, 10), End: at(1, 18), Leader: "maya"},
		{ID: "s-202", Person: "ava", Station: "Register", Start: at(2, 16), End: at(2, 22), Leader: "sam"},
		{ID: "s-203", Person: "dev", Station: "Drive-thru", Start: at(4, 6), End: at(4, 14), Leader: "maya"},
		{ID: "s-204", Person: "marco", Station: "Kitchen", Start: at(1, 6), End: at(1, 14), Leader: "maya"},
	}
	r.updates = []Update{
		{date(-6), "Menu", "Spicy bean wrap launched", "The spicy bean wrap is on the menu from today. It contains wheat, milk and egg. The new prep card is by the wrap station."},
		{date(-5), "Policy", "Non-slip shoes are now checked at clock-in", "After two slips last month, shift leaders check for non-slip shoes at the start of every shift. Without them you'll be asked to change before you start."},
		{date(-4), "Equipment", "Fryer 2 was serviced", "Fryer 2's high-limit sensor was replaced. If it shows E-41 again, switch it off and tell the shift leader."},
		{date(-3), "Schedule", "Holiday hours", "We close at 18:00 on the 24th and are closed on the 25th. Shifts that day are already adjusted in the app."},
		{date(-2), "Team", "Shout-out to Lena", "Lena got a perfect score on last week's mystery shop. Drive-thru times are down to 2:51 on average."},
		{date(-1), "Operations", "New drive-thru headset system", "The drive-thru headsets were replaced. Hold the side button to talk; a quick training is on the break room tablet."},
	}
	r.guides = []Guide{
		{"FRY-02", "E-41", "The high-limit sensor tripped: the oil got too hot.", "Hot oil. Don't touch the vat, and wear the long gloves.", []string{"Switch the fryer off at the front switch.", "Wait 30 seconds.", "Press and hold the red reset button under the control panel for 3 seconds.", "Switch the fryer back on and check the display reads HEAT."}, 30, "E-41 comes back within an hour, or the oil smokes"},
		{"FRY-02", "E-07", "The oil level is low.", "Hot oil. Never top up a fryer that is above 200°F without the long gloves.", []string{"Switch the fryer to standby.", "Top up with fresh oil from the jug to the MIN line.", "Switch back to cook and wait for HEAT."}, 25, "the level drops again in the same shift: there may be a leak"},
		{"POS-1", "E-NET", "The register lost the network.", "None.", []string{"Check the blue cable is clicked into the back of the register.", "Tap Retry on the screen.", "If it still shows E-NET, restart the register from the side button and wait 20 seconds."}, 30, "the register is offline for more than 5 minutes"},
		{"ICE-1", "BIN", "The ice bin full sensor is blocked.", "Don't reach into the chute while it runs.", []string{"Scoop ice from the bin until it's below the sensor line.", "Wipe the sensor eye on the left wall with a clean, dry cloth.", "Wait a minute for the machine to start again."}, 20, "no ice is made within 15 minutes"},
	}
	r.menu = []map[string]any{
		{"item": "Harbor Burger", "contains": []string{"wheat", "milk", "egg", "sesame"}, "notes": "Quarter-pound patty, cheddar, house sauce, sesame bun"},
		{"item": "Spicy bean wrap", "contains": []string{"wheat", "milk", "egg"}, "notes": "New this week; black beans, pepper jack, chipotle mayo"},
		{"item": "Garden salad", "contains": []string{"milk"}, "notes": "Comes with ranch; vinaigrette is dairy-free"},
		{"item": "Seasoned fries", "contains": []string{}, "notes": "Cooked in shared oil with the fish bites"},
		{"item": "Fish bites", "contains": []string{"fish", "wheat"}, "notes": "Cod, beer-free batter"},
		{"item": "Vanilla shake", "contains": []string{"milk"}, "notes": "Small 12 oz, large 20 oz"},
		{"item": "Lemonade", "contains": []string{}, "notes": "Fresh every 4 hours; sweetened with cane sugar"},
	}
	return r
}

func (r *restaurant) routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /crew/schedule", r.schedule)
	mux.HandleFunc("POST /crew/callout-check", r.calloutCheck)
	mux.HandleFunc("GET /crew/coverage", r.coverage)
	mux.HandleFunc("POST /crew/swaps", r.swap)
	mux.HandleFunc("POST /crew/leader-notes", r.leaderNote)
	mux.HandleFunc("GET /crew/handbook", r.searchHandbook)
	mux.HandleFunc("GET /crew/updates", r.updatesSince)
	mux.HandleFunc("GET /crew/equipment", r.equipment)
	mux.HandleFunc("POST /crew/tickets", r.ticket)
	mux.HandleFunc("GET /crew/team", r.team)
	mux.HandleFunc("POST /crew/check-ins", r.checkIn)
	mux.HandleFunc("GET /crew/menu", r.menuItems)
	mux.HandleFunc("POST /crew/reflections", r.reflection)
	mux.HandleFunc("GET /crew/leader", r.leaderView)
}

func (r *restaurant) shiftView(s Shift) map[string]any {
	return map[string]any{
		"id": s.ID, "station": s.Station, "shift_leader": r.people[s.Leader].Name,
		"start": s.Start.Format("Mon 2 Jan 15:04"), "end": s.End.Format("15:04"),
	}
}

func (r *restaurant) schedule(w http.ResponseWriter, _ *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	mine := []map[string]any{}
	for _, s := range r.shifts {
		if s.Person == me {
			mine = append(mine, r.shiftView(s))
		}
	}
	answer(w, map[string]any{"shifts": mine})
}

// stopWords say nothing about which passage a question is about.
var stopWords = map[string]bool{
	"the": true, "and": true, "can": true, "are": true, "what": true, "how": true, "for": true,
	"with": true, "does": true, "you": true, "your": true, "our": true, "about": true, "rules": true,
	"policy": true, "own": true, "when": true, "need": true, "should": true, "there": true,
	"have": true, "has": true, "any": true, "get": true, "use": true, "much": true, "many": true,
}

var mustStayHome = []string{"vomit", "diarrhea", "fever", "jaundice", "sore throat"}

func (r *restaurant) calloutCheck(w http.ResponseWriter, req *http.Request) {
	var asked struct {
		ShiftID  string   `json:"shift_id"`
		Symptoms []string `json:"symptoms"`
	}
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil {
		http.Error(w, "a shift_id and symptoms are required", http.StatusBadRequest)
		return
	}
	shift, ok := r.shift(asked.ShiftID)
	if !ok {
		http.Error(w, "no shift "+asked.ShiftID, http.StatusNotFound)
		return
	}
	reportable := []string{}
	for _, symptom := range asked.Symptoms {
		if negated(symptom) {
			continue
		}
		for _, word := range mustStayHome {
			if strings.Contains(strings.ToLower(symptom), word) {
				reportable = append(reportable, symptom)
				break
			}
		}
	}
	hours := time.Until(shift.Start).Hours()
	cleared := len(reportable) == 0 && hours >= 4
	reason := "At least 4 hours' notice and no symptoms that need you home: you may find your own cover."
	switch {
	case len(reportable) > 0:
		reason = "With " + strings.Join(reportable, ", ") + " you must stay home until 24 hours after the last symptom. The shift leader arranges cover."
	case hours < 4:
		reason = "Less than 4 hours' notice: the shift leader arranges cover."
	}
	answer(w, map[string]any{
		"shift": r.shiftView(shift), "must_stay_home": len(reportable) > 0,
		"cleared_to_find_cover": cleared, "hours_until_shift": int(hours), "reason": reason,
		"policy": "Calling out sick",
	})
}

func (r *restaurant) shift(id string) (Shift, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, s := range r.shifts {
		if s.ID == id {
			return s, true
		}
	}
	return Shift{}, false
}

func (r *restaurant) coverage(w http.ResponseWriter, req *http.Request) {
	shift, ok := r.shift(req.URL.Query().Get("shift_id"))
	if !ok {
		http.Error(w, "no such shift", http.StatusNotFound)
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	free := []map[string]any{}
	ids := make([]string, 0, len(r.people))
	for id := range r.people {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		p := r.people[id]
		if !contains(p.Roles, shift.Station) {
			continue
		}
		busy := false
		for _, s := range r.shifts {
			if s.Person == id && s.Start.Before(shift.End) && shift.Start.Before(s.End) {
				busy = true
			}
		}
		if !busy {
			free = append(free, map[string]any{"id": id, "name": p.Name, "stations": p.Roles})
		}
	}
	answer(w, map[string]any{"shift": r.shiftView(shift), "available": free})
}

func (r *restaurant) swap(w http.ResponseWriter, req *http.Request) {
	var asked struct {
		ShiftID    string `json:"shift_id"`
		TeammateID string `json:"teammate_id"`
		Note       string `json:"note"`
	}
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil {
		http.Error(w, "a shift_id and a teammate_id are required", http.StatusBadRequest)
		return
	}
	shift, ok := r.shift(asked.ShiftID)
	teammate, known := r.people[asked.TeammateID]
	if !ok || !known {
		http.Error(w, "no such shift or teammate", http.StatusNotFound)
		return
	}
	item := r.add("swap", fmt.Sprintf("Swap: %s → %s", shift.Station+" "+shift.Start.Format("Mon 15:04"), teammate.Name),
		strings.TrimSpace("Asked "+teammate.Name+" to take the shift. "+asked.Note), "Waiting for "+teammate.Name)
	answer(w, map[string]any{"request_id": item.ID, "status": "sent to " + teammate.Name + "; the shift leader sees it too"})
}

func (r *restaurant) leaderNote(w http.ResponseWriter, req *http.Request) {
	var asked struct {
		Message string `json:"message"`
		ShiftID string `json:"shift_id"`
	}
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil || asked.Message == "" {
		http.Error(w, "a message is required", http.StatusBadRequest)
		return
	}
	to := "the shift leader"
	if shift, ok := r.shift(asked.ShiftID); ok {
		to = r.people[shift.Leader].Name
	}
	item := r.add("alert", "Note for "+to, asked.Message, "Sent")
	answer(w, map[string]any{"id": item.ID, "sent_to": to})
}

// searchHandbook answers with the passages that match best, as citations, which is the result
// the router shows as a reply's sources. Its text is all the agent gets, in each citation.
func (r *restaurant) searchHandbook(w http.ResponseWriter, req *http.Request) {
	passages, err := r.handbook.passages(req.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	words := strings.Fields(strings.ToLower(req.URL.Query().Get("q")))
	type scored struct {
		passage handbookPassage
		score   int
	}
	var found []scored
	for _, p := range passages {
		text := strings.ToLower(p.Heading + " " + p.Text)
		score := 0
		for _, word := range words {
			word = strings.Trim(word, "?.,!'\"")
			if len(word) < 3 || stopWords[word] {
				continue
			}
			if strings.Contains(text, word) {
				score++
			}
			if strings.Contains(strings.ToLower(p.Heading), word) {
				score += 2
			}
		}
		if score > 0 {
			found = append(found, scored{p, score})
		}
	}
	sort.SliceStable(found, func(i, j int) bool { return found[i].score > found[j].score })
	citations := []map[string]string{}
	for i, f := range found {
		if i == 3 || f.score*2 < found[0].score {
			break
		}
		citations = append(citations, f.passage.citation())
	}
	answer(w, map[string]any{"status": "answered", "citations": citations})
}

func (r *restaurant) updatesSince(w http.ResponseWriter, req *http.Request) {
	since := req.URL.Query().Get("since")
	found := []Update{}
	for _, u := range r.updates {
		if since == "" || u.Date >= since {
			found = append(found, u)
		}
	}
	answer(w, map[string]any{"updates": found})
}

func (r *restaurant) equipment(w http.ResponseWriter, req *http.Request) {
	asset := strings.ToUpper(strings.TrimSpace(req.URL.Query().Get("asset")))
	code := strings.ToUpper(strings.TrimSpace(req.URL.Query().Get("code")))
	known := []string{}
	for _, g := range r.guides {
		if g.Asset == asset && (code == "" || strings.ReplaceAll(g.Code, "-", "") == strings.ReplaceAll(code, "-", "")) {
			answer(w, map[string]any{"found": true, "guide": g})
			return
		}
		known = append(known, g.Asset+" "+g.Code)
	}
	answer(w, map[string]any{"found": false, "known_codes": known})
}

func (r *restaurant) ticket(w http.ResponseWriter, req *http.Request) {
	var asked struct {
		Asset    string `json:"asset"`
		Summary  string `json:"summary"`
		Details  string `json:"details"`
		Priority string `json:"priority"`
	}
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil || asked.Summary == "" {
		http.Error(w, "a summary is required", http.StatusBadRequest)
		return
	}
	title := asked.Summary
	if asked.Asset != "" && !strings.Contains(strings.ToUpper(title), strings.ToUpper(asked.Asset)) {
		title = asked.Asset + ": " + title
	}
	details := asked.Details
	if asked.Priority != "" {
		details = "Priority: " + asked.Priority + ". " + details
	}
	item := r.add("ticket", title, details, "Draft, waiting for "+r.people["taylor"].Name)
	answer(w, map[string]any{"ticket_id": item.ID, "status": "drafted for the owner to review and submit"})
}

func (r *restaurant) team(w http.ResponseWriter, _ *http.Request) {
	ids := []string{"taylor", "maya", "sam", "priya", "dev"}
	team := []Person{}
	for _, id := range ids {
		team = append(team, r.people[id])
	}
	answer(w, map[string]any{"team": team})
}

func (r *restaurant) checkIn(w http.ResponseWriter, req *http.Request) {
	var asked struct {
		With  string `json:"with"`
		When  string `json:"when"`
		Topic string `json:"topic"`
	}
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil || asked.With == "" || asked.When == "" {
		http.Error(w, "with and when are required", http.StatusBadRequest)
		return
	}
	name := asked.With
	if p, ok := r.people[strings.ToLower(asked.With)]; ok {
		name = p.Name
	}
	item := r.add("check_in", "1-on-1 with "+name, asked.When+". "+asked.Topic, "Booked")
	answer(w, map[string]any{"id": item.ID, "booked_with": name, "when": asked.When})
}

func (r *restaurant) menuItems(w http.ResponseWriter, _ *http.Request) {
	answer(w, map[string]any{"menu": r.menu})
}

func (r *restaurant) reflection(w http.ResponseWriter, req *http.Request) {
	var asked Reflection
	if err := json.NewDecoder(req.Body).Decode(&asked); err != nil {
		http.Error(w, "energy, themes and a win are expected", http.StatusBadRequest)
		return
	}
	asked.At = time.Now()
	r.mu.Lock()
	r.reflections = append(r.reflections, asked)
	r.mu.Unlock()
	answer(w, map[string]any{"saved": true, "shared_with_leaders": "themes and energy only, without your name"})
}

func (r *restaurant) leaderView(w http.ResponseWriter, _ *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	inbox := append([]InboxItem{}, r.inbox...)
	sort.SliceStable(inbox, func(i, j int) bool { return inbox[i].At.After(inbox[j].At) })
	themes := map[string]int{}
	energy := 0
	wins := []string{}
	for _, reflection := range r.reflections {
		energy += reflection.Energy
		for _, theme := range reflection.Themes {
			themes[strings.ToLower(theme)]++
		}
		if reflection.Win != "" {
			wins = append(wins, reflection.Win)
		}
	}
	pulse := map[string]any{"reflections": len(r.reflections), "themes": themes, "wins": wins}
	if len(r.reflections) > 0 {
		pulse["average_energy"] = float64(energy) / float64(len(r.reflections))
	}
	answer(w, map[string]any{"inbox": inbox, "pulse": pulse})
}

func (r *restaurant) add(kind, title, details, status string) InboxItem {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextID++
	item := InboxItem{
		ID: fmt.Sprintf("%s-%d", kind, r.nextID), Kind: kind, From: "Jordan Lee",
		Title: title, Details: details, Status: status, At: time.Now(),
	}
	r.inbox = append(r.inbox, item)
	return item
}

// negated says whether a symptom is one they said they don't have, such as "no fever".
func negated(symptom string) bool {
	lower := " " + strings.ToLower(strings.TrimSpace(symptom))
	for _, word := range []string{" no ", " not ", " without ", " never "} {
		if strings.Contains(lower, word) {
			return true
		}
	}
	return false
}

func contains(values []string, value string) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}
