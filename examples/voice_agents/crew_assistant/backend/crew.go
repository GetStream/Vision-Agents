package main

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Crew roles, kept in each person's Stream user custom data as crew_role. A leader can read
// the leaders' inbox.
const (
	roleLeader     = "leader"
	roleTrainer    = "trainer"
	roleTeamMember = "team_member"
)

// Person is someone on the team, by their Stream user id.
type Person struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	Title    string   `json:"role"`
	CrewRole string   `json:"-"`
	Stations []string `json:"stations,omitempty"`
	Note     string   `json:"note,omitempty"`
	Email    string   `json:"email,omitempty"`
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

// refusal is a request the restaurant turns down, with the HTTP status that says why.
type refusal struct {
	status  int
	message string
}

func (r refusal) Error() string { return r.message }

func refuse(status int, format string, args ...any) error {
	return refusal{status: status, message: fmt.Sprintf(format, args...)}
}

// restaurant is Harbor Grill: its people, schedule, handbook, equipment and the leaders' inbox.
// Everything a team member reads or writes is theirs, by the Stream user id that asks.
type restaurant struct {
	mu          sync.Mutex
	day         time.Time
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
	owner       = "crew-taylor"
	supervisor  = "crew-supervisor"
	trainer     = "crew-priya"
	buddy       = "crew-dev"
	handbookURL = "https://handbook.example.com/"
)

func newRestaurant(now time.Time) *restaurant {
	r := &restaurant{
		day:    time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.Local),
		people: map[string]Person{},
	}
	at, date := r.at, r.date
	for _, p := range []Person{
		{ID: "crew-maya", Name: "Maya Chen", Title: "Shift leader, front of house", CrewRole: roleLeader, Note: "Runs most morning shifts"},
		{ID: "crew-sam", Name: "Sam Ortiz", Title: "Shift leader, kitchen", CrewRole: roleLeader, Note: "Runs most evening shifts"},
		{ID: owner, Name: "Taylor Morgan", Title: "Owner", CrewRole: roleLeader, Note: "Approves repair tickets and schedules"},
		{ID: supervisor, Name: "Alex Rivera", Title: "Restaurant manager", CrewRole: roleLeader, Note: "Direct supervisor of every team member, for 1-on-1s about their role and growth"},
		{ID: trainer, Name: "Priya Patel", Title: "Trainer", CrewRole: roleTrainer, Note: "Trains new team members in their first 30 days"},
		{ID: buddy, Name: "Dev Harris", Title: "Team member", CrewRole: roleTeamMember, Stations: []string{"Drive-thru", "Register"}, Note: "Onboarding buddy for new team members"},
		{ID: "crew-jordan", Name: "Jordan Lee", Title: "Team member", CrewRole: roleTeamMember, Stations: []string{"Drive-thru", "Kitchen", "Register"}},
		{ID: "crew-lena", Name: "Lena Brooks", Title: "Team member", CrewRole: roleTeamMember, Stations: []string{"Drive-thru", "Kitchen"}},
		{ID: "crew-marco", Name: "Marco Ruiz", Title: "Team member", CrewRole: roleTeamMember, Stations: []string{"Kitchen"}},
		{ID: "crew-ava", Name: "Ava Kim", Title: "Team member", CrewRole: roleTeamMember, Stations: []string{"Register", "Drive-thru"}},
	} {
		first, _, _ := strings.Cut(strings.ToLower(p.Name), " ")
		p.Email = first + "@example.com"
		r.people[p.ID] = p
	}
	r.shifts = []Shift{
		{ID: "s-101", Person: "crew-jordan", Station: "Drive-thru", Start: at(1, 6), End: at(1, 14), Leader: "crew-maya"},
		{ID: "s-102", Person: "crew-jordan", Station: "Kitchen", Start: at(2, 16), End: at(2, 22), Leader: "crew-sam"},
		{ID: "s-103", Person: "crew-jordan", Station: "Register", Start: at(4, 10), End: at(4, 16), Leader: "crew-maya"},
		{ID: "s-201", Person: "crew-lena", Station: "Kitchen", Start: at(1, 10), End: at(1, 18), Leader: "crew-maya"},
		{ID: "s-202", Person: "crew-ava", Station: "Register", Start: at(2, 16), End: at(2, 22), Leader: "crew-sam"},
		{ID: "s-203", Person: buddy, Station: "Drive-thru", Start: at(4, 6), End: at(4, 14), Leader: "crew-maya"},
		{ID: "s-204", Person: "crew-marco", Station: "Kitchen", Start: at(1, 6), End: at(1, 14), Leader: "crew-maya"},
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
	r.nextID = 300
	return r
}

func (r *restaurant) at(days, hour int) time.Time {
	return r.day.AddDate(0, 0, days).Add(time.Duration(hour) * time.Hour)
}

func (r *restaurant) date(days int) string {
	return r.day.AddDate(0, 0, days).Format("2006-01-02")
}

// team is everyone on it, as their Stream users are kept.
func (r *restaurant) team() []Person {
	r.mu.Lock()
	defer r.mu.Unlock()
	team := make([]Person, 0, len(r.people))
	for _, p := range r.people {
		team = append(team, p)
	}
	sort.Slice(team, func(i, j int) bool { return team[i].ID < team[j].ID })
	return team
}

// enrol adds someone the restaurant hasn't met as a new hire, with a starter schedule, and says
// whether they were new.
func (r *restaurant) enrol(userID, name string) (Person, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if p, known := r.people[userID]; known {
		return p, false
	}
	if name == "" {
		name = userID
	}
	p := Person{ID: userID, Name: name, Title: "New team member", CrewRole: roleTeamMember, Stations: []string{"Register", "Drive-thru"}}
	r.people[userID] = p
	for _, s := range []Shift{
		{Station: "Register", Start: r.at(2, 10), End: r.at(2, 16), Leader: "crew-maya"},
		{Station: "Drive-thru", Start: r.at(3, 16), End: r.at(3, 22), Leader: "crew-sam"},
	} {
		r.nextID++
		s.ID, s.Person = fmt.Sprintf("s-%d", r.nextID), userID
		r.shifts = append(r.shifts, s)
	}
	return p, true
}

// member is the team member userID, who must be on the team.
func (r *restaurant) member(userID string) (Person, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.people[userID]
	if !ok {
		return Person{}, refuse(403, "%s isn't on the team", userID)
	}
	return p, nil
}

func (r *restaurant) shiftView(s Shift) map[string]any {
	return map[string]any{
		"id": s.ID, "station": s.Station, "shift_leader": r.people[s.Leader].Name,
		"start": s.Start.Format("Mon 2 Jan 15:04"), "end": s.End.Format("15:04"),
	}
}

func (r *restaurant) schedule(userID string) (any, error) {
	if _, err := r.member(userID); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	mine := []map[string]any{}
	for _, s := range r.shifts {
		if s.Person == userID {
			mine = append(mine, r.shiftView(s))
		}
	}
	return map[string]any{"shifts": mine}, nil
}

// ownShift is one of userID's own shifts.
func (r *restaurant) ownShift(userID, shiftID string) (Shift, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, s := range r.shifts {
		if s.ID == shiftID {
			if s.Person != userID {
				return Shift{}, refuse(403, "shift %s isn't yours", shiftID)
			}
			return s, nil
		}
	}
	return Shift{}, refuse(404, "no shift %s", shiftID)
}

// stopWords say nothing about which passage a question is about.
var stopWords = map[string]bool{
	"the": true, "and": true, "can": true, "are": true, "what": true, "how": true, "for": true,
	"with": true, "does": true, "you": true, "your": true, "our": true, "about": true, "rules": true,
	"policy": true, "own": true, "when": true, "need": true, "should": true, "there": true,
	"have": true, "has": true, "any": true, "get": true, "use": true, "much": true, "many": true,
}

var mustStayHome = []string{"vomit", "diarrhea", "fever", "jaundice", "sore throat"}

func (r *restaurant) calloutCheck(userID, shiftID string, symptoms []string) (any, error) {
	shift, err := r.ownShift(userID, shiftID)
	if err != nil {
		return nil, err
	}
	reportable := []string{}
	for _, symptom := range symptoms {
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
	r.mu.Lock()
	view := r.shiftView(shift)
	r.mu.Unlock()
	return map[string]any{
		"shift": view, "must_stay_home": len(reportable) > 0,
		"cleared_to_find_cover": cleared, "hours_until_shift": int(hours), "reason": reason,
		"policy": "Calling out sick",
	}, nil
}

func (r *restaurant) coverage(userID, shiftID string) (any, error) {
	shift, err := r.ownShift(userID, shiftID)
	if err != nil {
		return nil, err
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
		if id == userID || !contains(p.Stations, shift.Station) {
			continue
		}
		busy := false
		for _, s := range r.shifts {
			if s.Person == id && s.Start.Before(shift.End) && shift.Start.Before(s.End) {
				busy = true
			}
		}
		if !busy {
			free = append(free, map[string]any{"id": id, "name": p.Name, "stations": p.Stations})
		}
	}
	return map[string]any{"shift": r.shiftView(shift), "available": free}, nil
}

func (r *restaurant) swap(userID, shiftID, teammateID, note string) (any, error) {
	from, err := r.member(userID)
	if err != nil {
		return nil, err
	}
	shift, err := r.ownShift(userID, shiftID)
	if err != nil {
		return nil, err
	}
	teammate, err := r.member(teammateID)
	if err != nil || teammateID == userID {
		return nil, refuse(404, "no teammate %s to swap with", teammateID)
	}
	item := r.add(from, "swap", fmt.Sprintf("Swap: %s → %s", shift.Station+" "+shift.Start.Format("Mon 15:04"), teammate.Name),
		strings.TrimSpace("Asked "+teammate.Name+" to take the shift. "+note), "Waiting for "+teammate.Name)
	return map[string]any{"request_id": item.ID, "status": "sent to " + teammate.Name + "; the shift leader sees it too"}, nil
}

// absenceReasons are the reasons a leader's note gives for someone being out, so nothing they
// said about their health reaches it.
var absenceReasons = map[string]string{
	"sick":      "is out sick",
	"family":    "is out for a family matter",
	"transport": "can't make it in",
	"other":     "is out",
}

func (r *restaurant) leaderNote(userID, reason, shiftID string, coverAsked bool) (any, error) {
	from, err := r.member(userID)
	if err != nil {
		return nil, err
	}
	out, ok := absenceReasons[reason]
	if !ok {
		return nil, refuse(400, "reason must be sick, family, transport or other")
	}
	to, which := "the shift leader", ""
	if shift, err := r.ownShift(userID, shiftID); err == nil {
		r.mu.Lock()
		to = r.people[shift.Leader].Name
		r.mu.Unlock()
		which = " for " + shift.Station + " " + shift.Start.Format("Mon 15:04")
	}
	cover := "No cover asked for yet."
	if coverAsked {
		cover = "They asked a teammate to cover."
	}
	message := from.Name + " " + out + which + ". " + cover
	item := r.add(from, "alert", "Note for "+to, message, "Sent")
	return map[string]any{"id": item.ID, "sent_to": to, "note": message}, nil
}

// searchHandbook answers with the passages that match best, as citations, which is the result
// the router shows as a reply's sources. Its text is all the agent gets, in each citation.
func (r *restaurant) searchHandbook(ctx context.Context, query string) (any, error) {
	passages, err := r.handbook.passages(ctx)
	if err != nil {
		return nil, err
	}
	words := strings.Fields(strings.ToLower(query))
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
	return map[string]any{"status": "answered", "citations": citations}, nil
}

func (r *restaurant) updatesSince(since string) any {
	found := []Update{}
	for _, u := range r.updates {
		if since == "" || u.Date >= since {
			found = append(found, u)
		}
	}
	return map[string]any{"updates": found}
}

func (r *restaurant) equipment(asset, code string) any {
	asset = strings.ToUpper(strings.TrimSpace(asset))
	code = strings.ToUpper(strings.TrimSpace(code))
	known := []string{}
	for _, g := range r.guides {
		if g.Asset == asset && (code == "" || strings.ReplaceAll(g.Code, "-", "") == strings.ReplaceAll(code, "-", "")) {
			return map[string]any{"found": true, "guide": g}
		}
		known = append(known, g.Asset+" "+g.Code)
	}
	return map[string]any{"found": false, "known_codes": known}
}

func (r *restaurant) ticket(userID, asset, summary, details, priority string) (any, error) {
	from, err := r.member(userID)
	if err != nil {
		return nil, err
	}
	if summary == "" {
		return nil, refuse(400, "a summary is required")
	}
	title := summary
	if asset != "" && !strings.Contains(strings.ToUpper(title), strings.ToUpper(asset)) {
		title = asset + ": " + title
	}
	if priority != "" {
		details = "Priority: " + priority + ". " + details
	}
	r.mu.Lock()
	reviewer := r.people[owner].Name
	r.mu.Unlock()
	item := r.add(from, "ticket", title, details, "Draft, waiting for "+reviewer)
	return map[string]any{"ticket_id": item.ID, "status": "drafted for the owner to review and submit"}, nil
}

// leadership is who leads and trains the team, and the buddy who helps new team members.
// setSupervisor names the direct supervisor in place of the made-up one, with the email their
// invites go to. Either left empty keeps the made-up one's.
func (r *restaurant) setSupervisor(name, email string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	p := r.people[supervisor]
	if name != "" {
		p.Name = name
	}
	if email != "" {
		p.Email = email
	}
	r.people[supervisor] = p
}

func (r *restaurant) leadership(userID string) (any, error) {
	if _, err := r.member(userID); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	team := []Person{}
	for _, id := range []string{owner, supervisor, "crew-maya", "crew-sam", trainer, buddy} {
		if id != userID {
			team = append(team, r.people[id])
		}
	}
	return map[string]any{"team": team}, nil
}

func (r *restaurant) checkIn(userID, with, when, topic string) (any, error) {
	from, err := r.member(userID)
	if err != nil {
		return nil, err
	}
	if with == "" || when == "" {
		return nil, refuse(400, "with and when are required")
	}
	name, email := with, ""
	if p, ok := r.find(with); ok {
		name, email = p.Name, p.Email
	}
	item := r.add(from, "check_in", "1-on-1 with "+name, strings.TrimSpace(when+". "+topic), "Booked")
	return map[string]any{"id": item.ID, "booked_with": name, "email": email, "when": when}, nil
}

// find is the person named by id, by id without its crew- prefix, or by first or full name.
func (r *restaurant) find(named string) (Person, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	named = strings.ToLower(strings.TrimSpace(named))
	for _, p := range r.people {
		first, _, _ := strings.Cut(strings.ToLower(p.Name), " ")
		if p.ID == named || strings.TrimPrefix(p.ID, "crew-") == named || strings.ToLower(p.Name) == named || first == named {
			return p, true
		}
	}
	return Person{}, false
}

func (r *restaurant) menuItems() any {
	return map[string]any{"menu": r.menu}
}

func (r *restaurant) reflection(userID string, reflection Reflection) (any, error) {
	if _, err := r.member(userID); err != nil {
		return nil, err
	}
	if reflection.Energy < 1 || reflection.Energy > 5 {
		return nil, refuse(400, "energy from 1 to 5 is required")
	}
	reflection.At = time.Now()
	r.mu.Lock()
	r.reflections = append(r.reflections, reflection)
	r.mu.Unlock()
	return map[string]any{"saved": true, "shared_with_leaders": "themes, energy and the win only, without your name"}, nil
}

func (r *restaurant) leaderView() any {
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
	return map[string]any{"inbox": inbox, "pulse": pulse}
}

func (r *restaurant) add(from Person, kind, title, details, status string) InboxItem {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextID++
	item := InboxItem{
		ID: fmt.Sprintf("%s-%d", kind, r.nextID), Kind: kind, From: from.Name,
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

// statusOf is the HTTP status an error from the restaurant answers with.
func statusOf(err error) int {
	var refused refusal
	if errors.As(err, &refused) {
		return refused.status
	}
	return 502
}
