package campaign

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"unicode/utf8"
)

const (
	EndSuccessful   = "successful"
	EndUnsuccessful = "unsuccessful"
)

const (
	OutcomeSuccess       = "success"
	OutcomeNotInterested = "not_interested"
	OutcomeDoNotCall     = "do_not_call"
	OutcomeOther         = "other"
	OutcomeNoResponse    = "no_response"
)

const (
	VoicemailOff          = "off"
	VoicemailHangUp       = "hang_up"
	VoicemailLeaveMessage = "leave_message"
)

const (
	stepOne          = "step_1"
	maxAttempts      = 10
	MaxGoal          = 2000
	MaxVoicemail     = 1000
	doNotCallName    = "Do not call"
	doNotCallMeaning = "The person asked not to be called or contacted again."
)

var outcomeID = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)

// Sequence is what a campaign does with each contact. Only one step is allowed for now.
type Sequence struct {
	Steps      []Step `json:"steps"`
	DoubleDial bool   `json:"double_dial"`
}

type Step struct {
	ID         string    `json:"id"`
	ConfigID   string    `json:"config_id"`
	Goal       string    `json:"goal,omitempty"`
	Attempts   []Attempt `json:"attempts"`
	Outcomes   []Outcome `json:"outcomes"`
	Other      Route     `json:"other"`
	NoResponse Route     `json:"no_response"`
	// Voicemail is what the agent does when an answering machine picks up. A step saved
	// before it existed has none, which is off.
	Voicemail VoicemailSetting `json:"voicemail,omitzero"`
}

// VoicemailSetting is Action off (or empty), which does nothing special on a machine, hang_up,
// or leave_message, which says Message word for word.
type VoicemailSetting struct {
	Action  string `json:"action,omitempty"`
	Message string `json:"message,omitempty"`
}

func (v VoicemailSetting) off() bool { return v.Action == "" || v.Action == VoicemailOff }

func (v VoicemailSetting) hangsUp() bool { return v.Action != VoicemailLeaveMessage }

type Attempt struct {
	DelayMinutes int `json:"delay_minutes"`
}

type Outcome struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Next        Next   `json:"next"`
}

type Route struct {
	Next Next `json:"next"`
}

type Next struct {
	End  string `json:"end,omitempty"`
	Step string `json:"step,omitempty"` // reserved for branching; Validate refuses it
}

// DefaultSequence is what a new draft starts with: four attempts over twenty days.
func DefaultSequence(configID string) Sequence {
	return Sequence{
		Steps: []Step{{
			ID:       stepOne,
			ConfigID: configID,
			Attempts: []Attempt{{0}, {4320}, {10080}, {14400}},
			Outcomes: []Outcome{
				{ID: OutcomeSuccess, Name: "Success", Description: "The person agreed to what the call was about", Next: Next{End: EndSuccessful}},
				{ID: OutcomeNotInterested, Name: "Not interested", Description: "The person declined", Next: Next{End: EndUnsuccessful}},
			},
			Other:      Route{Next: Next{End: EndUnsuccessful}},
			NoResponse: Route{Next: Next{End: EndUnsuccessful}},
		}},
		DoubleDial: true,
	}
}

// LegacySequence is what campaigns made before sequences got: one attempt, no outcomes.
// It must marshal as the migration wrote it, with "outcomes": [] and not null.
func LegacySequence(configID string) Sequence {
	return Sequence{
		Steps: []Step{{
			ID:         stepOne,
			ConfigID:   configID,
			Attempts:   []Attempt{{0}},
			Outcomes:   []Outcome{},
			Other:      Route{Next: Next{End: EndUnsuccessful}},
			NoResponse: Route{Next: Next{End: EndUnsuccessful}},
		}},
	}
}

func ParseSequence(raw []byte) (Sequence, error) {
	var s Sequence
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return Sequence{}, fmt.Errorf("campaign: sequence: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return Sequence{}, errors.New("campaign: sequence: unexpected data after the JSON value")
	}
	return s, nil
}

// Validate checks the shape, which is what a PATCH may store.
func (s Sequence) Validate() error {
	if len(s.Steps) != 1 {
		return errors.New("a sequence has exactly one step for now")
	}
	return s.Steps[0].validate()
}

// Ready is Validate plus what a launch needs.
func (s Sequence) Ready() error {
	if err := s.Validate(); err != nil {
		return err
	}
	step := s.First()
	if step.ConfigID == "" {
		return errors.New("the step needs an agent config before the campaign can launch")
	}
	if strings.TrimSpace(step.Goal) == "" {
		return errors.New("the conversation needs a goal before the campaign can launch")
	}
	if step.Voicemail.Action == VoicemailLeaveMessage && strings.TrimSpace(step.Voicemail.Message) == "" {
		return errors.New("the voicemail needs a message to leave before the campaign can launch")
	}
	for _, o := range step.Outcomes {
		if o.Name == "" {
			return fmt.Errorf("the outcome %q needs a name before the campaign can launch", o.ID)
		}
		if o.Description == "" {
			return fmt.Errorf("the outcome %q needs a description before the campaign can launch", o.ID)
		}
	}
	return nil
}

func (s Sequence) First() Step {
	if len(s.Steps) == 0 {
		return Step{}
	}
	return s.Steps[0]
}

// Find looks an outcome up by id, the built-in do_not_call included.
func (st Step) Find(id string) (Outcome, bool) {
	if id == OutcomeDoNotCall {
		return doNotCall(), true
	}
	for _, o := range st.Outcomes {
		if o.ID == id {
			return o, true
		}
	}
	return Outcome{}, false
}

// Choices are the outcomes the classifier picks from.
func (st Step) Choices() []Outcome {
	out := make([]Outcome, 0, len(st.Outcomes)+1)
	out = append(out, st.Outcomes...)
	return append(out, doNotCall())
}

func doNotCall() Outcome {
	return Outcome{ID: OutcomeDoNotCall, Name: doNotCallName, Description: doNotCallMeaning, Next: Next{End: EndUnsuccessful}}
}

func (st Step) validate() error {
	if st.ID != stepOne {
		return fmt.Errorf("the step must be called %s", stepOne)
	}
	if utf8.RuneCountInString(st.Goal) > MaxGoal {
		return fmt.Errorf("the goal is longer than %d characters", MaxGoal)
	}
	if n := len(st.Attempts); n < 1 || n > maxAttempts {
		return fmt.Errorf("a step needs between 1 and %d attempts", maxAttempts)
	}
	for i, a := range st.Attempts {
		if i == 0 && a.DelayMinutes != 0 {
			return errors.New("the first attempt must have a delay of 0 minutes")
		}
		if i > 0 && a.DelayMinutes < 1 {
			return fmt.Errorf("attempt %d needs a delay of at least 1 minute", i+1)
		}
	}
	switch st.Voicemail.Action {
	case "", VoicemailOff, VoicemailHangUp, VoicemailLeaveMessage:
	default:
		return fmt.Errorf("the voicemail action must be %q, %q or %q", VoicemailOff, VoicemailHangUp, VoicemailLeaveMessage)
	}
	if utf8.RuneCountInString(st.Voicemail.Message) > MaxVoicemail {
		return fmt.Errorf("the voicemail message is longer than %d characters", MaxVoicemail)
	}
	seen := map[string]bool{}
	for _, o := range st.Outcomes {
		switch o.ID {
		case OutcomeDoNotCall, OutcomeOther, OutcomeNoResponse:
			return fmt.Errorf("the outcome id %q is built in and cannot be used", o.ID)
		}
		if !outcomeID.MatchString(o.ID) {
			return fmt.Errorf("the outcome id %q must start with a lowercase letter and use only lowercase letters, digits and underscores, up to 64 characters", o.ID)
		}
		if seen[o.ID] {
			return fmt.Errorf("the outcome id %q is used twice", o.ID)
		}
		seen[o.ID] = true
		if err := o.Next.validate(fmt.Sprintf("the outcome %q", o.ID)); err != nil {
			return err
		}
	}
	if err := st.Other.Next.validate("the route for other answers"); err != nil {
		return err
	}
	if err := st.NoResponse.Next.validate("the no response route"); err != nil {
		return err
	}
	if st.NoResponse.Next.End != EndUnsuccessful {
		return errors.New("the no response route must end unsuccessfully")
	}
	return nil
}

func (n Next) validate(what string) error {
	if n.Step != "" {
		return fmt.Errorf("%s cannot go to another step: branching is not supported yet", what)
	}
	if n.End != EndSuccessful && n.End != EndUnsuccessful {
		return fmt.Errorf("%s must end as %q or %q", what, EndSuccessful, EndUnsuccessful)
	}
	return nil
}
