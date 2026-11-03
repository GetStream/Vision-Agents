package campaign

import (
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type Result string

const (
	Answered   Result = "answered"
	Unanswered Result = "unanswered"
	// Voicemail is a call an answering machine took. It uses the attempt like Unanswered, but
	// ringing the same machine again at once would only leave a second message.
	Voicemail     Result = "voicemail"
	InvalidNumber Result = "invalid_number"
	Refused       Result = "refused"
	Deferred      Result = "deferred"
	Paused        Result = "paused"
	// Canceled is a dial cut off by a shutdown before the vendor took it: nobody was rung.
	Canceled Result = "canceled"
)

const (
	reasonInvalidNumber = "invalid_number"
	reasonRefused       = "refused"
	reasonNoHours       = "no_calling_hours"
	reasonUnknownResult = "unknown_result"
	codeConcurrent      = "concurrent"
	concurrentRetry     = 5 * time.Minute
)

// Dialed is how one dial ended. Reason is the refusal or pause code.
type Dialed struct {
	Result       Result
	Reason       string
	CallID       string
	VendorCallID string
}

// afterDial says what follows a dial, which is 1 or 2. An unanswered first dial of a double
// dial writes nothing: the caller dials again at once.
func afterDial(sequence Sequence, hours CallingHours, zone *time.Location, contact store.Contact, dial int, dialed Dialed, started, now time.Time) (settle store.Settle, dialAgain bool) {
	step := sequence.First()
	attempt := max(contact.Attempt, 1)
	settle = store.Settle{StepID: step.ID, Attempt: contact.Attempt, CallID: dialed.CallID, VendorCallID: dialed.VendorCallID}
	if dial == 2 && (dialed.Result == Deferred || dialed.Result == Paused || dialed.Result == Refused || dialed.Result == Canceled) {
		// Dial 1 already rang the person, so the attempt is used whatever stopped dial 2.
		return unanswered(step, hours, zone, contact, started, settle), false
	}
	switch dialed.Result {
	case Answered:
		settle.Stage = store.AwaitingOutcome
		settle.Attempt = attempt
	case Unanswered:
		if dial == 1 && sequence.DoubleDial && dialed.Reason != reasonNoSession {
			return store.Settle{}, true
		}
		return unanswered(step, hours, zone, contact, started, settle), false
	case Voicemail:
		return unanswered(step, hours, zone, contact, started, settle), false
	case Deferred:
		at := now.Add(concurrentRetry)
		if dialed.Reason != codeConcurrent {
			local := now.In(zone)
			at = time.Date(local.Year(), local.Month(), local.Day()+1, 0, 0, 0, 0, zone)
		}
		return enrolledAt(hours, zone, at, settle), false
	case Paused, Canceled:
		settle.Stage = store.Enrolled
		settle.NextAttemptAt = &now
	case Refused:
		settle.Stage = store.Failed
		settle.Reason = dialed.Reason
	case InvalidNumber:
		settle.Stage = store.Failed
		settle.Reason = reasonInvalidNumber
	default:
		settle.Stage = store.Failed
		settle.Reason = reasonUnknownResult
	}
	return settle, false
}

// afterSilence is for a call that was answered but nobody spoke. It counts as unanswered,
// and a double dial would only ring the same quiet line again.
func afterSilence(sequence Sequence, hours CallingHours, zone *time.Location, contact store.Contact, started, now time.Time) store.Settle {
	settle := store.Settle{StepID: sequence.First().ID, Attempt: contact.Attempt}
	return unanswered(sequence.First(), hours, zone, contact, started, settle)
}

// afterOutcome ends a contact the way the step's route for the outcome says. An id the step
// does not define follows the route for other answers.
func afterOutcome(step Step, contact store.Contact, outcomeID string) store.Settle {
	next := step.Other.Next
	if outcome, ok := step.Find(outcomeID); ok {
		next = outcome.Next
	} else {
		outcomeID = OutcomeOther
	}
	return store.Settle{
		Stage:     store.ContactCompleted,
		StepID:    step.ID,
		Attempt:   max(contact.Attempt, 1),
		Outcome:   outcomeID,
		Converted: next.End == EndSuccessful,
	}
}

// resultOf maps the error of a dial. It looks through wrapping.
func resultOf(err error) Dialed {
	if errors.Is(err, phone.ErrNotPlaced) {
		return Dialed{Result: Canceled}
	}
	if code := pausing(err); code != "" {
		return Dialed{Result: Paused, Reason: code}
	}
	switch {
	case errors.Is(err, dlc.ErrRefused):
		code := dlc.RefusalCode(err)
		if dlc.RefusalResets(code) {
			return Dialed{Result: Deferred, Reason: code}
		}
		if code == "" {
			code = reasonRefused
		}
		return Dialed{Result: Refused, Reason: code}
	case errors.Is(err, phone.ErrNotDialable):
		return Dialed{Result: InvalidNumber, Reason: reasonInvalidNumber}
	}
	return Dialed{Result: Unanswered}
}

func unanswered(step Step, hours CallingHours, zone *time.Location, contact store.Contact, started time.Time, settle store.Settle) store.Settle {
	attempt := max(contact.Attempt, 1)
	if attempt >= len(step.Attempts) {
		settle.Stage = store.ContactCompleted
		settle.Attempt = attempt
		settle.Outcome = OutcomeNoResponse
		return settle
	}
	settle.Attempt = attempt + 1
	return enrolledAt(hours, zone, started.Add(time.Duration(step.Attempts[attempt].DelayMinutes)*time.Minute), settle)
}

// enrolledAt waits for the first opening at or after at. With none in sight the contact
// fails rather than carrying no time at all.
func enrolledAt(hours CallingHours, zone *time.Location, at time.Time, settle store.Settle) store.Settle {
	next, ok := hours.Next(at, zone)
	if !ok {
		settle.Stage = store.Failed
		settle.Reason = reasonNoHours
		return settle
	}
	settle.Stage = store.Enrolled
	settle.NextAttemptAt = &next
	return settle
}
