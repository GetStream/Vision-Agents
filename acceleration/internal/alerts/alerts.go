// Package alerts checks named conditions in the process that can see them, and sends one
// Sentry event per condition that holds at most once per interval, which Sentry's alert rules
// route by the usage_alert tag. Without a Sentry DSN an alert is only logged, at ERROR, which
// it always is too.
package alerts

import (
	"context"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/getsentry/sentry-go"
)

// Level is how loud an alert is.
type Level string

const (
	// Warn is for a person to look at today.
	Warn Level = "warn"
	// Page is for a person to look at now.
	Page Level = "page"
)

// sentryLevel is the level a Sentry event of the alert is sent at.
func (l Level) sentryLevel() sentry.Level {
	if l == Page {
		return sentry.LevelFatal
	}
	return sentry.LevelError
}

// Check reports whether a condition holds at now, and the numbers to send with it.
type Check func(now time.Time) (holds bool, extra map[string]any)

// ItemCheck reports the items a condition holds for at now, such as the models whose usage
// went missing, each with the numbers to send.
type ItemCheck func(now time.Time) map[string]map[string]any

// Condition is one named thing that should never be true.
type Condition struct {
	// Name is the alert's name, which Sentry's rules and the fingerprint use.
	Name  string
	Level Level
	Check Check
	// Items, in place of Check, is a condition that holds for items not known in advance. Each
	// item is its own alert, fingerprinted usage_<name>:<item> and limited on its own, so one
	// item that keeps failing does not hide another.
	Items ItemCheck
}

// Options configure a Loop.
type Options struct {
	// Hub sends the events; nil is sentry.CurrentHub().
	Hub    *sentry.Hub
	Logger *slog.Logger
	// Source names the process in the alert's source tag, such as "usage/<pod>".
	Source string
	// Tags are added to every event, such as the router's shard.
	Tags map[string]string
	// Every is how often the conditions are checked; defaults to 15 s.
	Every time.Duration
	// Limit is the least time between two events of one condition; defaults to a minute.
	Limit time.Duration
	// Now is the loop's clock; nil is time.Now.
	Now func() time.Time
}

// Loop checks its conditions every interval until it is stopped.
type Loop struct {
	options    Options
	conditions []Condition
	sent       map[string]time.Time
}

// New returns a loop over conditions; Run starts it.
func New(options Options, conditions ...Condition) *Loop {
	if options.Hub == nil {
		options.Hub = sentry.CurrentHub()
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Every == 0 {
		options.Every = 15 * time.Second
	}
	if options.Limit == 0 {
		options.Limit = time.Minute
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Loop{options: options, conditions: conditions, sent: map[string]time.Time{}}
}

// Add registers more conditions. Call it before Run.
func (l *Loop) Add(conditions ...Condition) {
	l.conditions = append(l.conditions, conditions...)
}

// Run checks every condition at once and then every interval, until ctx is done, and once more
// then: what a process's last moments did, such as the rows its shutdown lost, is still sent.
func (l *Loop) Run(ctx context.Context) {
	ticker := time.NewTicker(l.options.Every)
	defer ticker.Stop()
	for {
		l.check()
		select {
		case <-ctx.Done():
			l.check()
			return
		case <-ticker.C:
		}
	}
}

// check sends each condition that holds and was not sent within the limit.
func (l *Loop) check() {
	now := l.options.Now()
	for _, condition := range l.conditions {
		if condition.Items != nil {
			l.checkItems(now, condition)
			continue
		}
		holds, extra := condition.Check(now)
		if !holds {
			continue
		}
		if last, ok := l.sent[condition.Name]; ok && now.Sub(last) < l.options.Limit {
			continue
		}
		l.sent[condition.Name] = now
		send(l.options.Hub, l.options.Logger, l.options.Source, l.options.Tags, condition.Name, "", condition.Level, extra)
	}
}

// checkItems sends each item the condition holds for that was not sent within the limit, in
// order so the events of one check go out the same way every time.
func (l *Loop) checkItems(now time.Time, condition Condition) {
	held := condition.Items(now)
	for _, item := range slices.Sorted(maps.Keys(held)) {
		key := condition.Name + ":" + item
		if last, ok := l.sent[key]; ok && now.Sub(last) < l.options.Limit {
			continue
		}
		l.sent[key] = now
		send(l.options.Hub, l.options.Logger, l.options.Source, l.options.Tags, condition.Name, item, condition.Level, held[item])
	}
}

// Send sends one alert now, for a short-lived process (a job) that has no loop. It goes to
// the current hub, tagged with the binary's name; the caller flushes Sentry before it exits.
func Send(name string, level Level, extra map[string]any) {
	SendItem(name, "", level, extra)
}

// SendItem is Send for one item, such as the hour a job missed: the alert names it, and one
// item's alert is kept apart from another's.
func SendItem(name, item string, level Level, extra map[string]any) {
	send(sentry.CurrentHub(), slog.Default(), filepath.Base(os.Args[0]), nil, name, item, level, extra)
}

// send sends one alert, of one item of the condition when item is not empty.
func send(hub *sentry.Hub, logger *slog.Logger, source string, tags map[string]string, name, item string, level Level, extra map[string]any) {
	fields := []any{"alert", name, "level", level, "source", source, "extra", extra}
	if item != "" {
		fields = append(fields, "item", item)
	}
	logger.Error("usage alert", fields...)
	event := sentry.NewEvent()
	event.Level = level.sentryLevel()
	event.Message = "usage alert: " + name
	event.Fingerprint = []string{"usage_alert:" + name}
	event.Tags = map[string]string{}
	for key, value := range tags {
		event.Tags[key] = value
	}
	if item != "" {
		event.Message += " " + item
		event.Fingerprint = []string{"usage_" + name + ":" + item}
		event.Tags["item"] = item
	}
	event.Tags["usage_alert"] = name
	event.Tags["source"] = source
	// The SDK has no extra data any more; a context is where Sentry shows the numbers.
	event.Contexts = map[string]sentry.Context{"usage_alert": extra}
	hub.CaptureEvent(event)
}
