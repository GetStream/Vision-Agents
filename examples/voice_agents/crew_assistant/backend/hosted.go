package main

import (
	"context"
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// hostedAgent is the agent the restaurant's tools are hosted under: the router offers them to
// every session of it and sends their calls here.
type hostedAgent struct {
	name  string
	tools *tools.Registry
}

func (a hostedAgent) Name() string           { return a.name }
func (a hostedAgent) Tools() *tools.Registry { return a.tools }

type (
	none         struct{}
	calloutCheck struct {
		ShiftID  string   `json:"shift_id" schema:"The shift, from get_my_schedule"`
		Symptoms []string `json:"symptoms" schema:"What they said they have, in a few words each, such as \"fever\""`
	}
	coverage struct {
		ShiftID string `json:"shift_id" schema:"The shift to cover"`
	}
	leaderNote struct {
		Reason     string `json:"reason" enum:"sick,family,transport,other" schema:"Why they're out"`
		ShiftID    string `json:"shift_id,omitempty" schema:"The shift"`
		CoverAsked bool   `json:"cover_asked" schema:"Whether a teammate was asked to cover"`
	}
	handbookQuery struct {
		Query string `json:"query" schema:"What to look for, in a few words"`
	}
	updatesSince struct {
		Since string `json:"since" schema:"The first day, as YYYY-MM-DD"`
	}
	equipment struct {
		Asset string `json:"asset" schema:"The asset id"`
		Code  string `json:"code,omitempty" schema:"The error code it shows, such as E-41"`
	}
	swap struct {
		Note       string `json:"note,omitempty" schema:"A short note to the teammate"`
		ShiftID    string `json:"shift_id" schema:"The shift to give away"`
		TeammateID string `json:"teammate_id" schema:"The teammate, from find_coverage"`
	}
	ticket struct {
		Asset    string `json:"asset,omitempty" schema:"The asset id"`
		Details  string `json:"details" schema:"What they saw and already tried"`
		Priority string `json:"priority,omitempty" schema:"low, normal or urgent"`
		Summary  string `json:"summary" schema:"The problem, in a short line"`
	}
	checkIn struct {
		Topic string `json:"topic,omitempty" schema:"What it is about"`
		When  string `json:"when" schema:"When, such as \"Thursday 14:00\""`
		With  string `json:"with" schema:"Who, by id or name, from get_team"`
	}
	reflection struct {
		Energy int      `json:"energy" schema:"From 1 to 5"`
		Themes []string `json:"themes"`
		Win    string   `json:"win,omitempty" schema:"Their win, in a short line"`
	}
)

// hostedTools are the restaurant's tools, run here for whoever's turn called them.
//
// A simulation's conversation has no user, so its calls act for simulationUser. Left empty, a
// call that names no user is refused rather than acting for anyone.
//
// get_local_context is the phone's: a phone that declares it answers it. This one answers only
// in a session with no phone, with the server's clock.
//
// With a simulationUser, this also hosts the phone's three tools that ask the person first, which
// a simulation has no phone to run, and runs them without asking. A phone declares its own, which
// win, so they only ever run in a session with no phone.
func (s *server) hostedTools(simulationUser string) (*tools.Registry, error) {
	registry := tools.NewRegistry()
	whose := func(ctx context.Context) (string, error) {
		userID := ""
		if call := tools.ToolCallContextFrom(ctx); call != nil {
			userID = call.User.ID
		}
		if userID == "" {
			userID = simulationUser
		}
		if userID == "" {
			return "", errors.New("the call names no user")
		}
		return userID, s.join(ctx, userID)
	}
	forUser := func(run func(string) (any, error)) func(context.Context) (any, error) {
		return func(ctx context.Context) (any, error) {
			userID, err := whose(ctx)
			if err != nil {
				return nil, err
			}
			return run(userID)
		}
	}
	registered := errors.Join(
		tools.Register(registry, "get_local_context",
			"The person's current date, local time, timezone and language. Call it before answering anything that depends on today or the time.",
			func(context.Context, none) (any, error) {
				now := time.Now()
				zone, _ := now.Zone()
				return map[string]string{
					"language": "en-US", "local": now.Format(time.RFC1123), "timeZone": zone,
					"utc": now.UTC().Format(time.RFC3339), "source": "the restaurant's clock: no phone is connected",
				}, nil
			}),
		tools.Register(registry, "get_my_schedule",
			"The person's upcoming shifts, each with its id, station, start, end and shift leader.",
			func(ctx context.Context, _ none) (any, error) {
				return forUser(s.crew.schedule)(ctx)
			}),
		tools.Register(registry, "check_callout",
			"Whether the person may find their own cover for a shift, by the restaurant's call-out policy, given their symptoms. Call it before helping with coverage.",
			func(ctx context.Context, in calloutCheck) (any, error) {
				return forUser(func(userID string) (any, error) { return s.crew.calloutCheck(userID, in.ShiftID, in.Symptoms) })(ctx)
			}),
		tools.Register(registry, "find_coverage",
			"Teammates trained for the shift’s station and free at that time.",
			func(ctx context.Context, in coverage) (any, error) {
				return forUser(func(userID string) (any, error) { return s.crew.coverage(userID, in.ShiftID) })(ctx)
			}),
		tools.Register(registry, "notify_shift_leader",
			"Tells the shift's leader the person is out: why, from a fixed list, and whether cover was asked for. It never carries health details.",
			func(ctx context.Context, in leaderNote) (any, error) {
				return forUser(func(userID string) (any, error) {
					return s.crew.leaderNote(userID, in.Reason, in.ShiftID, in.CoverAsked)
				})(ctx)
			}),
		tools.Register(registry, "search_handbook",
			"Searches the team handbook: policies, uniform, food safety, allergens, breaks, swaps, time off, meals and values. The passages it finds are the only source for answers about them, and show as the reply's sources.",
			func(ctx context.Context, in handbookQuery) (any, error) {
				return s.crew.searchHandbook(ctx, in.Query)
			}),
		tools.Register(registry, "get_updates_since",
			"Announcements and team news since a date.",
			func(_ context.Context, in updatesSince) (any, error) { return s.crew.updatesSince(in.Since), nil }),
		tools.Register(registry, "get_equipment_guide",
			"The quick-fix guide for an error code on a piece of equipment, by its asset id, such as FRY-02, POS-1 or ICE-1.",
			func(_ context.Context, in equipment) (any, error) { return s.crew.equipment(in.Asset, in.Code), nil }),
		tools.Register(registry, "get_team",
			"The leadership team, the trainer and the onboarding buddy, with what each does.",
			func(ctx context.Context, _ none) (any, error) {
				return forUser(s.crew.leadership)(ctx)
			}),
		tools.Register(registry, "get_menu",
			"The menu, with what each item contains and notes, for questions and quizzes.",
			func(context.Context, none) (any, error) { return s.crew.menuItems(), nil }),
		tools.Register(registry, "save_shift_reflection",
			"Saves the end-of-shift reflection. Only the themes, energy and win reach the leaders, never who wrote it.",
			func(ctx context.Context, in reflection) (any, error) {
				return forUser(func(userID string) (any, error) {
					return s.crew.reflection(userID, Reflection{Energy: in.Energy, Themes: in.Themes, Win: in.Win})
				})(ctx)
			}),
	)
	if registered != nil || simulationUser == "" {
		return registry, registered
	}
	return registry, errors.Join(
		tools.Register(registry, "request_shift_swap",
			"Asks a teammate to take one of the person’s shifts. The phone asks the person first.",
			func(ctx context.Context, in swap) (any, error) {
				return forUser(func(userID string) (any, error) { return s.crew.swap(userID, in.ShiftID, in.TeammateID, in.Note) })(ctx)
			}),
		tools.Register(registry, "create_support_ticket",
			"Drafts a repair ticket for the owner to review and submit. The phone asks the person first.",
			func(ctx context.Context, in ticket) (any, error) {
				return forUser(func(userID string) (any, error) {
					return s.crew.ticket(userID, in.Asset, in.Summary, in.Details, in.Priority)
				})(ctx)
			}),
		tools.Register(registry, "book_check_in",
			"Books a 1-on-1 check-in with a leader or trainer. The phone asks the person first.",
			func(ctx context.Context, in checkIn) (any, error) {
				return forUser(func(userID string) (any, error) { return s.crew.checkIn(userID, in.With, in.When, in.Topic) })(ctx)
			}),
	)
}
