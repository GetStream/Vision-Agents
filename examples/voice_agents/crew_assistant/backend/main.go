// Command backend is the half of the crew assistant that stays up on a server.
//
// It stands in for the restaurant's own systems: the schedule, the team, the handbook, the
// equipment guides, the repair tickets and the leaders' inbox. It hosts the agent's tools that
// read and write them on the router, which sends it their calls with the user who asked, and it
// answers the phone for the three tools the phone asks the person about first, and for the
// leaders' inbox. Everything lives in memory and starts from the same restaurant each time.
//
// Who is who is the Stream user: every request but the token carries a Stream token this signed,
// and each person's crew role, which says who leads, is kept in their Stream user's custom data.
// The token itself is signed for whichever user the phone names, which stands in for the
// restaurant's own sign-in. Someone it hasn't met is enrolled as a new hire.
//
// It also makes the agent speak first. A clock-in, a clock-out or a return from time off is an
// event the restaurant's systems would send: here the app sends it, and this creates a
// scheduled task for the person's conversation and runs it at once, so the agent's message is
// the first thing in it.
//
//	STREAM_API_KEY=<the key of the Stream app the agent runs in> \
//	STREAM_API_SECRET=<its secret> \
//	GOWORK=off go run .
//
// It reaches Stream's hosted router. STREAM_ACCELERATION_URL, with
// STREAM_ACCELERATION_CUSTOMER_ID for one without keys, points it at a router of your own.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// agentName is the agent configure stored from this directory.
const agentName = "crew_assistant"

func main() {
	addr := flag.String("addr", ":8097", "where to listen")
	simulationUser := flag.String("simulation-user", "", "the user a tool call that names none acts for, as a simulation's do; empty refuses them")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *addr, *simulationUser); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func run(ctx context.Context, addr, simulationUser string) error {
	key, secret := os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET")
	if key == "" || secret == "" {
		return errors.New("STREAM_API_KEY and STREAM_API_SECRET are needed: the Stream app the router runs in")
	}
	roles, err := newStreamRoles(key, secret)
	if err != nil {
		return err
	}
	crew := newRestaurant(time.Now())
	crew.setSupervisor(os.Getenv("CREW_SUPERVISOR_NAME"), os.Getenv("CREW_SUPERVISOR_EMAIL"))
	if err := roles.publish(ctx, crew.team()); err != nil {
		return fmt.Errorf("keeping the crew roles on the Stream users: %w", err)
	}
	s := &server{secret: []byte(secret), mint: roles.mint, crew: crew, roles: roles, speak: agentSpeaksFirst}

	hosted, err := s.hostedTools(simulationUser)
	if err != nil {
		return err
	}
	dispatch, err := stream.NewDispatch(stream.DispatchOptions{})
	if err != nil {
		return err
	}
	dispatch.Host(hostedAgent{name: agentName, tools: hosted}, 30*time.Second)

	mux := http.NewServeMux()
	s.routes(mux)
	server := &http.Server{Addr: addr, Handler: logged(mux), ReadHeaderTimeout: 5 * time.Second}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	hosting := make(chan error, 1)
	go func() {
		hosting <- dispatch.Run(ctx)
		cancel()
	}()
	go func() {
		<-ctx.Done()
		closing, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		_ = server.Shutdown(closing)
	}()

	log.Printf("serving Harbor Grill for %s on %s, with its tools hosted on the router", agentName, addr)
	served := server.ListenAndServe()
	cancel()
	if err := <-hosting; err != nil && !errors.Is(err, context.Canceled) {
		return fmt.Errorf("hosting the tools: %w", err)
	}
	return served
}

// events are what the restaurant's systems tell the agent about, by kind, and what the agent
// is told to do when one comes in.
var events = map[string]struct{ title, instructions string }{
	"clock_in": {
		title: "Pre-shift fit check",
		instructions: "The person is about to clock in. Greet them by first name, look up their next " +
			"shift with get_my_schedule, and give them the pre-shift fit check as interactive UI: " +
			"the uniform checklist and the focus areas for that shift.",
	},
	"clock_out": {
		title: "End of shift",
		instructions: "The person just clocked out. Thank them for the shift and give them the " +
			"end-of-shift reflection as interactive UI: a win, a challenge, a guest moment and " +
			"their energy.",
	},
	"back_from_time_off": {
		title: "While you were out",
		instructions: "The person is back from time off that started six days ago. Welcome them " +
			"back, call get_local_context, then get_updates_since with the day their time off " +
			"started, and summarize what they missed: what changes for them first, then team news.",
	},
	"onboarding": {
		title: "Your first 30 days",
		instructions: "The person started at Harbor Grill this week. Welcome them warmly, introduce " +
			"the leadership team and their onboarding buddy with get_team, say what their first " +
			"week looks like in three short points, and offer a quick menu quiz or booking a " +
			"check-in with their trainer.",
	},
}

// agentSpeaksFirst has the agent open userID's conversation about an event, by creating a
// scheduled task in it and running it now.
func agentSpeaksFirst(ctx context.Context, userID, sessionID, kind, timezone string) (string, error) {
	event := events[kind]
	client, err := stream.Backend{ActingFor: userID}.Client()
	if err != nil {
		return "", err
	}
	in := "P1D"
	var zone *string
	if timezone != "" {
		zone = &timezone
	}
	created, err := client.CreateScheduledTaskWithResponse(ctx,
		&acceleration.CreateScheduledTaskParams{XStreamUserId: &userID},
		acceleration.CreateScheduledTaskJSONRequestBody{
			Title:        event.title,
			Instructions: event.instructions,
			Schedule:     acceleration.ScheduledTaskSchedule{In: &in, Timezone: zone},
			SessionId:    &sessionID,
		})
	if err != nil {
		return "", err
	}
	task := created.JSON201
	if task == nil {
		return "", errors.New(refused("creating the task", created.StatusCode(), created.Body))
	}
	ran, err := client.RunScheduledTaskWithResponse(ctx, task.Id)
	if err != nil {
		return "", err
	}
	if ran.StatusCode() >= http.StatusMultipleChoices {
		return "", errors.New(refused("running the task", ran.StatusCode(), ran.Body))
	}
	// The task was only a way to run once now: it goes once its run is well under way.
	go func(id string) {
		time.Sleep(2 * time.Minute)
		if deleted, err := client.DeleteScheduledTaskWithResponse(context.Background(), id); err != nil || deleted.StatusCode() >= http.StatusMultipleChoices {
			log.Printf("the %s task %s stays scheduled: %v", kind, id, err)
		}
	}(task.Id)
	return task.Id, nil
}

// refused says what the router answered instead of the success it was asked for.
func refused(doing string, status int, body []byte) string {
	said := strings.TrimSpace(string(body))
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error.Message != "" {
		said = envelope.Error.Message
	}
	return fmt.Sprintf("%s (%d): %s", doing, status, said)
}
