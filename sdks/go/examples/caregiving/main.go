// Command caregiving is a text agent that helps a family find care: it answers common
// questions from its knowledge base, remembers what the family told it, and searches for and
// books caregivers through two functions that run in this process.
//
// Memory is kept per user_id, so the second conversation knows what the first was told. mem0
// learns from a conversation a few seconds after it ends, so recall is best on a second run.
//
//	cd sdks/go/examples/caregiving
//	STREAM_API_KEY=your_api_key STREAM_API_SECRET=your_api_secret go run .
//
// `getstream agents test caregiving` from this folder syncs the same agent and tests its
// knowledge and skills, but not find_caregivers or book_interview: they only exist while this
// program runs.
//
// Against a local router, which needs TURBOPUFFER_API_KEY, EXA_API_KEY and MEM0_API_KEY:
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 STREAM_ACCELERATION_CUSTOMER_ID=acme go run .
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx); err != nil && ctx.Err() == nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	agent, err := agents.New(agents.Options{
		Name:         "caregiving",
		MemoryFilter: map[string]string{"user_id": "family-42"},
	})
	if err != nil {
		return err
	}

	err = tools.Register(agent.Tools(), "find_caregivers", "Find caregivers available near a ZIP code",
		findCaregivers)
	if err != nil {
		return err
	}
	calendar := &bookings{byKey: map[string]string{}}
	err = tools.Register(agent.Tools(), "book_interview", "Book an interview with a caregiver",
		calendar.book)
	if err != nil {
		return err
	}

	if _, err := agent.Sync(ctx); err != nil {
		return err
	}

	conversations := [][]string{
		{"Hi! I need after-school care for my 8 year old. What should I ask in an interview?"},
		{
			"My mom needs help a few days a week too. How does that cost compare to what I asked about last time?",
			"Who is available for her in 94110?",
			"Book the first one for Thursday at 4pm.",
		},
	}
	for _, questions := range conversations {
		if err := converse(ctx, agent, questions); err != nil {
			return err
		}
	}
	return nil
}

type FindCaregivers struct {
	CareFor string `json:"care_for" schema:"who the care is for" enum:"child,senior"`
	ZIP     string `json:"zip" schema:"the family's 5-digit ZIP code" pattern:"^[0-9]{5}$"`
}

type Caregiver struct {
	ID              string `json:"id"`
	Name            string `json:"name"`
	HourlyRate      int    `json:"hourly_rate"`
	YearsExperience int    `json:"years_experience"`
}

var caregivers = map[string][]Caregiver{
	"child":  {{"cg_101", "Maya Lopez", 24, 6}, {"cg_102", "Sam Okafor", 21, 3}},
	"senior": {{"cg_201", "Grace Kim", 27, 11}, {"cg_202", "Luis Ortega", 25, 5}},
}

func findCaregivers(_ context.Context, in FindCaregivers) (any, error) {
	return caregivers[in.CareFor], nil
}

type BookInterview struct {
	CaregiverID string `json:"caregiver_id" schema:"the id find_caregivers returned, e.g. cg_101"`
	Time        string `json:"time" schema:"when, as the family said it, e.g. Thursday 4pm"`
}

// bookings stands in for your booking system. The call's id is the idempotency key, since the
// router sends a call again with the same id after a reconnect.
type bookings struct {
	mu    sync.Mutex
	byKey map[string]string
}

func (b *bookings) book(ctx context.Context, in BookInterview) (any, error) {
	toolCallContext := tools.ToolCallContextFrom(ctx)
	toolCallContext.Update(ctx, "Checking the caregiver's calendar")

	b.mu.Lock()
	defer b.mu.Unlock()
	if confirmation, ok := b.byKey[toolCallContext.ID]; ok {
		return confirmation, nil
	}
	confirmation := fmt.Sprintf("Booked an interview with %s on %s. Confirmation number IV-%d.",
		in.CaregiverID, in.Time, 1001+len(b.byKey))
	b.byKey[toolCallContext.ID] = confirmation
	return confirmation, nil
}

func converse(ctx context.Context, agent *agents.Agent, questions []string) error {
	session, err := agent.Chat(ctx)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))

	for _, question := range questions {
		fmt.Println("> " + question)
		answer, err := ask(ctx, session, question)
		if err != nil {
			return err
		}
		fmt.Println(answer)
		fmt.Println()
	}
	return nil
}

func ask(ctx context.Context, session *agents.Session, text string) (string, error) {
	if _, err := session.Responses.Create(ctx, text); err != nil {
		return "", err
	}
	var answer strings.Builder
	events := session.Events()
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case event, ok := <-events:
			if !ok {
				return "", errors.New("caregiving: the conversation ended before the answer did")
			}
			switch event.Kind {
			case "response_delta":
				answer.WriteString(event.Text)
			case "error":
				return "", fmt.Errorf("caregiving: %s", event.Error)
			case "responded":
				if !event.PendingWork {
					return answer.String(), nil
				}
			}
		}
	}
}
