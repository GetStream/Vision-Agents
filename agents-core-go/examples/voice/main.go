// Command voice holds a spoken conversation, on a Stream call or on the phone.
//
// The whole pipeline runs in the acceleration backend: it joins the call, transcribes,
// answers and speaks. What runs here is the get_weather function and the deciding of what
// the agent is.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=acme \
//	STREAM_API_KEY=... STREAM_API_SECRET=... \
//	go run ./examples/voice                                   # join a call and print a link to it
//	go run ./examples/voice -number +15125551234              # answer that number instead
//	go run ./examples/voice -number +1512... -call +1555...   # ring somebody
//
// The default joins a Stream call and prints a link a person can open to talk to the agent
// from a browser, which needs no telephony and costs nothing. The phone modes need a vendor
// configured, and buying a number starts a monthly charge.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/GetStream/Vision-Agents/agents-core-go/agents"
	"github.com/GetStream/Vision-Agents/agents-core-go/edge"
	"github.com/GetStream/Vision-Agents/agents-core-go/stream"
)

func main() {
	number := flag.String("number", "", "answer this number of yours; empty joins a Stream call instead")
	dial := flag.String("call", "", "ring this number rather than waiting to be rung; needs -number")
	buy := flag.Bool("buy", false, "buy a number to answer on, which starts a monthly charge")
	greeting := flag.String("greeting", "Hey, I'm Jean. What can I do for you?",
		"said on joining without asking the model; empty waits to be spoken to")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *number, *dial, *buy, *greeting); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, number, dial string, buy bool, greeting string) error {
	llm := stream.Accelerated(stream.Config{
		STT:      "deepgram/flux-general-en",
		TTS:      "cartesia/sonic-preview",
		LLM:      "gemini/gemini-3.5-flash-lite",
		Greeting: greeting,
	})

	if err := agents.RegisterFunction(llm, "get_weather",
		"Get the current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
		}) (any, error) {
			return fmt.Sprintf("It is 20 degrees and sunny in %s.", in.Location), nil
		}); err != nil {
		return err
	}

	agent, err := agents.New(agents.Options{
		Name:         "jean",
		Instructions: "You are Jean, a voice assistant. Be brief and warm, and answer in one or two sentences.",
		LLM:          llm,
		// Flash-Lite keeps the conversation quick but is not the model to work an
		// arithmetic or multi-step question out on. A subagent is what turns the
		// built-in think, recall and explain skills on: Jean hands the hard ones to Sol
		// and keeps talking while it reasons.
		Harness: &agents.Harness{
			UseSkills: true,
			Subagents: map[string]string{"default": "openai/gpt-5.6-sol"},
		},
	})
	if err != nil {
		return err
	}

	session, err := start(ctx, agent, number, dial, buy)
	if err != nil {
		return err
	}
	defer session.Close(context.WithoutCancel(ctx))

	if link, err := session.MonitorURL(); err == nil {
		fmt.Println()
		fmt.Println("  Open this to talk to Jean:")
		fmt.Println(" ", link)
		fmt.Println()
	}

	for event := range session.Events() {
		switch event.Kind {
		case "joined":
			fmt.Printf("%s is on the call\n", who(event.Participant))
		case "heard":
			fmt.Printf("caller: %s\n", event.Text)
		case "responded":
			if event.Text != "" {
				fmt.Printf("jean:   %s\n", event.Text)
			}
		case "tool_ran":
			fmt.Printf("        (ran %s)\n", event.Frame.String("tool"))
		case "error":
			fmt.Fprintln(os.Stderr, "error:", event.Error)
		}
	}
	fmt.Println("the call ended")
	return nil
}

// who names a participant, falling back to the id for one that joined without a name.
func who(participant stream.Participant) string {
	if participant.Name != "" {
		return participant.Name
	}
	return participant.UserID
}

// start puts the agent wherever the flags say the conversation is.
func start(ctx context.Context, agent *agents.Agent, number, dial string, buy bool) (*agents.Session, error) {
	if number == "" && buy {
		bought, err := agent.PurchaseAnyNumber(ctx, agents.NumberSearch{Country: "US"})
		if err != nil {
			return nil, err
		}
		fmt.Println("bought", bought)
		number = bought
	}

	switch {
	case number == "" && dial != "":
		return nil, fmt.Errorf("ringing somebody needs one of your own numbers to ring from; pass -number")
	case number == "":
		fmt.Println("joining a Stream call")
		return agent.Join(ctx, edge.Call{})
	case dial != "":
		fmt.Printf("ringing %s from %s\n", dial, number)
		return agent.StartCall(ctx, number, dial)
	default:
		fmt.Println("waiting for a call on", number)
		return agent.WaitForCall(ctx, number)
	}
}
