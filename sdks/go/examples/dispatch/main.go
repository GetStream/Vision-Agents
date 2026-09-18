// Command dispatch waits for messages written to an agent and answers them.
//
// It is the Go counterpart of examples/voice_agents/chat_support. Nothing here is
// reachable from the internet: the worker connects out to the router and waits, and a
// message somebody wrote in an agent channel is pushed down that connection.
//
// The model runs in the backend and the functions it calls run here, which is the point of
// waiting rather than being called. A worker in another language holds its own functions
// the same way.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=acme \
//	go run ./examples/dispatch
//
// Then write in an agent channel. A channel a conversation has already been held in is
// claimed by that conversation; a new one has to name the agent config answering in it,
// under the channel's own "agent_config_id".
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	dispatch, err := agents.NewDispatch(agents.DispatchOptions{Capacity: 8})
	if err != nil {
		return err
	}

	dispatch.OnMessage(func(ctx context.Context, message agents.InboundMessage) error {
		conversation, err := dispatch.Conversation(ctx, message, build)
		if err != nil {
			return err
		}
		// Nothing is waited for. The answer is written into the channel by the backend as
		// it is generated, so the person who wrote is already reading it.
		return conversation.Respond(message.Text)
	})

	log.Println("waiting for messages; write in an agent channel to start one")
	return dispatch.Run(ctx)
}

// build makes the agent for a channel nothing is answering yet.
//
// It is called once per conversation rather than once per message: the agent that answered
// the last message on a channel is the one that knows what has been said, and it is kept
// for the next.
func build(_ context.Context, message agents.InboundMessage) (*agents.Agent, error) {
	// The config the channel was last answered under, so a worker serving several agents
	// answers as the one that was written to.
	llm := stream.Accelerated(stream.Config{ConfigID: message.ConfigID, LLM: "llm-fast"})

	if err := agents.RegisterFunction(llm, "get_weather",
		"Get the current weather for a location",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city and state, e.g. Boulder, CO"`
		}) (any, error) {
			return fmt.Sprintf("It is 20 degrees and sunny in %s.", in.Location), nil
		}); err != nil {
		return nil, err
	}

	return agents.New(agents.Options{
		Name:         "jean",
		Instructions: "You are Jean, a friendly assistant. Keep answers to a sentence or two.",
		LLM:          llm,
		// Whatever the channel was created with arrives unread, which is where a worker
		// finds what the conversation is for and the router has no opinion about. Whoever
		// created the channel decided what is in it, so it is a claim rather than a fact:
		// fine to bill and scope memory by, not to grant anything on.
		CostTracking: map[string]string{
			"user_id":         message.UserID,
			"organization_id": message.Custom["organization_id"],
		},
	})
}
