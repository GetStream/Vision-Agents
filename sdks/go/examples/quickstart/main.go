// Command quickstart is the Go code from the quickstart: the agent in agents/myagent, with
// one function the model can call, asked one question in writing.
//
// The model runs in the backend; when it asks for lookup_order, the call comes to this
// process, which runs it and sends the result back. Sync stores the folder in the backend,
// which is what `agents sync` does in the quickstart.
//
//	cd examples/quickstart
//	STREAM_API_KEY=your_api_key STREAM_API_SECRET=your_api_secret go run .
//
// Against a local router instead, name it and the customer:
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 STREAM_ACCELERATION_CUSTOMER_ID=acme go run .
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

type LookupOrder struct {
	OrderID string `json:"order_id" schema:"the order number, e.g. 1042"`
}

func (LookupOrder) Name() string        { return "lookup_order" }
func (LookupOrder) Description() string { return "Look up an order by its number" }
func (l LookupOrder) Run(ctx context.Context) (any, error) {
	return fmt.Sprintf("Order %s shipped yesterday.", l.OrderID), nil
}

func main() {
	ctx := context.Background()

	agent, err := agents.New(agents.Options{
		Name:  "myagent",
		Tools: []tools.Tool{LookupOrder{}},
	})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := agent.Sync(ctx); err != nil {
		log.Fatal(err)
	}

	session, err := agent.Sessions.Create(ctx, agents.SessionOptions{})
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close(ctx)

	if _, err := session.Responses.Create(ctx, "Where is order 1042?"); err != nil {
		log.Fatal(err)
	}
	for event := range session.Events() {
		if event.Kind == "response_delta" {
			fmt.Print(event.Text)
		}
		if event.Kind == "responded" && !event.PendingWork {
			fmt.Println()
			break
		}
	}
}
