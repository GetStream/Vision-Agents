package moderator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The tools a session of an agent_moderator config reads the app's moderation data with. The
// ones that count it across the app are in aggregate.go.
const (
	QueryReviewQueueTool       = "query_review_queue"
	GetReviewQueueItemTool     = "get_review_queue_item"
	QueryModerationConfigsTool = "query_moderation_configs"
	QueryModerationRulesTool   = "query_moderation_rules"
	ListBlocklistsTool         = "list_blocklists"
	QueryModerationLogsTool    = "query_moderation_logs"
)

// Tools are the reads of the app's moderation data a session of an agent_moderator config
// is offered. None of them changes anything: the review queue is read without locking an
// item, so asking Nora about a case never takes it from a person reviewing it.
func Tools() []harness.Tool {
	return []harness.Tool{
		{
			Name: QueryReviewQueueTool,
			Description: "Query the app's moderation review queue: the flagged messages, users and other content " +
				"waiting for or past review, with their status, severity and recommended action.",
			Parameters:   queryParameters("Filter on built-in fields such as status, entity_type, channel_cid, severity or recommended_action, or on custom moderation_payload keys."),
			DisplayTitle: "Reading the review queue",
		},
		{
			Name:        GetReviewQueueItemTool,
			Description: "Get one review queue item by id, with the flags raised on it and the moderation actions taken.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"id": map[string]any{"type": "string", "description": "The review queue item's id."}},
				"required":   []any{"id"},
			},
			DisplayTitle: "Reading a review queue item",
		},
		{
			Name: QueryModerationConfigsTool,
			Description: "Query the app's moderation configs (policies): the LLM labels and their instructions in llm_config, " +
				"the AI text, image and video settings, and which blocklists each config applies.",
			Parameters:   queryParameters("Filter on fields such as key or team, for example {\"key\": \"chat:messaging\"}."),
			DisplayTitle: "Reading the moderation configs",
		},
		{
			Name:         QueryModerationRulesTool,
			Description:  "Query the app's moderation rules: the conditions that act on content or users automatically.",
			Parameters:   queryParameters("Filter on fields such as name, rule_type, config_keys or enabled."),
			DisplayTitle: "Reading the moderation rules",
		},
		{
			Name:         ListBlocklistsTool,
			Description:  "List the app's blocklists with their words and how each one matches.",
			Parameters:   map[string]any{"type": "object", "properties": map[string]any{}},
			DisplayTitle: "Reading the blocklists",
		},
		{
			Name:         QueryModerationLogsTool,
			Description:  "Query the app's moderation logs: the actions moderators and automations took, newest first unless sorted otherwise.",
			Parameters:   queryParameters("Filter on fields such as entity_id, entity_type, entity_creator_id, user_id or created_at."),
			DisplayTitle: "Reading the moderation logs",
		},
		{
			Name: TopFlaggedUsersTool,
			Description: "Rank the users whose content was flagged over the last days by violations: flagged items " +
				"whose action was not overturned by a restore or unban. Each user comes with their flagged items, flags, " +
				"items a human moderator acted on, overturned items, Nora's recommendations and most common labels. " +
				"Use it for questions such as who violates most; the router does the counting. When partial is true, " +
				"say the counts are partial.",
			Parameters: windowParameters(map[string]any{
				"label": map[string]any{"type": "string", "description": "Count only items flagged with this label, such as harassment."},
				"limit": map[string]any{"type": "integer", "description": "How many users to return, at most 25. Defaults to 10.", "minimum": 1, "maximum": maxLimit},
			}),
			DisplayTitle: "Counting the most flagged users",
		},
		{
			Name: ModerationSummaryTool,
			Description: "Count the items flagged over the last days by label (with how many were overturned), by " +
				"moderation action taken on them (with how many a human moderator took) and by channel, the 25 largest " +
				"of each. Use it for questions such as which labels fire most or are most often overturned, or which " +
				"channels have the most flagged content. When partial is true, say the counts are partial.",
			Parameters:   windowParameters(nil),
			DisplayTitle: "Counting the moderation activity",
		},
	}
}

// queryParameters is the schema every query tool takes: Stream's filter, sort and page.
func queryParameters(filter string) map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"filter": map[string]any{"type": "object", "description": filter + " Uses Stream's query operators, such as $eq, $in and $gte."},
			"sort": map[string]any{
				"type":        "array",
				"description": "Sort order, for example [{\"field\": \"created_at\", \"direction\": -1}].",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"field":     map[string]any{"type": "string"},
						"direction": map[string]any{"type": "integer", "enum": []any{-1, 1}},
					},
				},
			},
			"limit": map[string]any{"type": "integer", "description": "How many to return, at most 25.", "minimum": 1, "maximum": maxLimit},
			"next":  map[string]any{"type": "string", "description": "The next cursor a previous page returned."},
		},
	}
}

// maxLimit is the most a query returns at once. A model may ask past what its schema says,
// and a page of review queue items that size would fill the context it answers from.
const maxLimit = 25

// arguments is what any of the tools may be called with.
type arguments struct {
	ID     string                       `json:"id"`
	Filter map[string]any               `json:"filter"`
	Sort   []getstream.SortParamRequest `json:"sort"`
	Limit  *int                         `json:"limit"`
	Next   *string                      `json:"next"`

	Days       *int   `json:"days"`
	Label      string `json:"label"`
	EntityType string `json:"entity_type"`
}

// Runner runs the moderation tools it was given in one Stream app, the session's, and hands
// every other call on.
type Runner struct {
	client *getstream.Stream
	names  map[string]bool
	next   agent.ToolRunner
}

// NewRunner reads the moderation data of the app client acts in, for the tools named.
func NewRunner(client *getstream.Stream, tools []harness.Tool, next agent.ToolRunner) *Runner {
	names := make(map[string]bool, len(tools))
	for _, tool := range tools {
		names[tool.Name] = true
	}
	return &Runner{client: client, names: names, next: next}
}

func (r *Runner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if !r.names[call.Name] {
		return r.next.Run(ctx, call)
	}
	var args arguments
	if raw := strings.TrimSpace(call.Arguments); raw != "" && call.Name != ListBlocklistsTool {
		if err := json.Unmarshal([]byte(raw), &args); err != nil {
			return nil, stack.Wrap(fmt.Errorf("moderator: %s arguments: %w", call.Name, err))
		}
	}
	if args.Limit != nil && *args.Limit > maxLimit {
		limit := maxLimit
		args.Limit = &limit
	}
	moderation := r.client.Moderation()
	var (
		data any
		err  error
	)
	switch call.Name {
	case QueryReviewQueueTool:
		data, err = read(moderation.QueryReviewQueue(ctx, &getstream.QueryReviewQueueRequest{
			Filter: args.Filter, Sort: args.Sort, Limit: args.Limit, Next: args.Next,
		}))
	case GetReviewQueueItemTool:
		if args.ID == "" {
			return nil, stack.Wrap(fmt.Errorf("moderator: %s needs the item's id", call.Name))
		}
		data, err = read(moderation.GetReviewQueueItem(ctx, args.ID, &getstream.GetReviewQueueItemRequest{}))
	case QueryModerationConfigsTool:
		data, err = read(moderation.QueryModerationConfigs(ctx, &getstream.QueryModerationConfigsRequest{
			Filter: args.Filter, Sort: args.Sort, Limit: args.Limit, Next: args.Next,
		}))
	case QueryModerationRulesTool:
		data, err = read(moderation.QueryModerationRules(ctx, &getstream.QueryModerationRulesRequest{
			Filter: args.Filter, Sort: args.Sort, Limit: args.Limit, Next: args.Next,
		}))
	case ListBlocklistsTool:
		data, err = read(r.client.ListBlockLists(ctx, &getstream.ListBlockListsRequest{}))
	case QueryModerationLogsTool:
		data, err = read(moderation.QueryModerationLogs(ctx, &getstream.QueryModerationLogsRequest{
			Filter: args.Filter, Sort: args.Sort, Limit: args.Limit, Next: args.Next,
		}))
	case TopFlaggedUsersTool:
		data, err = r.topFlaggedUsers(ctx, args)
	case ModerationSummaryTool:
		data, err = r.moderationSummary(ctx, args)
	default:
		return r.next.Run(ctx, call)
	}
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("moderator: %s: %w", call.Name, err))
	}
	answer, err := json.Marshal(data)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("moderator: %s answer: %w", call.Name, err))
	}
	return llm.TextParts(string(answer)), nil
}

// read is what Stream answered, without the envelope around it.
func read[T any](response *getstream.StreamResponse[T], err error) (any, error) {
	if err != nil {
		return nil, err
	}
	return response.Data, nil
}
