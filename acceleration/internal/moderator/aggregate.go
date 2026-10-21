package moderator

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The tools that count the app's moderation data, so a question across the app ("who
// violates most?") is answered by the router's arithmetic rather than the model's.
const (
	TopFlaggedUsersTool   = "top_flagged_users"
	ModerationSummaryTool = "moderation_summary"
)

const (
	// defaultDays and maxDays bound the window an aggregate reads.
	defaultDays = 7
	maxDays     = 30
	// defaultTop is how many users top_flagged_users answers with when not told.
	defaultTop = 10
	// pageSize and maxPages cap what one call reads from the review queue: 2,000 items, so
	// a busy app's week cannot hold a tool call open for minutes. Past it, the answer says
	// its counts are partial.
	pageSize = 100
	maxPages = 20
	// scanBudget stops paging a slow Stream early. The moderation tools run outside the
	// timeout a caller's own tools get, and the model waits on every page read.
	scanBudget = 20 * time.Second
	// maxRows is the most entries one breakdown of moderation_summary lists.
	maxRows = 25
)

// The action types that undo an earlier action: the item they are on was not a violation.
var overturnActions = map[string]bool{"restore": true, "unban": true, "unblock": true}

// The action types a moderator takes without deciding the content violated anything.
var nonConfirmingActions = map[string]bool{
	"mark_reviewed": true, "escalate": true, "de_escalate": true, "flag": true, agentRecommendation: true,
}

const (
	// agentRecommendation is the action Nora records as its suggestion on an item.
	agentRecommendation = "agent_recommendation"
	moderatorReporter   = "moderator"
)

// windowParameters is the schema both aggregate tools take: the window and what to count.
func windowParameters(extra map[string]any) map[string]any {
	properties := map[string]any{
		"days": map[string]any{
			"type": "integer", "description": "How many days back to count, ending now. Defaults to 7.",
			"minimum": 1, "maximum": maxDays,
		},
		"entity_type": map[string]any{
			"type":        "string",
			"description": "Count only items of this entity type, such as stream:chat:v1:message or stream:user.",
		},
	}
	for name, schema := range extra {
		properties[name] = schema
	}
	return map[string]any{"type": "object", "properties": properties}
}

// window is the span an aggregate counted.
type window struct {
	From time.Time `json:"from"`
	To   time.Time `json:"to"`
	Days int       `json:"days"`
}

// scan is the review queue items flagged in a window, newest first, as far as the page cap
// let them be read.
type scan struct {
	Window    window `json:"window"`
	ItemsRead int    `json:"items_read"`
	Partial   bool   `json:"partial"`
	Note      string `json:"note,omitempty"`
	items     []getstream.ReviewQueueItemResponse
}

// readWindow pages the review queue from now back over the window the arguments name.
func (r *Runner) readWindow(ctx context.Context, args arguments) (*scan, error) {
	days := defaultDays
	if args.Days != nil {
		days = min(max(*args.Days, 1), maxDays)
	}
	started := time.Now()
	to := started.UTC()
	from := to.AddDate(0, 0, -days)
	filter := map[string]any{"created_at": map[string]any{"$gte": from.Format(time.RFC3339)}}
	if args.EntityType != "" {
		filter["entity_type"] = args.EntityType
	}
	field, direction, limit := "created_at", -1, pageSize
	request := &getstream.QueryReviewQueueRequest{
		Filter: filter,
		Sort:   []getstream.SortParamRequest{{Field: &field, Direction: &direction}},
		Limit:  &limit,
	}
	result := &scan{Window: window{From: from, To: to, Days: days}}
	for page := 0; page < maxPages && (page == 0 || time.Since(started) < scanBudget); page++ {
		response, err := r.client.Moderation().QueryReviewQueue(ctx, request)
		if err != nil {
			return nil, stack.Wrap(err)
		}
		result.items = append(result.items, response.Data.Items...)
		next := response.Data.Next
		if next == nil || *next == "" || len(response.Data.Items) == 0 {
			result.ItemsRead = len(result.items)
			return result, nil
		}
		request.Next = next
	}
	result.ItemsRead = len(result.items)
	result.Partial = true
	result.Note = fmt.Sprintf("Only the newest %d items flagged in the window were read; older ones were not counted, "+
		"so every count is a lower bound. Say so in the answer, or ask again with fewer days.", result.ItemsRead)
	return result, nil
}

// labels are the distinct labels the flags on an item carry.
func labels(item getstream.ReviewQueueItemResponse) []string {
	var found []string
	for _, flag := range item.Flags {
		for _, label := range flag.Labels {
			if label != "" && !slices.Contains(found, label) {
				found = append(found, label)
			}
		}
	}
	return found
}

// hasLabel says whether any flag on the item carries the label, ignoring case.
func hasLabel(item getstream.ReviewQueueItemResponse, label string) bool {
	return slices.ContainsFunc(labels(item), func(l string) bool { return strings.EqualFold(l, label) })
}

// overturned says whether an action on the item undid an earlier one.
func overturned(item getstream.ReviewQueueItemResponse) bool {
	return slices.ContainsFunc(item.Actions, func(a getstream.ActionLogResponse) bool { return overturnActions[a.Type] })
}

// confirmedByModerator says whether a human moderator acted on the item as a violation.
func confirmedByModerator(item getstream.ReviewQueueItemResponse) bool {
	return slices.ContainsFunc(item.Actions, func(a getstream.ActionLogResponse) bool {
		return a.ReporterType == moderatorReporter && !overturnActions[a.Type] && !nonConfirmingActions[a.Type]
	})
}

// creator is the user an item is about: the author of flagged content, or the flagged user.
func creator(item getstream.ReviewQueueItemResponse) string {
	if item.EntityCreatorID != nil && *item.EntityCreatorID != "" {
		return *item.EntityCreatorID
	}
	if item.EntityType == "stream:user" {
		return item.EntityID
	}
	return ""
}

// flagCount is how many flags were raised on an item.
func flagCount(item getstream.ReviewQueueItemResponse) int {
	return max(item.FlagsCount, len(item.Flags))
}

// labelCount is how often a label was on a user's items.
type labelCount struct {
	Label string `json:"label"`
	Count int    `json:"count"`
}

// flaggedUser is one user's record over the window. Violations leaves out the items whose
// action was overturned, since a restore or unban says the flag was wrong.
type flaggedUser struct {
	UserID               string       `json:"user_id"`
	Violations           int          `json:"violations"`
	FlaggedItems         int          `json:"flagged_items"`
	Flags                int          `json:"flags"`
	ConfirmedByModerator int          `json:"confirmed_by_moderator"`
	Overturned           int          `json:"overturned"`
	NoraRecommendations  int          `json:"nora_recommendations"`
	TopLabels            []labelCount `json:"top_labels"`

	labels map[string]int
}

// topFlaggedUsers is the answer of top_flagged_users.
type topFlaggedUsers struct {
	*scan
	Label string         `json:"label,omitempty"`
	Users []*flaggedUser `json:"users"`
}

// topFlaggedUsers ranks the users whose content was flagged in the window by violations.
func (r *Runner) topFlaggedUsers(ctx context.Context, args arguments) (any, error) {
	read, err := r.readWindow(ctx, args)
	if err != nil {
		return nil, err
	}
	users := map[string]*flaggedUser{}
	for _, item := range read.items {
		id := creator(item)
		if id == "" || (args.Label != "" && !hasLabel(item, args.Label)) {
			continue
		}
		user := users[id]
		if user == nil {
			user = &flaggedUser{UserID: id, labels: map[string]int{}}
			users[id] = user
		}
		user.FlaggedItems++
		user.Flags += flagCount(item)
		for _, action := range item.Actions {
			if action.Type == agentRecommendation {
				user.NoraRecommendations++
			}
		}
		if overturned(item) {
			user.Overturned++
			continue
		}
		user.Violations++
		if confirmedByModerator(item) {
			user.ConfirmedByModerator++
		}
		for _, label := range labels(item) {
			user.labels[label]++
		}
	}
	ranked := make([]*flaggedUser, 0, len(users))
	for _, user := range users {
		user.TopLabels = ranks(user.labels, 3, func(label string, count int) labelCount {
			return labelCount{Label: label, Count: count}
		})
		ranked = append(ranked, user)
	}
	slices.SortFunc(ranked, func(a, b *flaggedUser) int {
		return cmp.Or(cmp.Compare(b.Violations, a.Violations), cmp.Compare(b.Flags, a.Flags), cmp.Compare(a.UserID, b.UserID))
	})
	top := defaultTop
	if args.Limit != nil {
		top = max(*args.Limit, 1)
	}
	return topFlaggedUsers{scan: read, Label: args.Label, Users: ranked[:min(top, len(ranked))]}, nil
}

// labelRow is one label's line in moderation_summary.
type labelRow struct {
	Label      string `json:"label"`
	Items      int    `json:"items"`
	Overturned int    `json:"overturned"`
}

// actionRow is one action type's line in moderation_summary.
type actionRow struct {
	Action      string `json:"action"`
	Count       int    `json:"count"`
	ByModerator int    `json:"by_moderator"`
}

// channelRow is one channel's line in moderation_summary.
type channelRow struct {
	ChannelCID string `json:"channel_cid"`
	Items      int    `json:"items"`
	Overturned int    `json:"overturned"`
}

// moderationSummary is the answer of moderation_summary.
type moderationSummary struct {
	*scan
	Overturned int          `json:"overturned"`
	ByLabel    []labelRow   `json:"by_label"`
	ByAction   []actionRow  `json:"by_action"`
	ByChannel  []channelRow `json:"by_channel"`
}

// moderationSummary counts the items flagged in the window by label, action and channel.
func (r *Runner) moderationSummary(ctx context.Context, args arguments) (any, error) {
	read, err := r.readWindow(ctx, args)
	if err != nil {
		return nil, err
	}
	var (
		overturnedItems              int
		byLabel, labelOverturned     = map[string]int{}, map[string]int{}
		byAction, actionByModerator  = map[string]int{}, map[string]int{}
		byChannel, channelOverturned = map[string]int{}, map[string]int{}
	)
	for _, item := range read.items {
		undone := overturned(item)
		if undone {
			overturnedItems++
		}
		for _, label := range labels(item) {
			byLabel[label]++
			if undone {
				labelOverturned[label]++
			}
		}
		for _, action := range item.Actions {
			byAction[action.Type]++
			if action.ReporterType == moderatorReporter {
				actionByModerator[action.Type]++
			}
		}
		if item.Message != nil && item.Message.Cid != "" {
			byChannel[item.Message.Cid]++
			if undone {
				channelOverturned[item.Message.Cid]++
			}
		}
	}
	return moderationSummary{
		scan:       read,
		Overturned: overturnedItems,
		ByLabel: ranks(byLabel, maxRows, func(label string, count int) labelRow {
			return labelRow{Label: label, Items: count, Overturned: labelOverturned[label]}
		}),
		ByAction: ranks(byAction, maxRows, func(action string, count int) actionRow {
			return actionRow{Action: action, Count: count, ByModerator: actionByModerator[action]}
		}),
		ByChannel: ranks(byChannel, maxRows, func(cid string, count int) channelRow {
			return channelRow{ChannelCID: cid, Items: count, Overturned: channelOverturned[cid]}
		}),
	}, nil
}

// ranks turns counts into at most n rows, the largest first and then by key, so the same
// data always answers the same way. It answers an empty list rather than null.
func ranks[T any](counts map[string]int, n int, row func(key string, count int) T) []T {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b string) int { return cmp.Or(cmp.Compare(counts[b], counts[a]), cmp.Compare(a, b)) })
	rows := make([]T, 0, min(n, len(keys)))
	for _, key := range keys[:min(n, len(keys))] {
		rows = append(rows, row(key, counts[key]))
	}
	return rows
}
