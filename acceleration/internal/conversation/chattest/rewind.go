package chattest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Range is one created_at range a channel's rewinds hide, as Chat holds it: To is always
// hidden, From only when FromInclusive.
type Range struct {
	From, To      time.Time
	FromInclusive bool
}

// Contains is whether a message created at t is hidden by the range.
func (r Range) Contains(t time.Time) bool {
	if t.After(r.To) {
		return false
	}
	if r.FromInclusive {
		return !t.Before(r.From)
	}
	return t.After(r.From)
}

// maxRewinds is Chat's MaxRewindsPerChannel: the most separate ranges a channel can have.
const maxRewinds = 100

// rewindState is what the rewind endpoint holds and is steered by.
type rewindState struct {
	// ranges are each channel's merged hidden ranges, by cid.
	ranges map[string][]Range
	// cap is the most merged ranges a channel can have, 0 for Chat's own.
	cap int
	// off answers every rewind as an app without the feature, or with it frozen.
	off bool
	// forking are the cids answered as forks whose copy is not complete.
	forking map[string]bool
	made    int
}

// DisableRewind answers every rewind from now on as Chat does for an app that is not
// granted the feature, or has it frozen: a 403 naming no reason.
func (s *Server) DisableRewind() {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.rewind.off = true
}

// CapRewinds sets the most separate hidden ranges a channel can have, Chat's 100 until set.
func (s *Server) CapRewinds(n int) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.rewind.cap = n
}

// ForkInProgress answers a rewind of the channel as one of a fork whose copy is not
// complete, until done is called.
func (s *Server) ForkInProgress(id string) (done func()) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	if s.db.rewind.forking == nil {
		s.db.rewind.forking = map[string]bool{}
	}
	cid := cidOf(id)
	s.db.rewind.forking[cid] = true
	return func() {
		s.db.mu.Lock()
		defer s.db.mu.Unlock()
		delete(s.db.rewind.forking, cid)
	}
}

// HiddenRanges are the channel's merged hidden ranges, oldest first.
func (s *Server) HiddenRanges(id string) []Range {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return slices.Clone(s.db.rewind.ranges[cidOf(id)])
}

// Visible are the ids of a channel's messages a read returns, in the order they were
// written: those its rewinds hide are left out, as Stored keeps them.
func (s *Server) Visible(id string) []string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var ids []string
	for _, message := range s.db.visibleIn(cidOf(id)) {
		ids = append(ids, message["id"].(string))
	}
	return ids
}

// createdAt is when a stored message was created, as it was stamped.
func createdAt(message map[string]any) time.Time {
	switch at := message["created_at"].(type) {
	case int64:
		return time.Unix(0, at).UTC()
	case float64:
		return time.Unix(0, int64(at)).UTC()
	case string:
		parsed, _ := time.Parse(time.RFC3339Nano, at)
		return parsed
	}
	return time.Time{}
}

// micro is t at the microsecond, the precision Chat stores created_at at.
func micro(t time.Time) time.Time { return t.UTC().Truncate(time.Microsecond) }

// hidden is whether a stored message falls in one of its channel's ranges.
func (db *store) hidden(message map[string]any) bool {
	if message == nil {
		return false
	}
	cid, _ := message["cid"].(string)
	at := micro(createdAt(message))
	for _, r := range db.rewind.ranges[cid] {
		if r.Contains(at) {
			return true
		}
	}
	return false
}

// visibleIn are the messages of the channel cid a read returns, in the order they were
// written. A channel with no ranges reads exactly as messagesIn.
func (db *store) visibleIn(cid string) []map[string]any {
	all := db.messagesIn(cid)
	if len(db.rewind.ranges[cid]) == 0 {
		return all
	}
	return slices.DeleteFunc(all, db.hidden)
}

// pageAfter answers a channel read. With id_gt or id_gte it is the page of at most limit
// messages from there, oldest first, as Chat reads ascending; any other read gets every
// visible message, as this fake always has.
func (db *store) pageAfter(cid string, body map[string]any) []map[string]any {
	visible := db.visibleIn(cid)
	params, _ := body["messages"].(map[string]any)
	gt, _ := params["id_gt"].(string)
	gte, _ := params["id_gte"].(string)
	if gt == "" && gte == "" {
		return visible
	}
	from := gte
	if from == "" {
		from = gt
	}
	// Chat pages by (created_at, id) of the message named, which need not be visible.
	anchor := db.messages[from]
	if anchor == nil {
		return []map[string]any{}
	}
	at, id := micro(createdAt(anchor)), from
	page := []map[string]any{}
	for _, message := range visible {
		mat, mid := micro(createdAt(message)), message["id"].(string)
		order := mat.Compare(at)
		if order == 0 {
			order = strings.Compare(mid, id)
		}
		if order > 0 || (order == 0 && gte != "") {
			page = append(page, message)
		}
	}
	if limit, ok := params["limit"].(float64); ok && int(limit) < len(page) {
		page = page[:int(limit)]
	}
	return page
}

// afterHiddenRanges is when a message sent at now is stored: Chat moves a send that is not
// after the channel's newest range to one microsecond past it, so it cannot land in the hole.
func (db *store) afterHiddenRanges(cid string, now time.Time) time.Time {
	ranges := db.rewind.ranges[cid]
	if len(ranges) == 0 {
		return now
	}
	newest := ranges[len(ranges)-1].To
	if micro(now).After(newest) {
		return now
	}
	return newest.Add(time.Microsecond)
}

// mergeRanges is Chat's rewindstate.Merge: overlapping and touching ranges become one, and a
// range starting where another does keeps the inclusive lower bound.
func mergeRanges(ranges []Range) []Range {
	sorted := slices.Clone(ranges)
	slices.SortStableFunc(sorted, func(a, b Range) int {
		if c := a.From.Compare(b.From); c != 0 {
			return c
		}
		switch {
		case a.FromInclusive == b.FromInclusive:
			return 0
		case a.FromInclusive:
			return -1
		default:
			return 1
		}
	})
	merged := make([]Range, 0, len(sorted))
	for _, r := range sorted {
		if n := len(merged); n > 0 && !r.From.After(merged[n-1].To) {
			last := &merged[n-1]
			if r.From.Equal(last.From) {
				last.FromInclusive = last.FromInclusive || r.FromInclusive
			}
			if r.To.After(last.To) {
				last.To = r.To
			}
			continue
		}
		merged = append(merged, r)
	}
	return merged
}

// chatError is Chat's error body: exception_fields carries the reason a client switches on.
func chatError(status, code int, message string, fields map[string]string) map[string]any {
	answer := map[string]any{"code": code, "message": message, "StatusCode": status}
	if fields != nil {
		answer["exception_fields"] = fields
	}
	return answer
}

// rewindChannel is Chat's RewindChannel, its checks in Chat's order (lib/chat/state
// channel_rewinds.go): it answers with the status and body Chat would.
func (db *store) rewindChannel(channelType, id string, body map[string]any) (int, map[string]any) {
	cid := channelType + ":" + id
	if db.rewind.off {
		return http.StatusForbidden, chatError(http.StatusForbidden, 17, "this endpoint needs a feature flag, contact support to get it enabled", nil)
	}
	var req struct {
		MessageID             string `json:"message_id"`
		Inclusive             bool   `json:"inclusive"`
		ExpectedLastMessageID string `json:"expected_last_message_id"`
	}
	raw, _ := json.Marshal(body)
	_ = json.Unmarshal(raw, &req)
	if req.MessageID == "" || req.ExpectedLastMessageID == "" {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, "message_id and expected_last_message_id are required", nil)
	}
	if channelType != "agent" {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, fmt.Sprintf("only agent channels can be rewound, %q is not one", channelType), nil)
	}
	if _, ok := db.channels[cid]; !ok {
		return http.StatusNotFound, chatError(http.StatusNotFound, 16, "channel "+cid+" does not exist", nil)
	}
	if db.rewind.forking[cid] {
		return http.StatusConflict, chatError(http.StatusConflict, 131, "channel "+cid+" is a fork whose copy is not complete yet",
			map[string]string{"reason": "fork_in_progress"})
	}

	target := db.messages[req.MessageID]
	if target == nil || target["cid"] != cid || db.hidden(target) {
		return http.StatusNotFound, chatError(http.StatusNotFound, 16, "message "+req.MessageID+" does not exist", nil)
	}
	if parent, _ := target["parent_id"].(string); parent != "" {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, "message "+req.MessageID+" is a thread reply, a channel can only be rewound to a top-level message", nil)
	}
	if target["type"] == "deleted" {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, "message "+req.MessageID+" is deleted", nil)
	}

	visible := db.visibleIn(cid)
	newest := visible[len(visible)-1]
	if newest["id"] != req.ExpectedLastMessageID {
		return http.StatusConflict, chatError(http.StatusConflict, 125, "the channel changed since expected_last_message_id was read, reload and retry",
			map[string]string{"reason": "rewind_conflict", "last_message_id": newest["id"].(string)})
	}
	if !req.Inclusive && newest["id"] == req.MessageID {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, "message "+req.MessageID+" is the newest visible message, a rewind to it that is not inclusive hides nothing",
			map[string]string{"reason": "nothing_to_rewind"})
	}
	from := micro(createdAt(target))
	tied := 0
	for _, message := range visible {
		if micro(createdAt(message)).Equal(from) {
			tied++
		}
	}
	if tied > 1 {
		return http.StatusConflict, chatError(http.StatusConflict, 126, "another message has the same created_at as message "+req.MessageID+", the channel cannot be rewound to it",
			map[string]string{"reason": "rewind_point_ambiguous"})
	}

	to := micro(db.now())
	if at := micro(createdAt(newest)); at.After(to) {
		to = at
	}
	to = to.Add(time.Microsecond)
	added := Range{From: from, To: to, FromInclusive: req.Inclusive}
	hiddenCount := 0
	for _, message := range visible {
		if added.Contains(micro(createdAt(message))) {
			hiddenCount++
		}
	}
	limit := db.rewind.cap
	if limit == 0 {
		limit = maxRewinds
	}
	merged := mergeRanges(append(slices.Clone(db.rewind.ranges[cid]), added))
	if len(merged) > limit {
		return http.StatusBadRequest, chatError(http.StatusBadRequest, 4, fmt.Sprintf("a channel can have at most %d separate rewound ranges", limit),
			map[string]string{"reason": "too_many_rewinds", "limit": strconv.Itoa(limit)})
	}
	if db.rewind.ranges == nil {
		db.rewind.ranges = map[string][]Range{}
	}
	db.rewind.ranges[cid] = merged
	db.rewind.made++
	return http.StatusCreated, map[string]any{
		"rewind": map[string]any{
			"id": fmt.Sprintf("rewind-%d", db.rewind.made), "message_id": req.MessageID, "inclusive": req.Inclusive,
			"hidden_count_at_rewind": hiddenCount, "created_at": db.now().UnixNano(),
		},
		"channel":  db.channels[cid],
		"duration": "0ms",
	}
}
