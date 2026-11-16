package client

import (
	"errors"
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func TestARegenerateAsksTheRouterToReplaceTheResponseWithNoText(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	regenerated, err := responses.Regenerate(t.Context(), "response-2", ReasoningEffort("low"))
	if err != nil {
		t.Fatal(err)
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/responses")
	if body["replaces"] != "response-2" || body["text"] != "" || body["reasoning_effort"] != "low" {
		t.Errorf("the regenerate went over as %v", body)
	}
	if id, _ := body["request_id"].(string); id == "" {
		t.Error("a regenerate went without a request id, so a retry would be answered twice")
	}
	if regenerated.ID() != "response-3" {
		t.Errorf("the new response is %q", regenerated.ID())
	}
	if branch := regenerated.Created.Branch; branch == nil || branch.Index != 2 || branch.Count != 2 {
		t.Errorf("its place among its alternatives came back as %+v", branch)
	}
	if rewound := regenerated.Created.RewoundTo; rewound == nil || rewound.MessageId != "question-2" {
		t.Errorf("where the channel was rewound to came back as %+v", rewound)
	}
}

func TestAnEditAsksTheNewTextInPlaceOfTheQuestion(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	if _, err := responses.Edit(t.Context(), "response-2", "And on Sunday?", RequestID("edit-1")); err != nil {
		t.Fatal(err)
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/responses")
	if body["replaces"] != "response-2" || body["text"] != "And on Sunday?" || body["request_id"] != "edit-1" {
		t.Errorf("the edit went over as %v", body)
	}
}

func TestAnEditWithNoTextIsNotSentAsARegenerate(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	if _, err := responses.Edit(t.Context(), "response-2", ""); err == nil {
		t.Fatal("an edit with no text was sent")
	}
	if asked := backend.requests("POST", "/v1/agents/sessions/session-1/responses"); asked != 0 {
		t.Errorf("the router was asked %d times", asked)
	}
}

func TestAResponseWithNoIDCannotBeReplaced(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	if _, err := responses.Regenerate(t.Context(), ""); err == nil {
		t.Fatal("a response with no id was regenerated")
	}
	if asked := backend.requests("POST", "/v1/agents/sessions/session-1/responses"); asked != 0 {
		t.Errorf("the router was asked %d times", asked)
	}
}

func TestSiblingsArePagedByTheCursorTheRouterHandsOut(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	first, err := responses.Siblings(t.Context(), "response-3", acceleration.ListResponseSiblingsParamsOfQuestion, 1, "")
	if err != nil {
		t.Fatal(err)
	}
	if !first.HasMore || first.NextCursor == nil || len(first.Items) != 1 || first.Items[0].Id != "response-2" {
		t.Fatalf("the first page came back as %+v", first)
	}
	query := backend.query(t, "GET", "/v1/agents/sessions/session-1/responses/response-3/siblings")
	if query.Get("of") != "question" || query.Get("limit") != "1" || query.Has("cursor") {
		t.Errorf("the first page was asked for as %v", query)
	}
	rest, err := responses.Siblings(t.Context(), "response-3", acceleration.ListResponseSiblingsParamsOfQuestion, 1, *first.NextCursor)
	if err != nil {
		t.Fatal(err)
	}
	if rest.HasMore || len(rest.Items) != 1 || !rest.Items[0].Active || rest.Items[0].Index != 2 {
		t.Errorf("the last page came back as %+v", rest)
	}
}

func TestAHiddenBranchIsReadAsChatMessages(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	read, err := responses.Branch(t.Context(), "response-2", 0, "")
	if err != nil {
		t.Fatal(err)
	}

	if read.Active || len(read.Items) != 1 || read.Items[0].Id != "reply-2" || read.Items[0].Text != "Yes." {
		t.Errorf("the branch came back as %+v", read)
	}
	if query := backend.query(t, "GET", "/v1/agents/sessions/session-1/responses/response-2/branch"); len(query) != 0 {
		t.Errorf("an unset limit and cursor went over as %v", query)
	}
}

func TestTheActiveBranchIsReadAsActiveWithNoMessages(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	read, err := responses.Branch(t.Context(), "response-3", 0, "")
	if err != nil {
		t.Fatal(err)
	}

	if !read.Active || len(read.Items) != 0 {
		t.Errorf("the active branch came back as %+v", read)
	}
}

func TestABranchWhoseCopiesWereDeletedSaysSo(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	_, err := responses.Branch(t.Context(), "truncated", 0, "")

	var refused *stream.RouterError
	if !errors.As(err, &refused) || refused.Status != http.StatusGone || refused.Code != "branch_unavailable" {
		t.Fatalf("the refusal came back as %v", err)
	}
}

func TestActivatingABranchSaysWhatItSentAgain(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("session-1")

	activated, err := responses.Activate(t.Context(), "response-2")
	if err != nil {
		t.Fatal(err)
	}

	if backend.requests("POST", "/v1/agents/sessions/session-1/responses/response-2/activate") != 1 {
		t.Error("the router was not asked to activate the branch")
	}
	if activated.LeafResponseId != "response-2" || activated.Resent != 1 || activated.AlreadyActive ||
		activated.RewoundTo == nil || activated.RewoundTo.MessageId != "question-2" {
		t.Errorf("the activation came back as %+v", activated)
	}
}
