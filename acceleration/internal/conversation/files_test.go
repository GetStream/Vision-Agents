package conversation

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/stretchr/testify/require"
)

func TestAnImageIsUploadedToTheChannelAsTheAgent(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)

	shown, err := c.Publish(context.Background(), sandbox.File{Name: "/tmp/render.png", MIME: "image/png", Data: []byte("png")})

	require.NoError(t, err)
	require.Equal(t, sandbox.Attachment{Name: "render.png", MIME: "image/png", URL: "https://cdn.fake/image/render.png", Size: 3}, shown)
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Len(t, db.uploads, 1)
	require.Equal(t, "image", db.uploads[0].endpoint)
	require.Equal(t, []byte("png"), db.uploads[0].data)
	require.JSONEq(t, `{"id":"artist"}`, db.uploads[0].user)
}

func TestAFileThatIsNotAnImageIsUploadedAsAFile(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)

	shown, err := c.Publish(context.Background(), sandbox.File{Name: "scene.blend", MIME: "application/octet-stream", Data: []byte("blend")})

	require.NoError(t, err)
	require.Equal(t, "https://cdn.fake/file/scene.blend", shown.URL)
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Equal(t, "file", db.uploads[0].endpoint)
}

func TestAnEmptyFileIsNotUploaded(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)

	_, err = c.Publish(context.Background(), sandbox.File{Name: "render.png", MIME: "image/png"})

	require.Error(t, err)
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Empty(t, db.uploads)
}

func TestARenderIsAttachedToTheReplyAndComesBackWithItsHistory(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("render a teapot"))
	c.Observe(agent.Delegated{TaskID: "task-1", Skill: "render"})
	c.Observe(agent.Responded{Text: "Rendering it.", PendingWork: true})
	render, err := c.Publish(context.Background(), sandbox.File{Name: "teapot.png", MIME: "image/png", Data: []byte("png")})
	require.NoError(t, err)

	c.Observe(agent.TaskSettled{TaskID: "task-1", Skill: "render", Text: "A teapot.", Files: []sandbox.Attachment{render}})
	c.Observe(agent.ResponseDelta{Text: " Here is your teapot."})
	c.Observe(agent.Responded{})
	saved(t, c)

	require.Equal(t, []sandbox.Attachment{render}, current(c).Files)
	reply := current(c).ID
	db.mu.Lock()
	raw, _ := json.Marshal(db.messages[reply]["attachments"])
	db.mu.Unlock()
	var stored []map[string]any
	require.NoError(t, json.Unmarshal(raw, &stored))
	var image map[string]any
	for _, attachment := range stored {
		if attachment["type"] == "image" {
			image = attachment
		}
	}
	require.NotNil(t, image, "the reply carries the render as an image attachment")
	require.Equal(t, "https://cdn.fake/image/teapot.png", image["image_url"])
	require.Equal(t, "teapot.png", image["title"])

	// Somebody coming back later, on a replica that never saw it, still sees the render.
	cold := NewForChat(client)
	defer cold.Close()
	page, err := cold.History(context.Background(), "customer", "artist", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	require.Empty(t, page.Messages[0].Files, "the person's own message carries nothing")
	require.Equal(t, []sandbox.Attachment{render}, page.Messages[1].Files)
}

func TestFilesCodeTheReplyRanHandsBackAreAttachedToIt(t *testing.T) {
	_, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("render a teapot"))
	render := sandbox.Attachment{Name: "teapot.png", MIME: "image/png", URL: "https://cdn.fake/image/teapot.png", Size: 3}

	c.Observe(agent.ToolStarted{ID: "run-1", Tool: sandbox.ToolName, StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "run-1", Tool: sandbox.ToolName, Result: "Attached.", Files: []sandbox.Attachment{render}})

	require.Equal(t, []sandbox.Attachment{render}, current(c).Files)
}

func TestFilesForWorkTheReplyNeverStartedAreNotAttached(t *testing.T) {
	_, client := newChat(t)
	s := NewForChat(client)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "artist", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("hello"))

	c.Observe(agent.TaskSettled{TaskID: "someone-elses", Files: []sandbox.Attachment{{Name: "x.png", MIME: "image/png", URL: "https://cdn.fake/x.png"}}})

	require.Empty(t, current(c).Files)
}

func TestTheSameFileRenderedAgainReplacesTheEarlierOne(t *testing.T) {
	first := sandbox.Attachment{Name: "render.png", URL: "https://cdn.fake/1"}
	second := sandbox.Attachment{Name: "render.png", URL: "https://cdn.fake/2"}
	other := sandbox.Attachment{Name: "depth.png", URL: "https://cdn.fake/3"}

	merged := mergeFiles([]sandbox.Attachment{first, other}, []sandbox.Attachment{second})

	require.Equal(t, []sandbox.Attachment{other, second}, merged)
}

func TestAPersonsPhotosAreUploadedAsThemAndShownOnTheirMessage(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	t.Cleanup(s.Close)
	c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)

	photos, err := c.PublishPhotos(t.Context(), []llm.ImagePart{{MIME: "image/jpeg", Data: []byte("jpeg")}})
	require.NoError(t, err)
	receipt, err := c.BeginCommand("ask-1", "What is this?", "", photos...)
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "A teapot."})
	c.Observe(agent.Responded{})
	saved(t, c)

	db.mu.Lock()
	require.Len(t, db.uploads, 1)
	require.Equal(t, "image", db.uploads[0].endpoint)
	require.Equal(t, "photo-1.jpeg", db.uploads[0].name)
	require.JSONEq(t, `{"id":"alice"}`, db.uploads[0].user)
	require.Equal(t, []byte("jpeg"), db.uploads[0].data)
	db.mu.Unlock()

	// Somebody coming back later, on a replica that never saw it, still sees the photo.
	cold := NewForChat(client)
	t.Cleanup(cold.Close)
	page, err := cold.HistoryForCaller(t.Context(), "customer", "agent", c.CID(), "", "alice")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	require.Equal(t, receipt.UserMessageID, page.Messages[0].ID)
	require.Equal(t, []sandbox.Attachment{{Name: "photo-1.jpeg", MIME: "image/jpeg", URL: "https://cdn.fake/image/photo-1.jpeg", Size: 4}}, page.Messages[0].Files)
}

func TestARetryWithTheSamePhotosIsTheSameCommandAndOtherPhotosAreAnother(t *testing.T) {
	_, client := newChat(t)
	s := NewForChat(client)
	t.Cleanup(s.Close)
	c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)
	teapot := llm.ImagePart{MIME: "image/jpeg", Data: []byte("teapot")}
	photos, err := c.PublishPhotos(t.Context(), []llm.ImagePart{teapot})
	require.NoError(t, err)
	first, err := c.BeginCommand("ask-1", "What is this?", "", photos...)
	require.NoError(t, err)

	retry, err := c.BeginCommand("ask-1", "What is this?", "", Photo{Sent: teapot})
	require.NoError(t, err)
	require.True(t, retry.Duplicate)
	require.Equal(t, first.UserMessageID, retry.UserMessageID)

	_, err = c.BeginCommand("ask-1", "What is this?", "", Photo{Sent: llm.ImagePart{MIME: "image/jpeg", Data: []byte("kettle")}})
	require.ErrorIs(t, err, ErrCommandConflict)
	_, err = c.BeginCommand("ask-1", "What is this?", "")
	require.ErrorIs(t, err, ErrCommandConflict, "the same words without the photo are another question")
}

func TestAPhotoLinkedFromElsewhereIsNotTaken(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	t.Cleanup(s.Close)
	c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)

	_, err = c.PublishPhotos(t.Context(), []llm.ImagePart{{URL: "https://example.com/teapot.jpg"}})

	require.ErrorContains(t, err, "data URIs")
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Empty(t, db.uploads)
}

func TestMorePhotosThanAMessageShowsAreNotTaken(t *testing.T) {
	db, client := newChat(t)
	s := NewForChat(client)
	t.Cleanup(s.Close)
	c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)
	images := make([]llm.ImagePart, maxFiles+1)
	for i := range images {
		images[i] = llm.ImagePart{MIME: "image/png", Data: []byte("png")}
	}

	_, err = c.PublishPhotos(t.Context(), images)

	require.ErrorContains(t, err, "at most 8 photos")
	db.mu.Lock()
	defer db.mu.Unlock()
	require.Empty(t, db.uploads)
}

func TestAReopenedConversationSaysAPhotoWasSentWithoutShowingItAgain(t *testing.T) {
	_, client := newChat(t)
	s := NewForChat(client)
	t.Cleanup(s.Close)
	c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)
	photos, err := c.PublishPhotos(t.Context(), []llm.ImagePart{{MIME: "image/jpeg", Data: []byte("a")}, {MIME: "image/jpeg", Data: []byte("b")}})
	require.NoError(t, err)
	_, err = c.BeginCommand("ask-1", "Which is nicer?", "", photos...)
	require.NoError(t, err)
	c.Observe(agent.ResponseDelta{Text: "The first."})
	c.Observe(agent.Responded{})
	saved(t, c)

	cold := NewForChat(client)
	t.Cleanup(cold.Close)
	history, _, err := cold.ContextForCaller(t.Context(), "customer", "agent", c.CID(), "alice")

	require.NoError(t, err)
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "Which is nicer?\n\n[sent with 2 photos, not shown again]"},
		{Role: llm.Assistant, Content: "The first."},
	}, history)
}

// A follow-up's reply is carried again as it finished, with its files, live and after the
// conversation is read back from Stream, as a scheduled task's retry sends it (AI-1882).
func TestAFollowUpsFilesAreReadBackLiveAndAcrossRestart(t *testing.T) {
	_, client := newChat(t)
	service := NewForChat(client)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	_, err = c.BeginFollowUp("st_run", "the task is due")
	require.NoError(t, err)
	render := sandbox.Attachment{Name: "teapot.png", MIME: "image/png", URL: "https://cdn.fake/image/teapot.png", Size: 3}
	c.Observe(agent.ToolStarted{ID: "run-1", Tool: sandbox.ToolName, StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "run-1", Tool: sandbox.ToolName, Result: "Attached.", Files: []sandbox.Attachment{render}})
	c.Observe(agent.ResponseDelta{Text: "Here is your teapot."})
	c.Observe(agent.Responded{})

	finished := FinishedReply{Customer: "customer", CID: c.CID(), MessageID: current(c).ID, CommandID: "st_run",
		Text: "Here is your teapot.", Files: []sandbox.Attachment{render}}
	require.Equal(t, finished, c.FollowUpFinished("st_run"))
	saved(t, c)
	c.Release()
	service.Close()

	recovered := NewForChat(client)
	t.Cleanup(recovered.Close)
	c, _, _, err = recovered.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	require.Equal(t, finished, c.FollowUpFinished("st_run"))
	require.Equal(t, FinishedReply{Customer: "customer", CID: c.CID(), CommandID: "st_other"}, c.FollowUpFinished("st_other"))
}

// A follow-up delivered again after an abandoned turn is read back with the files of the reply
// that answered it, not the abandoned turn's none.
func TestARetriedFollowUpsFilesAreReadBackAcrossRestart(t *testing.T) {
	_, client := newChat(t)
	service := NewForChat(client)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	_, err = c.BeginFollowUp("st_run", "the task is due")
	require.NoError(t, err)
	c.AbandonFollowUp("st_run")
	_, err = c.BeginFollowUp("st_run", "the task is due")
	require.NoError(t, err)
	render := sandbox.Attachment{Name: "teapot.png", MIME: "image/png", URL: "https://cdn.fake/image/teapot.png", Size: 3}
	c.Observe(agent.ToolStarted{ID: "run-1", Tool: sandbox.ToolName, StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "run-1", Tool: sandbox.ToolName, Result: "Attached.", Files: []sandbox.Attachment{render}})
	c.Observe(agent.ResponseDelta{Text: "Here is your teapot."})
	c.Observe(agent.Responded{})
	saved(t, c)
	c.Release()
	service.Close()

	recovered := NewForChat(client)
	t.Cleanup(recovered.Close)
	c, _, _, err = recovered.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	require.Equal(t, []sandbox.Attachment{render}, c.FollowUpFinished("st_run").Files)
}
