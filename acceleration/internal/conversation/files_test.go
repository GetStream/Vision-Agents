package conversation

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/stretchr/testify/require"
)

func TestAnImageIsUploadedToTheChannelAsTheAgent(t *testing.T) {
	db, client := newChat(t)
	s := newService(client)
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
	s := newService(client)
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
	s := newService(client)
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
	s := newService(client)
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
	cold := newService(client)
	defer cold.Close()
	page, err := cold.History(context.Background(), "customer", "artist", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	require.Empty(t, page.Messages[0].Files, "the person's own message carries nothing")
	require.Equal(t, []sandbox.Attachment{render}, page.Messages[1].Files)
}

func TestFilesForWorkTheReplyNeverStartedAreNotAttached(t *testing.T) {
	_, client := newChat(t)
	s := newService(client)
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
