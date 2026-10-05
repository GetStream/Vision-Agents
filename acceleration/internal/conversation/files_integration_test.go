//go:build integration

package conversation

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// TestARenderReachesStreamChatAndSurvivesAReload is the whole path against real Chat: the
// picture is uploaded to the channel, attached to the reply, and still there, and still
// downloadable, when the conversation is read back by a service that never held it.
func TestARenderReachesStreamChatAndSurvivesAReload(t *testing.T) {
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		t.Skip("STREAM_API_KEY and STREAM_API_SECRET not set")
	}
	ctx := context.Background()
	service, err := New()
	require.NoError(t, err)
	defer service.Close()
	artist := fmt.Sprintf("files-test-%d", time.Now().UnixNano())
	c, _, _, err := service.Open(ctx, "examples", artist, "")
	require.NoError(t, err)
	defer c.Release()

	picture := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for x := range 32 {
		for y := range 32 {
			picture.Set(x, y, color.RGBA{R: uint8(x * 8), G: 64, B: uint8(y * 8), A: 255})
		}
	}
	var encoded bytes.Buffer
	require.NoError(t, png.Encode(&encoded, picture))

	require.NoError(t, c.Begin("render a gradient"))
	c.Observe(agent.Delegated{TaskID: "task-1", Skill: "render"})
	c.Observe(agent.Responded{Text: "Rendering it.", PendingWork: true})
	render, err := c.Publish(ctx, sandbox.File{Name: "gradient.png", MIME: "image/png", Data: encoded.Bytes()})
	require.NoError(t, err)
	c.Observe(agent.TaskSettled{TaskID: "task-1", Skill: "render", Text: "A gradient.", Files: []sandbox.Attachment{render}})
	c.Observe(agent.ResponseDelta{Text: " Here is your gradient."})
	c.Observe(agent.Responded{})
	require.Eventually(t, func() bool { return current(c).Saved }, 30*time.Second, 100*time.Millisecond)

	cold, err := New()
	require.NoError(t, err)
	defer cold.Close()
	page, err := cold.History(ctx, "examples", artist, c.CID(), "")
	require.NoError(t, err)
	require.NotEmpty(t, page.Messages)
	reply := page.Messages[len(page.Messages)-1]
	require.Len(t, reply.Files, 1, "the reply read back from Chat carries the render")
	require.Equal(t, "gradient.png", reply.Files[0].Name)
	require.Equal(t, "image/png", reply.Files[0].MIME)

	response, err := http.Get(reply.Files[0].URL)
	require.NoError(t, err)
	defer response.Body.Close()
	require.Equal(t, http.StatusOK, response.StatusCode)
	served, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	_, err = png.Decode(bytes.NewReader(served))
	require.NoError(t, err, "what Chat serves at the attachment is the picture")
	t.Logf("conversation %s, render at %s", c.CID(), reply.Files[0].URL)
}
