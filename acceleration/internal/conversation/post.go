package conversation

import (
	"context"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// Post writes a finished reply of the agent's into the conversation cid outside any turn: what
// the agent sends on its own once work it started is done, such as a picture drawn after the
// turn that asked for it ended. agent is the agent's Stream user, and app the Stream app the
// conversation is kept in.
//
// id and startedAt are the message's. Written again with the same two, after a crash between
// writing it and recording that it was written, it is found written and not sent twice
// (sendWith). A thread channel's reply leaves for its external thread as any finished reply does
// (OnFinishedReply), whose receiver drops one it has seen.
func (s *Service) Post(ctx context.Context, customer string, app int64, cid, agent, id string, startedAt time.Time,
	text string, files []sandbox.Attachment) error {
	client, err := s.chats.ForApp(ctx, customer, app)
	if err != nil {
		return err
	}
	at := startedAt.UTC()
	message := Message{
		ID: id, Role: "assistant", TextLayout: 1, Text: text, State: "completed",
		StartedAt: at, StateStartedAt: at, FinishedAt: &at, Sequence: 1, Tools: []Tool{}, Files: files,
	}
	c := &Conversation{service: s, data: ledger{CID: cid, Customer: customer, Agent: agent, StreamApp: app}}
	if err := c.sendWith(ctx, client, operation{Message: message, Create: true}, false, nil); err != nil {
		return err
	}
	if _, channel := channelOf(cid); strings.HasPrefix(channel, ThreadChannelPrefix) {
		if fn := s.finished.Load(); fn != nil {
			(*fn)(FinishedReply{Customer: customer, CID: cid, MessageID: id, Text: text, Files: files})
		}
	}
	return nil
}

// UploadTo uploads a file into the conversation cid, kept in the Stream app app, as uploader,
// and returns where it can be seen.
func (s *Service) UploadTo(ctx context.Context, customer string, app int64, cid, uploader string,
	file sandbox.File) (sandbox.Attachment, error) {
	client, err := s.chats.ForApp(ctx, customer, app)
	if err != nil {
		return sandbox.Attachment{}, err
	}
	channelType, channel := channelOf(cid)
	return Upload(ctx, client, channelType, channel, file, uploader)
}
