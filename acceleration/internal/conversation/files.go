package conversation

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	getstream "github.com/GetStream/getstream-go/v5"
)

// maxFiles is how many files one reply may carry.
const maxFiles = 8

// Publish uploads a file the agent made to the conversation's channel, as the agent, and
// returns where it can be seen. The file is attached to a reply only once the work that
// made it settles; until then it is an upload nothing points at.
func (c *Conversation) Publish(ctx context.Context, file sandbox.File) (sandbox.Attachment, error) {
	name := filepath.Base(file.Name)
	if name == "." || name == string(filepath.Separator) || len(file.Data) == 0 {
		return sandbox.Attachment{}, errors.New("conversation: a file needs a name and something in it")
	}
	// The Chat client uploads from a path, and names the upload after it.
	dir, err := os.MkdirTemp("", "agent-file-")
	if err != nil {
		return sandbox.Attachment{}, err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, file.Data, 0o600); err != nil {
		return sandbox.Attachment{}, err
	}

	channel := c.service.client.Chat().Channel("agent", strings.TrimPrefix(c.data.CID, "agent:"))
	uploader := &getstream.OnlyUserID{ID: c.data.Agent}
	var url *string
	if strings.HasPrefix(file.MIME, "image/") {
		uploaded, err := channel.UploadChannelImage(ctx, &getstream.UploadChannelImageRequest{File: &path, User: uploader})
		if err != nil {
			return sandbox.Attachment{}, err
		}
		url = uploaded.Data.File
	} else {
		uploaded, err := channel.UploadChannelFile(ctx, &getstream.UploadChannelFileRequest{File: &path, User: uploader})
		if err != nil {
			return sandbox.Attachment{}, err
		}
		url = uploaded.Data.File
	}
	if url == nil || *url == "" {
		return sandbox.Attachment{}, errors.New("conversation: Chat stored the file without saying where")
	}
	return sandbox.Attachment{Name: name, MIME: file.MIME, URL: *url, Size: len(file.Data)}, nil
}

// mergeFiles adds files to a reply, a later file of the same name replacing the earlier.
func mergeFiles(existing, additions []sandbox.Attachment) []sandbox.Attachment {
	merged := append([]sandbox.Attachment(nil), existing...)
	for _, file := range additions {
		merged = slices.DeleteFunc(merged, func(kept sandbox.Attachment) bool { return kept.Name == file.Name })
		merged = append(merged, file)
	}
	if len(merged) > maxFiles {
		merged = merged[len(merged)-maxFiles:]
	}
	return merged
}

// fileAttachments are files as Chat shows them: an image inline, anything else as a
// download.
func fileAttachments(files []sandbox.Attachment) []map[string]any {
	attachments := make([]map[string]any, 0, len(files))
	for _, file := range files {
		attachment := map[string]any{"title": file.Name, "mime_type": file.MIME, "file_size": file.Size}
		if strings.HasPrefix(file.MIME, "image/") {
			attachment["type"] = "image"
			attachment["image_url"] = file.URL
			attachment["fallback"] = file.Name
		} else {
			attachment["type"] = "file"
			attachment["asset_url"] = file.URL
		}
		attachments = append(attachments, attachment)
	}
	return attachments
}

// filesFromAttachments reads back the files fileAttachments wrote.
func filesFromAttachments(attachments []getstream.Attachment) []sandbox.Attachment {
	var files []sandbox.Attachment
	for _, attachment := range attachments {
		if attachment.Type == nil || attachment.Title == nil || len(files) == maxFiles {
			continue
		}
		file := sandbox.Attachment{Name: *attachment.Title}
		switch {
		case *attachment.Type == "image" && attachment.ImageUrl != nil:
			file.URL = *attachment.ImageUrl
		case *attachment.Type == "file" && attachment.AssetUrl != nil:
			file.URL = *attachment.AssetUrl
		default:
			continue
		}
		file.MIME, _ = attachment.Custom["mime_type"].(string)
		if size, ok := attachment.Custom["file_size"].(float64); ok {
			file.Size = int(size)
		}
		files = append(files, file)
	}
	return files
}
