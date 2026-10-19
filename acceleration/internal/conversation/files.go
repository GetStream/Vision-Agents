package conversation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	getstream "github.com/GetStream/getstream-go/v5"
)

// maxFiles is how many files one reply may carry.
const maxFiles = 8

// Publish uploads a file the agent made to the conversation's channel, as the agent, and
// returns where it can be seen. The file is attached to a reply only once the work that
// made it settles; until then it is an upload nothing points at.
func (c *Conversation) Publish(ctx context.Context, file sandbox.File) (sandbox.Attachment, error) {
	return c.upload(ctx, file, c.data.Agent)
}

// Photo is one a person sent with a question: as sent, which a retry of the question is
// matched on, and where it was uploaded, which their message shows. A retry uploads
// nothing, so Uploaded is empty on one.
type Photo struct {
	Sent     llm.ImagePart
	Uploaded sandbox.Attachment
}

// PublishPhotos uploads the photos a person sent with a question to the conversation's
// channel, as them. The conversation keeps its photos in Chat, so each one is sent as its
// bytes, a data URI, rather than as a link to somewhere else.
func (c *Conversation) PublishPhotos(ctx context.Context, images []llm.ImagePart) ([]Photo, error) {
	if len(images) > maxFiles {
		return nil, fmt.Errorf("conversation: a question carries at most %d photos", maxFiles)
	}
	photos := make([]Photo, 0, len(images))
	for index, image := range images {
		if len(image.Data) == 0 {
			return nil, errors.New("conversation: a saved conversation takes photos as data URIs")
		}
		file := sandbox.File{Name: fmt.Sprintf("photo-%d.%s", index+1, strings.TrimPrefix(image.MIME, "image/")), MIME: image.MIME, Data: image.Data}
		uploaded, err := c.upload(ctx, file, c.userAuthor())
		if err != nil {
			return nil, err
		}
		photos = append(photos, Photo{Sent: image, Uploaded: uploaded})
	}
	return photos, nil
}

// upload stores a file on the conversation's channel as uploader.
func (c *Conversation) upload(ctx context.Context, file sandbox.File, uploader string) (sandbox.Attachment, error) {
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

	// The file goes where the conversation is kept.
	client, err := c.chat(ctx)
	if err != nil {
		return sandbox.Attachment{}, err
	}
	channel := client.Chat().Channel("agent", strings.TrimPrefix(c.data.CID, "agent:"))
	by := &getstream.OnlyUserID{ID: uploader}
	var url *string
	if strings.HasPrefix(file.MIME, "image/") {
		uploaded, err := channel.UploadChannelImage(ctx, &getstream.UploadChannelImageRequest{File: &path, User: by})
		if err != nil {
			return sandbox.Attachment{}, err
		}
		url = uploaded.Data.File
	} else {
		uploaded, err := channel.UploadChannelFile(ctx, &getstream.UploadChannelFileRequest{File: &path, User: by})
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

// chatFileAttachments are files as a new message carries them, the way fileAttachments are
// for a partial update: Chat reads a new message's custom attachment fields from "custom".
func chatFileAttachments(files []sandbox.Attachment) []getstream.Attachment {
	attachments := make([]getstream.Attachment, 0, len(files))
	for _, file := range files {
		attachment := getstream.Attachment{Title: &file.Name, Custom: map[string]any{"mime_type": file.MIME, "file_size": file.Size}}
		if strings.HasPrefix(file.MIME, "image/") {
			attachment.Type, attachment.ImageUrl, attachment.Fallback = getstream.PtrTo("image"), &file.URL, &file.Name
		} else {
			attachment.Type, attachment.AssetUrl = getstream.PtrTo("file"), &file.URL
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
