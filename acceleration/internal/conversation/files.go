package conversation

import (
	"context"
	"crypto/sha256"
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
// nothing, so Uploaded is empty on one. A voice note is carried as one too
// (PublishVoiceNote), its audio in Sent.
type Photo struct {
	Sent     llm.ImagePart
	Uploaded sandbox.Attachment
	// Heard is what a voice note said, which a retry of its question carries rather than
	// hear it again (CheckBegin). Empty for a photo.
	Heard string
}

// VoiceNote is the audio of one a person sent with a question, and what was heard in it.
type VoiceNote struct {
	MIME  string
	Audio []byte
	Heard string
}

// Sent is the voice note as a retry of its question is matched on.
func (n VoiceNote) Sent() Photo {
	return Photo{Sent: llm.ImagePart{MIME: n.MIME, Data: n.Audio}, Heard: n.Heard}
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

// PublishVoiceNote uploads the voice note a person sent with a question to the
// conversation's channel, as them, for their message to show beside its transcript.
func (c *Conversation) PublishVoiceNote(ctx context.Context, note VoiceNote) (Photo, error) {
	file := sandbox.File{Name: "voice-note." + strings.TrimPrefix(note.MIME, "audio/"), MIME: note.MIME, Data: note.Audio}
	uploaded, err := c.upload(ctx, file, c.userAuthor())
	if err != nil {
		return Photo{}, err
	}
	published := note.Sent()
	published.Uploaded = uploaded
	return published, nil
}

// PublishDocument uploads a file a person sent with a question to the conversation's channel,
// as them, for their message to show.
func (c *Conversation) PublishDocument(ctx context.Context, file sandbox.File) (Photo, error) {
	uploaded, err := c.upload(ctx, file, c.userAuthor())
	if err != nil {
		return Photo{}, err
	}
	published := DocumentSent(file)
	published.Uploaded = uploaded
	return published, nil
}

// DocumentSent is a file sent with a question as a retry of it is matched on: by its digest,
// so a retry holds no second copy of a file of megabytes.
func DocumentSent(file sandbox.File) Photo {
	digest := sha256.Sum256(file.Data)
	return Photo{Sent: llm.ImagePart{MIME: file.MIME, Data: digest[:]}}
}

// upload stores a file on the conversation's channel as uploader.
func (c *Conversation) upload(ctx context.Context, file sandbox.File, uploader string) (sandbox.Attachment, error) {
	// The file goes where the conversation is kept.
	client, err := c.chat(ctx)
	if err != nil {
		return sandbox.Attachment{}, err
	}
	channelType, id := channelOf(c.data.CID)
	return Upload(ctx, client, channelType, id, file, uploader)
}

// Upload stores a file on the channel channelType:channelID as uploader, as an image when its MIME
// type is image/*, and returns where Chat's CDN serves it. The channel must exist.
func Upload(ctx context.Context, client *getstream.Stream, channelType, channelID string, file sandbox.File, uploader string) (sandbox.Attachment, error) {
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

	channel := client.Chat().Channel(channelType, channelID)
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

// MergeFiles adds files to a reply, a later file of the same name replacing the earlier.
func MergeFiles(existing, additions []sandbox.Attachment) []sandbox.Attachment {
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
func fileAttachments(files []sandbox.Attachment, person bool) []map[string]any {
	attachments := make([]map[string]any, 0, len(files))
	for _, file := range files {
		attachment := map[string]any{"title": file.Name, "mime_type": file.MIME, "file_size": file.Size}
		switch {
		case strings.HasPrefix(file.MIME, "image/"):
			attachment["type"] = "image"
			attachment["image_url"] = file.URL
			attachment["fallback"] = file.Name
		case voiceRecording(file, person):
			attachment["type"] = voiceRecordingType
			attachment["asset_url"] = file.URL
		default:
			attachment["type"] = "file"
			attachment["asset_url"] = file.URL
		}
		attachments = append(attachments, attachment)
	}
	return attachments
}

// chatFileAttachments are files as a new message carries them, the way fileAttachments are
// for a partial update: Chat reads a new message's custom attachment fields from "custom".
func chatFileAttachments(files []sandbox.Attachment, person bool) []getstream.Attachment {
	attachments := make([]getstream.Attachment, 0, len(files))
	for _, file := range files {
		attachment := getstream.Attachment{Title: &file.Name, Custom: map[string]any{"mime_type": file.MIME, "file_size": file.Size}}
		switch {
		case strings.HasPrefix(file.MIME, "image/"):
			attachment.Type, attachment.ImageUrl, attachment.Fallback = getstream.PtrTo("image"), &file.URL, &file.Name
		case voiceRecording(file, person):
			attachment.Type, attachment.AssetUrl = getstream.PtrTo(voiceRecordingType), &file.URL
		default:
			attachment.Type, attachment.AssetUrl = getstream.PtrTo("file"), &file.URL
		}
		attachments = append(attachments, attachment)
	}
	return attachments
}

// voiceRecordingType is the attachment type Stream Chat's SDKs play as a voice recording
// rather than offer as a download (stream-chat-js isVoiceRecordingAttachment).
const voiceRecordingType = "voiceRecording"

// voiceRecording is whether file is a voice note: audio on a person's message, which only
// carries what they sent. Audio the agent made is a file like any other.
func voiceRecording(file sandbox.Attachment, person bool) bool {
	return person && strings.HasPrefix(file.MIME, "audio/")
}

// Images are the photos a person sent with m as a model is handed them: each by the URL Chat
// serves it from, which a read of the message gives afresh.
func Images(m Message) []llm.ImagePart {
	return imagesIn(m.Files)
}

func imagesIn(files []sandbox.Attachment) []llm.ImagePart {
	var images []llm.ImagePart
	for _, file := range files {
		if strings.HasPrefix(file.MIME, "image/") && file.URL != "" {
			images = append(images, llm.ImagePart{URL: file.URL, MIME: file.MIME})
		}
	}
	return images
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
		case (*attachment.Type == "file" || *attachment.Type == voiceRecordingType) && attachment.AssetUrl != nil:
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
