package channelbridge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxPhotoBytes bounds one photo downloaded from a provider: api's maxRunImageBytes, 5 MB, the
// most the model is handed in one image, so a bigger one would be fetched for nothing. Meta
// takes images of at most 5 MB (https://developers.facebook.com/docs/whatsapp/cloud-api/reference/media,
// «Supported media types», opened November 4, 2026).
const maxPhotoBytes = 5 << 20

// maxLookupBytes bounds a media lookup's answer, a small JSON object: Meta's holds url,
// mime_type, sha256, file_size, id and messaging_product (same page, «Get media URL»). A choice.
const maxLookupBytes = 64 << 10

// downloadTimeout bounds downloading one message's photos, the lookup and the file of each
// together: the 10 s the router gives one connector request (cmd/router connectorHTTPTimeout).
// It is part of the attempt's writeTimeout, which leaves the rest for the writes.
const downloadTimeout = 10 * time.Second

// fetched is a photo download fetched: at is its place in the message's media.
type fetched struct {
	at   int
	file sandbox.File
}

// download fetches the photos of message that its provider names by id (core.Media.ID), as
// the manifest of the connection's pinned revision finds them (MessageRule.MediaLookup), with
// the connection's own client, which carries its credential, for upload to store. named is
// how many files the message names by id, photos or not: another file is not downloaded, as
// photos writes no other file. Nor is a photo an earlier attempt of the row uploaded
// (core.Media.Uploaded): that upload is written instead. A photo that cannot be downloaded is
// left out and logged with ids only, as is its reason: the message goes on with its text, and
// the provider's id is no use to the agent. An error is the store failing, which the attempt
// is tried again for.
func (b *Bridge) download(ctx context.Context, thread store.ChannelThread, message core.InboundMessage, log *slog.Logger) (files []fetched, named int, err error) {
	var wanted []int
	for at, file := range message.Media {
		if file.ID == "" {
			continue
		}
		named++
		if strings.HasPrefix(file.MIMEType, "image/") && file.Uploaded == "" {
			wanted = append(wanted, at)
		}
	}
	if len(wanted) == 0 {
		return nil, named, nil
	}
	connection, err := b.store.ConnectorConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil {
		return nil, named, err
	}
	scheme, found := b.schemes[connection.AuthScheme]
	if !found {
		return nil, named, stack.Wrap(fmt.Errorf("%w: %q", store.ErrUnregisteredScheme, connection.AuthScheme))
	}
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return nil, named, err
	}
	resolved, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		return nil, named, stack.Wrap(err)
	}
	client := b.transports.Client(core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}, scheme)
	// A refusal of either GET is the file's, never the credential's: a link that has expired
	// must not move the connection to needs_reauthorization (core.Unjudged).
	downloading, cancel := context.WithTimeout(core.Unjudged(ctx), b.downloadTimeout)
	defer cancel()
	for i, at := range wanted {
		data, err := fetchPhoto(downloading, client, resolved, message.Media[at].ID)
		if err != nil {
			log.Warn("a photo the provider names by id could not be downloaded: the message goes on without it",
				"connection", connection.ID, "revision", connection.DefinitionRevision, "photo", i+1, "error", err)
			continue
		}
		mime := http.DetectContentType(data)
		files = append(files, fetched{at: at,
			file: sandbox.File{Name: fmt.Sprintf("photo-%d.%s", i+1, strings.TrimPrefix(mime, "image/")), MIME: mime, Data: data}})
	}
	return files, named, nil
}

// nothingToWrite is whether a message has nothing for the agent once download, then upload,
// is done: it names files by id, none of which it has as one of its kept photos, and has no
// text and no photo at a URL. A message that names no file by id, such as a Linq video alone,
// is written as before.
func nothingToWrite(message core.InboundMessage, kept, named int) bool {
	return named > 0 && kept == 0 && message.Text == "" && len(photos(message.Media)) == 0
}

// fetchPhoto looks the file id names up, and downloads it from where the answer says, when that
// is on the lookup's host (core.ResolvedManifest.MediaFile), as an image: the bytes, sniffed as
// the model's fetch does (api.fetchImages), must be one.
func fetchPhoto(ctx context.Context, client *http.Client, resolved core.ResolvedManifest, id string) ([]byte, error) {
	target, err := resolved.MediaLookupURL(id)
	if err != nil {
		return nil, err
	}
	answer, err := get(ctx, client, target, maxLookupBytes)
	if err != nil {
		return nil, err
	}
	source, err := resolved.MediaFile(answer)
	if err != nil {
		return nil, err
	}
	data, err := get(ctx, client, source, maxPhotoBytes)
	if err != nil {
		return nil, err
	}
	if sniffed := http.DetectContentType(data); !strings.HasPrefix(sniffed, "image/") {
		return nil, stack.Wrap(fmt.Errorf("channelbridge: the file is %s, not an image", sniffed))
	}
	return data, nil
}

// get is the body of a GET of target that answers 200, of at most limit bytes. Its errors
// name the host alone: the URL of a provider's file may be one that reads it.
func get(ctx context.Context, client *http.Client, target string, limit int64) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, stack.Wrap(errors.New("channelbridge: a media URL is not a request URL"))
	}
	response, err := client.Do(request)
	if err != nil {
		var failed *url.Error
		if errors.As(err, &failed) {
			err = failed.Err
		}
		return nil, stack.Wrap(fmt.Errorf("channelbridge: GET on %s: %w", request.URL.Host, err))
	}
	defer response.Body.Close() //nolint:errcheck // the answer has been read
	if response.StatusCode != http.StatusOK {
		return nil, stack.Wrap(fmt.Errorf("channelbridge: %s answered a GET with %d", request.URL.Host, response.StatusCode))
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("channelbridge: reading the answer of %s: %w", request.URL.Host, err))
	}
	if int64(len(body)) > limit {
		return nil, stack.Wrap(fmt.Errorf("channelbridge: %s answered more than %d bytes", request.URL.Host, limit))
	}
	return body, nil
}
