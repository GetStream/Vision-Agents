package egress

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"
)

// maxLFSBody is the most of a Git LFS batch body the proxy reads, 1 MiB, as maxGraphQLBody: git-lfs
// asks for 100 objects a batch by default (lfs.transfer.batchSize, tq/transfer_queue.go
// defaultBatchSize), each an oid of 64 hex digits and a size, about 10 KiB. A larger body is
// refused.
const maxLFSBody = 1 << 20

// lfsMediaType is the batch API's media type: «Content-Type: application/vnd.git-lfs+json», «The
// client may also include a charset=utf-8 parameter»
// (https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md).
const lfsMediaType = "application/vnd.git-lfs+json"

var (
	// lfsKeys are the members of a batch request (batch.md, «Requests»): operation, transfers,
	// ref, objects and hash_algo. git-lfs sends those alone (tq/api.go, batchRequest). Any other is
	// refused, as is a member added later: a host could read an operation from it.
	lfsKeys = map[string]bool{"operation": true, "transfers": true, "ref": true, "objects": true, "hash_algo": true}
	// lfsObjectKeys are the members of one object of a request, oid and size (batch.md), all git-lfs
	// sends (tq/transfer_queue.go, batch.ToTransfers).
	lfsObjectKeys = map[string]bool{"oid": true, "size": true}
)

// onlyDownloads says whether r is one Git LFS batch request that asks to download, or why not.
// It reads the body and puts back the same bytes, so the host gets them as the sandbox sent them.
// Each check refuses a request whose operation the host could read another way than the proxy:
// from the URL, another media type or charset, a content coding, a member the proxy does not
// read, or a key sent twice.
func onlyDownloads(r *http.Request) error {
	if r.URL.RawQuery != "" {
		return errors.New("an LFS batch POST with a URL query")
	}
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || len(r.Header.Values("Content-Type")) != 1 || media != lfsMediaType ||
		(params["charset"] != "" && !strings.EqualFold(params["charset"], "utf-8")) {
		return errors.New("an LFS batch POST that is not " + lfsMediaType + " in UTF-8")
	}
	for _, coding := range r.Header.Values("Content-Encoding") {
		for _, token := range strings.Split(coding, ",") {
			if token = strings.TrimSpace(token); token != "" && !strings.EqualFold(token, "identity") {
				return errors.New("an LFS batch POST with a content coding")
			}
		}
	}
	// A body cut short is checked as far as it came, and the host gets those bytes.
	body, _ := io.ReadAll(io.LimitReader(r.Body, maxLFSBody+1))
	r.Body = io.NopCloser(bytes.NewReader(body))
	switch {
	case len(body) > maxLFSBody:
		return errors.New("an LFS batch POST over the size the proxy reads")
	case !json.Valid(body):
		return errors.New("an LFS batch POST whose body is not JSON")
	}
	members, err := lfsMembers(body, lfsKeys)
	if err != nil {
		return err
	}
	// The operation is the request's alone: an object names none (batch.md), and lfsObjectKeys
	// refuses one that tries. «download» as the spec writes it: a host that read Download as a
	// download is not one to guess at.
	var operation string
	if json.Unmarshal(members["operation"], &operation) != nil || operation != "download" {
		return errors.New("an LFS batch request whose operation is not download")
	}
	var objects []json.RawMessage
	if json.Unmarshal(members["objects"], &objects) != nil || objects == nil {
		return errors.New("an LFS batch request whose objects are no list")
	}
	for _, object := range objects {
		if _, err := lfsMembers(object, lfsObjectKeys); err != nil {
			return err
		}
	}
	return nil
}

// lfsMembers is raw's members, when it is a JSON object whose every key is one of keys, once. It
// reads raw a token at a time, as object does: decoding into a map keeps the last of two equal
// keys, and a host could keep the first. raw is valid JSON (onlyDownloads).
func lfsMembers(raw []byte, keys map[string]bool) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if open, _ := decoder.Token(); open != json.Delim('{') {
		return nil, errors.New("an LFS batch request or object that is no JSON object")
	}
	members := map[string]json.RawMessage{}
	for decoder.More() {
		key, _ := decoder.Token()
		var value json.RawMessage
		_ = decoder.Decode(&value)
		name, _ := key.(string)
		if _, twice := members[name]; twice || !keys[name] {
			return nil, errors.New("an LFS batch request or object with a member the proxy does not read, or one twice")
		}
		members[name] = value
	}
	return members, nil
}

// lfsAPI says whether path is under a Git LFS API: git-lfs puts it at its remote's URL with
// .git/info/lfs, «/info/lfs» after a URL ending in .git (lfsapi/endpoint_finder.go:224-229, v3.8.1),
// and a read's pattern can take it without .git, as GitHub's /*/*/info/lfs/objects/batch does. Its
// batch, locks and locks/verify are under it.
func lfsAPI(path string) bool {
	return strings.Contains(path, "/info/lfs/")
}

// lfsError is the body of a Git LFS API error, its message alone (batch.md, «Response Errors»).
type lfsError struct {
	Message string `json:"message"`
}

// refuseRequest is refuse for a request in a tunnel: to a Git LFS API, the message goes in an LFS
// error, which git-lfs prints. git-lfs reads the message of an error answer only from a body in the
// LFS or the JSON media type, and prints «Authorization error: <url>» in place of any other 403's
// (lfshttp/errors.go handleResponse and defaultError, lfshttp/lfshttp.go DecodeJSON, v3.3.0 and
// v3.8.1; probe of both in the PR). A 403 on locks/verify stays a 403: git-lfs reads a 404 or a 501
// there as a host with no locking API and writes locksverify=false into the repository's config
// (locking/locks.go:272-275, commands/lockverifier.go:65-66, commands/uploader.go disableFor, v3.8.1).
func refuseRequest(w http.ResponseWriter, r *http.Request, host, decision string, status int, message string) {
	if !lfsAPI(r.URL.Path) {
		refuse(w, host, r.Method, decision, status, message)
		return
	}
	count(host, r.Method, decision, status)
	body, _ := json.Marshal(lfsError{Message: message})
	w.Header().Del("Content-Length")
	w.Header().Set("Content-Type", lfsMediaType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(append(body, '\n'))
}
