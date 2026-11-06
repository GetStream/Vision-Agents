//go:build integration

package egress_test

import (
	"bufio"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// lfsBatch is git lfs's batch target on the test's git host.
const lfsBatch = "https://github.test/o/r.git/info/lfs/objects/batch"

// lfsDownload is the body git-lfs sends for git lfs pull (tq/api.go, batchRequest, at
// debca01ac1ca27a20cc30ac6870151b48c014881): the operation, its objects, the ref and the hash.
const lfsDownload = `{"operation":"download","objects":[{"oid":"fb6fd6c3893c67503897449a27d66246623e8de42a498f7f81f72c3ecad16f9c","size":872}],` +
	`"ref":{"name":"refs/heads/main"},"hash_algo":"sha256"}`

// lfsErrorBody is the body of the proxy's refusal of a Git LFS API request: an LFS error whose message
// git-lfs prints (lfshttp/errors.go handleResponse, v3.3.0 and v3.8.1), what a fake that answered
// so showed both versions printing (probe in the PR).
func lfsErrorBody(message string) string {
	body, _ := json.Marshal(map[string]string{"message": message})
	return string(body) + "\n"
}

// lfsRequest is one request of git lfs in the sandbox, and the status the proxy answers it with.
type lfsRequest struct {
	name, method, target, body string
	header                     http.Header
	status                     int
}

// lfsCase is a batch POST of body with the batch API's media type, answered with status.
func lfsCase(name, body string, status int) lfsRequest {
	return lfsRequest{name: name, method: http.MethodPost, target: lfsBatch, body: body, status: status}
}

// readLFS gives the test's grant, for github.test, git's upload-pack read and an LFS read of the
// batch API whose objects download from media.github.test.
func (s *ProxySuite) readLFS() {
	grant := s.grant(s.placeholder, "github.test")
	grant.Reads = []core.SandboxRead{{Host: "github.test", Method: http.MethodPost, Path: "/*/*/git-upload-pack"},
		{Host: "github.test", Path: "/*/*/info/lfs/objects/batch", LFS: true, Downloads: []string{"media.github.test"}}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
}

// sendLFS sends each request as git does, the placeholder as Basic auth's password. One the proxy
// lets through reaches the host with the token and its body byte for byte, and the sandbox reads
// the host's echo of the token as the placeholder; one it refuses reaches no host, asks for no
// token, and gets read-only's answer.
func (s *ProxySuite) sendLFS(requests []lfsRequest) {
	for _, c := range requests {
		s.granted.forget()
		asked := s.asked
		request := s.request(c.method, c.target, basic("x-access-token", s.placeholder), c.body)
		request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")
		for name, values := range c.header {
			request.Header[name] = values
		}

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, c.name)
		s.Equal(c.status, response.StatusCode, c.name)
		if c.status != http.StatusOK {
			s.Empty(s.granted.requests(), c.name)
			s.Equal(asked, s.asked, c.name)
			s.Equal(lfsErrorBody("egress: "+c.method+" "+request.URL.Host+request.URL.EscapedPath()+" refused: code in the sandbox only reads through a "+
				"connector; write with the connector's tools"), body, c.name)
			s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"), c.name)
			continue
		}
		s.Require().Len(s.granted.requests(), 1, c.name)
		s.Equal(basic("x-access-token", "gho_first-token-of-the-connection"), s.granted.requests()[0].Authorization, c.name)
		s.Equal(c.body, s.granted.requests()[0].Body, c.name)
		s.Equal(basic("x-access-token", s.placeholder), response.Header.Get("X-Echo-Authorization"), c.name)
		s.Contains(body, basic("x-access-token", s.placeholder), c.name)
	}
}

// TestAnLFSReadTakesABatchThatDownloadsAlone: git lfs pull's batch POST reaches the host with the
// token, and the host's echo of it comes back as the placeholder; a batch that asks to upload, or
// whose operation the host could read as another than download, reaches no host
// (https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md, «operation - Should be download
// or upload»).
func (s *ProxySuite) TestAnLFSReadTakesABatchThatDownloadsAlone() {
	s.readLFS()

	s.sendLFS([]lfsRequest{
		lfsCase("git lfs pull", lfsDownload, http.StatusOK),
		lfsCase("a download with transfers", `{"operation":"download","transfers":["basic"],"objects":[{"oid":"a","size":1}]}`, http.StatusOK),
		lfsCase("a download of no objects", `{"operation":"download","objects":[]}`, http.StatusOK),
		lfsCase("a download with a null ref", `{"operation":"download","ref":null,"objects":[{"oid":"a","size":1}]}`, http.StatusOK),
		lfsCase("git lfs push", strings.Replace(lfsDownload, `"download"`, `"upload"`, 1), http.StatusForbidden),
		lfsCase("Download", strings.Replace(lfsDownload, `"download"`, `"Download"`, 1), http.StatusForbidden),
		lfsCase("DOWNLOAD", strings.Replace(lfsDownload, `"download"`, `"DOWNLOAD"`, 1), http.StatusForbidden),
		lfsCase("download with a space", strings.Replace(lfsDownload, `"download"`, `"download "`, 1), http.StatusForbidden),
		lfsCase("download escaped", strings.Replace(lfsDownload, `"download"`, `"\u0064ownload"`, 1), http.StatusOK),
		lfsCase("an operation that is no string", `{"operation":["download"],"objects":[]}`, http.StatusForbidden),
		lfsCase("a null operation", `{"operation":null,"objects":[]}`, http.StatusForbidden),
		lfsCase("no operation", `{"objects":[{"oid":"a","size":1}]}`, http.StatusForbidden),
		lfsCase("an unknown operation", `{"operation":"verify","objects":[]}`, http.StatusForbidden),
		lfsCase("download, then upload", `{"operation":"download","operation":"upload","objects":[]}`, http.StatusForbidden),
		lfsCase("upload, then download", `{"operation":"upload","operation":"download","objects":[]}`, http.StatusForbidden),
		lfsCase("upload in an object", `{"operation":"download","objects":[{"oid":"a","size":1},{"oid":"b","size":1,"operation":"upload"}]}`,
			http.StatusForbidden),
		lfsCase("an object with actions", `{"operation":"download","objects":[{"oid":"a","size":1,"actions":{"upload":{}}}]}`, http.StatusForbidden),
		lfsCase("an object's key twice", `{"operation":"download","objects":[{"oid":"a","oid":"b","size":1}]}`, http.StatusForbidden),
		lfsCase("an object that is no object", `{"operation":"download","objects":["a"]}`, http.StatusForbidden),
		lfsCase("objects that are no list", `{"operation":"download","objects":{"oid":"a","size":1}}`, http.StatusForbidden),
		lfsCase("null objects", `{"operation":"download","objects":null}`, http.StatusForbidden),
		lfsCase("no objects", `{"operation":"download"}`, http.StatusForbidden),
		lfsCase("a member the proxy does not read", `{"operation":"download","objects":[],"locks":true}`, http.StatusForbidden),
		lfsCase("an empty object", `{}`, http.StatusForbidden),
		lfsCase("null", `null`, http.StatusForbidden),
		lfsCase("a list of batches", `[`+lfsDownload+`]`, http.StatusForbidden),
		lfsCase("no JSON", `operation=download`, http.StatusForbidden),
		lfsCase("no body", ``, http.StatusForbidden),
	})
	s.Contains(s.logs.String(), `why="an LFS batch request whose operation is not download"`)
	s.Contains(s.logs.String(), `why="an LFS batch request or object with a member the proxy does not read, or one twice"`)
	s.Contains(s.logs.String(), `why="an LFS batch request whose objects are no list"`)
	s.Contains(s.logs.String(), `why="an LFS batch request or object that is no JSON object"`)
	s.Contains(s.logs.String(), `why="an LFS batch POST whose body is not JSON"`)
}

// TestAnLFSReadTakesItsMediaTypeAndAMiBAtMost: the proxy reads a batch as the batch API's media
// type in UTF-8, as sent, up to 1 MiB, at the batch path alone; any other is refused, as a host
// could read its operation another way.
func (s *ProxySuite) TestAnLFSReadTakesItsMediaTypeAndAMiBAtMost() {
	s.readLFS()
	// 1 MiB is egress.maxLFSBody; JSON takes whitespace after its value (RFC 8259 section 2).
	mib := lfsDownload + strings.Repeat(" ", 1<<20-len(lfsDownload))
	typed := func(name, contentType string, status int) lfsRequest {
		return lfsRequest{name: name, method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: status,
			header: http.Header{"Content-Type": {contentType}}}
	}

	s.sendLFS([]lfsRequest{
		lfsCase("a body of 1 MiB", mib, http.StatusOK),
		lfsCase("a body of 1 MiB and a byte", mib+" ", http.StatusForbidden),
		typed("no charset", "application/vnd.git-lfs+json", http.StatusOK),
		typed("UTF-8 in capitals", "application/vnd.git-lfs+json; charset=UTF-8", http.StatusOK),
		typed("Latin-1", "application/vnd.git-lfs+json; charset=iso-8859-1", http.StatusForbidden),
		typed("JSON", "application/json", http.StatusForbidden),
		typed("a form", "application/x-www-form-urlencoded", http.StatusForbidden),
		typed("no media type", "", http.StatusForbidden),
		{name: "two media types", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden,
			header: http.Header{"Content-Type": {"application/vnd.git-lfs+json", "application/json"}}},
		{name: "gzip", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden,
			header: http.Header{"Content-Encoding": {"gzip"}}},
		{name: "identity", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusOK,
			header: http.Header{"Content-Encoding": {"identity"}}},
		{name: "identity then gzip on two lines", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden,
			header: http.Header{"Content-Encoding": {"identity", "gzip"}}},
		{name: "identity then gzip in a list", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden,
			header: http.Header{"Content-Encoding": {"identity, gzip"}}},
		{name: "a URL query", method: http.MethodPost, target: lfsBatch + "?operation=upload", body: lfsDownload, status: http.StatusForbidden},
		{name: "a method override", method: http.MethodPost, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden,
			header: http.Header{"X-Http-Method-Override": {"PUT"}}},
		{name: "a PUT", method: http.MethodPut, target: lfsBatch, body: lfsDownload, status: http.StatusForbidden},
		{name: "a lock", method: http.MethodPost, target: "https://github.test/o/r.git/info/lfs/locks", body: `{"path":"a"}`,
			status: http.StatusForbidden},
		{name: "a lock's verify", method: http.MethodPost, target: "https://github.test/o/r.git/info/lfs/locks/verify", body: lfsDownload,
			status: http.StatusForbidden},
		{name: "under the batch path", method: http.MethodPost, target: lfsBatch + "/upload", body: lfsDownload, status: http.StatusForbidden},
		{name: "a dot segment", method: http.MethodPost, target: "https://github.test/o/r.git/info/lfs/objects/x/../batch", body: lfsDownload,
			status: http.StatusForbidden},
		{name: "another host's path", method: http.MethodPost, target: "https://media.github.test/o/r.git/info/lfs/objects/batch",
			body: lfsDownload, status: http.StatusForbidden},
	})
	s.Contains(s.logs.String(), `why="an LFS batch POST over the size the proxy reads"`)
	s.Contains(s.logs.String(), `why="an LFS batch POST with a content coding"`)
	s.Contains(s.logs.String(), `why="an LFS batch POST with a URL query"`)
	s.Contains(s.logs.String(), `why="an LFS batch POST that is not application/vnd.git-lfs+json in UTF-8"`)
}

// TestAnLFSBatchWithAMethodOverrideTrailerIsRefused: the proxy reads the batch to its end, which
// fills the trailer, and a method override there is refused as in a header.
func (s *ProxySuite) TestAnLFSBatchWithAMethodOverrideTrailerIsRefused() {
	s.readLFS()
	for _, trailer := range []string{"", "X-HTTP-Method-Override: PUT\r\n"} {
		s.granted.forget()
		conn := s.tunnelTo("github.test", &tls.Config{})
		_, err := io.WriteString(conn, "POST /o/r.git/info/lfs/objects/batch HTTP/1.1\r\nHost: github.test\r\n"+
			"Authorization: "+basic("x-access-token", s.placeholder)+"\r\nContent-Type: application/vnd.git-lfs+json\r\n"+
			"Transfer-Encoding: chunked\r\n\r\n"+fmt.Sprintf("%x\r\n%s\r\n0\r\n", len(lfsDownload), lfsDownload)+trailer+"\r\n")
		s.Require().NoError(err)
		response, err := http.ReadResponse(bufio.NewReader(conn), nil)
		s.Require().NoError(err)
		_, _ = io.Copy(io.Discard, response.Body)
		_ = response.Body.Close()

		if trailer == "" {
			s.Equal(http.StatusOK, response.StatusCode, "no trailer")
			s.Len(s.granted.requests(), 1, "no trailer")
			continue
		}
		s.Equal(http.StatusForbidden, response.StatusCode, trailer)
		s.Empty(s.granted.requests(), trailer)
	}
}

// TestAnLFSReadsDownloadHostGetsGETsWithNoToken: the host a batch answer sends git lfs to for an
// object is reached for GET and HEAD, with the request's own Authorization, never the token,
// placeholder or not: its authorization is in its URL (a presigned URL, probe 2026-11-05). Any
// other method is refused, and it is counted under its own entry.
func (s *ProxySuite) TestAnLFSReadsDownloadHostGetsGETsWithNoToken() {
	s.readLFS()
	for _, authorization := range []string{"", "RemoteAuth signed", "Bearer " + s.placeholder, basic("x-access-token", s.placeholder)} {
		s.granted.forget()
		asked := s.asked
		passed := s.requestCount("media.github.test", http.MethodGet, "granted_no_token", "2xx")

		response, _, err := s.send(s.sandbox(s.credential),
			s.request(http.MethodGet, "https://media.github.test/alambic/media/1/fb/6f/fb6f?X-Amz-Signature=a", authorization, ""))

		s.Require().NoError(err, authorization)
		s.Equal(http.StatusOK, response.StatusCode, authorization)
		s.Require().Len(s.granted.requests(), 1, authorization)
		s.Equal("media.github.test", s.granted.requests()[0].Host, authorization)
		s.Equal(authorization, s.granted.requests()[0].Authorization, authorization)
		s.Equal("X-Amz-Signature=a", s.granted.requests()[0].Query, authorization)
		s.Equal(asked, s.asked, "no token is asked for: "+authorization)
		s.Equal(passed+1, s.requestCount("media.github.test", http.MethodGet, "granted_no_token", "2xx"), authorization)
	}
	for _, method := range []string{http.MethodPut, http.MethodPost, http.MethodDelete} {
		s.granted.forget()

		response, _, err := s.send(s.sandbox(s.credential), s.request(method, "https://media.github.test/alambic/media/1", "", "object"))

		s.Require().NoError(err, method)
		s.Equal(http.StatusForbidden, response.StatusCode, method)
		s.Empty(s.granted.requests(), method)
	}
}

// TestADownloadHostIsReachedOnlyThroughAnLFSRead: a grant whose reads have no LFS read, or list
// the host on a read that is no LFS read, gives no tunnel to it; another host than a listed one
// gets none either.
func (s *ProxySuite) TestADownloadHostIsReachedOnlyThroughAnLFSRead() {
	basicAuth := basic("sbx", s.credential)
	s.readLFS()
	s.Equal(http.StatusOK, s.connectStatus("media.github.test:443", basicAuth).StatusCode, "listed")
	s.Equal(http.StatusForbidden, s.connectStatus("other.github.test:443", basicAuth).StatusCode, "not listed")

	grant := s.grant(s.placeholder, "github.test")
	grant.Reads = []core.SandboxRead{{Host: "github.test", Method: http.MethodPost, Path: "/*/*/info/lfs/objects/batch",
		Downloads: []string{"media.github.test"}}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	s.Equal(http.StatusForbidden, s.connectStatus("media.github.test:443", basicAuth).StatusCode, "on a read that is no LFS read")
}

// TestALFSBatchIsRefusedWithoutAnLFSRead is the control: a grant with git's and GraphQL's reads
// alone, as GitHub's revision 10 gives, refuses git lfs pull's batch POST with read-only's 403
// and sends it nowhere, as on the base, 668d15d5c (probe in the PR).
func (s *ProxySuite) TestALFSBatchIsRefusedWithoutAnLFSRead() {
	grant := s.grant(s.placeholder, "github.test", "api.github.test")
	grant.Reads = []core.SandboxRead{{Host: "github.test", Method: http.MethodPost, Path: "/*/*/git-upload-pack"},
		{Host: "api.github.test", Path: "/graphql", GraphQL: true}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	request := s.request(http.MethodPost, lfsBatch, basic("x-access-token", s.placeholder), lfsDownload)
	request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")

	response, body, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Equal(lfsErrorBody("egress: POST github.test/o/r.git/info/lfs/objects/batch refused: code in the sandbox only reads through a connector; "+
		"write with the connector's tools"), body)
	s.Contains(s.logs.String(), `why="a Git LFS request that no read lists"`)
	s.Empty(s.granted.requests())
	s.Equal(http.StatusForbidden, s.connectStatus("media.github.test:443", basic("sbx", s.credential)).StatusCode)
}

// TestGitLFSPushIsRefusedAtLocksVerifyWithTheHint: git lfs push first POSTs locks/verify
// (locking/api.go SearchVerifiable, v3.8.1), which no read lists. The proxy refuses it, and the
// lock requests beside it, with a 403 whose LFS error holds read-only's answer: git-lfs prints
// that message, where it printed «Authorization error: <url>» for a plain-text 403 (probe of
// v3.3.0 and v3.8.1 in the PR). It stays a 403: a 404 or a 501 there makes git-lfs write
// locksverify=false into the repository's config and go on to the upload. The log says why.
func (s *ProxySuite) TestGitLFSPushIsRefusedAtLocksVerifyWithTheHint() {
	s.readLFS()
	for _, target := range []string{"https://github.test/o/r.git/info/lfs/locks/verify", "https://github.test/o/r.git/info/lfs/locks",
		"https://github.test/o/r.git/info/lfs/locks/1/unlock", "https://github.test/o/r/info/lfs/locks/verify"} {
		s.granted.forget()
		asked := s.asked
		counted := s.requestCount("github.test", "POST", "denied_method", "4xx")
		request := s.request(http.MethodPost, target, basic("x-access-token", s.placeholder), `{"ref":{"name":"refs/heads/main"}}`)
		request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, target)
		s.Equal(http.StatusForbidden, response.StatusCode, target)
		s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"), target)
		s.Equal("nosniff", response.Header.Get("X-Content-Type-Options"), target)
		s.Equal(counted+1, s.requestCount("github.test", "POST", "denied_method", "4xx"), target)
		s.Equal(lfsErrorBody("egress: POST github.test"+request.URL.Path+" refused: code in the sandbox only reads through a connector; "+
			"write with the connector's tools"), body, target)
		s.Empty(s.granted.requests(), target)
		s.Equal(asked, s.asked, target)
	}
	s.Contains(s.logs.String(), `why="a Git LFS request that no read lists"`)
}

// TestARefusalOutsideAGitLFSAPIStaysPlainText is the control: a refused request whose path is no
// Git LFS API's, one with info/lfs but not as a directory included, gets read-only's answer in
// plain text, as on the base, d3a1dbeb1 (probe in the PR), and logs no why; nor does a read that
// passes.
func (s *ProxySuite) TestARefusalOutsideAGitLFSAPIStaysPlainText() {
	s.readLFS()
	for _, target := range []string{"https://github.test/o/r.git/info/lfsobjects/batch", "https://github.test/o/r.git/info/lfs",
		"https://github.test/repos/o/r/issues"} {
		request := s.request(http.MethodPost, target, basic("x-access-token", s.placeholder), lfsDownload)

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, target)
		s.Equal(http.StatusForbidden, response.StatusCode, target)
		s.Equal("text/plain; charset=utf-8", response.Header.Get("Content-Type"), target)
		s.Equal("egress: POST github.test"+request.URL.Path+" refused: code in the sandbox only reads through a connector; "+
			"write with the connector's tools\n", body, target)
	}
	s.NotContains(s.logs.String(), "why=")

	s.sendLFS([]lfsRequest{lfsCase("git lfs pull", lfsDownload, http.StatusOK)})
	s.Contains(s.logs.String(), "carried a connection's token")
	s.NotContains(s.logs.String(), "why=")
}

// TestAGitLFSRequestInATunnelWhoseGrantEndedGetsTheMessage: every refusal in a tunnel to a Git LFS
// API is an LFS error with the refusal's own status, such as an upgrade's 501 and the 403 of a
// grant that ended while its tunnel stayed open.
func (s *ProxySuite) TestAGitLFSRequestInATunnelWhoseGrantEndedGetsTheMessage() {
	s.readLFS()
	client := s.sandbox(s.credential)
	upgrade := s.request(http.MethodGet, "https://github.test/o/r.git/info/lfs/locks", "", "")
	upgrade.Header.Set("Connection", "Upgrade")
	upgrade.Header.Set("Upgrade", "websocket")
	response, body, err := s.send(client, upgrade)
	s.Require().NoError(err)
	s.Equal(http.StatusNotImplemented, response.StatusCode)
	s.Equal(lfsErrorBody("egress: protocol upgrades are not proxied"), body)

	response, _, err = s.send(client, s.request(http.MethodGet, "https://github.test/o/r.git/info/lfs/locks", "", ""))
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{s.grant(s.placeholder, "api.github.test")}})
	request := s.request(http.MethodPost, lfsBatch, basic("x-access-token", s.placeholder), lfsDownload)
	request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")

	response, body, err = s.send(client, request)

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"))
	s.Equal(lfsErrorBody("egress: the sandbox's grant for this host has ended"), body)
	s.Len(s.granted.requests(), 1)
}

// TestAGitLFSRequestWhoseConnectionGivesNoTokenGetsTheMessage: git lfs pull on a connection that
// is gone (401) or cannot give a token just now (502) reads the proxy's message, not git-lfs's
// «Authorization error», which only an LFS error body avoids.
func (s *ProxySuite) TestAGitLFSRequestWhoseConnectionGivesNoTokenGetsTheMessage() {
	s.readLFS()
	client := s.sandbox(s.credential)
	for _, failure := range []struct {
		err    error
		status int
	}{{sandbox.ErrNoCredential, http.StatusUnauthorized}, {errors.New("provider down"), http.StatusBadGateway}} {
		s.fail(failure.err)
		request := s.request(http.MethodPost, lfsBatch, basic("x-access-token", s.placeholder), lfsDownload)
		request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")

		response, body, err := s.send(client, request)

		s.Require().NoError(err)
		s.Equal(failure.status, response.StatusCode)
		s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"))
		s.Equal(lfsErrorBody("egress: the connection gives no token: github"), body)
	}
	s.Empty(s.granted.requests())
}

// TestAGitLFSRequestForAnotherHostInTheTunnelGetsTheMessage: the 421 of a Host the tunnel was not
// opened to is an LFS error too, on an LFS path.
func (s *ProxySuite) TestAGitLFSRequestForAnotherHostInTheTunnelGetsTheMessage() {
	s.readLFS()
	request := s.request(http.MethodPost, lfsBatch, basic("x-access-token", s.placeholder), lfsDownload)
	request.Host = "outside.test"

	response, body, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusMisdirectedRequest, response.StatusCode)
	s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"))
	s.Equal(lfsErrorBody("egress: Host is not the host the tunnel was opened to"), body)
	s.Empty(s.granted.requests())
}

// TestAGitLFSRequestWhenTheGrantsCannotBeReadGetsTheMessage: Redis lost after the CONNECT is a 503
// whose body is an LFS error on an LFS path.
func (s *ProxySuite) TestAGitLFSRequestWhenTheGrantsCannotBeReadGetsTheMessage() {
	s.readLFS()
	lost, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(lost)))
	s.T().Cleanup(s.proxy.Close)
	client := s.sandbox(s.credential)
	batch := func() *http.Request {
		request := s.request(http.MethodPost, lfsBatch, basic("x-access-token", s.placeholder), lfsDownload)
		request.Header.Set("Content-Type", "application/vnd.git-lfs+json; charset=utf-8")
		return request
	}
	warm, _, err := s.send(client, batch())
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, warm.StatusCode)
	lost.Close()

	response, body, err := s.send(client, batch())

	s.Require().NoError(err)
	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
	s.Equal("application/vnd.git-lfs+json", response.Header.Get("Content-Type"))
	s.Equal(lfsErrorBody("egress: the sandbox's grants cannot be read just now"), body)
}

// TestAnUnlistedGitLFSRequestWithAMethodOverrideLogsOneWhy: a refusal for two reasons logs the
// first check's alone, so a log line holds one why key (as on the base) and the override stays
// the reason an operator reads.
func (s *ProxySuite) TestAnUnlistedGitLFSRequestWithAMethodOverrideLogsOneWhy() {
	s.readLFS()
	request := s.request(http.MethodPost, "https://github.test/o/r.git/info/lfs/locks/verify", basic("x-access-token", s.placeholder), `{}`)
	request.Header.Set("X-HTTP-Method-Override", "DELETE")

	response, _, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Equal(1, strings.Count(s.logs.String(), "why="), s.logs.String())
	s.Contains(s.logs.String(), "a method override header")
	s.NotContains(s.logs.String(), "a Git LFS request that no read lists")
}
