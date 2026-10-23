package daytona

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// platform stands in for Daytona so the wire contract can be tested without a key.
type platform struct {
	server *httptest.Server

	mu sync.Mutex
	// created is every sandbox this platform was asked for.
	created []string
	// deleted is every sandbox it was asked to release.
	deleted []string
	// ran is the body of every code-run, in order.
	ran []map[string]any
	// creations is the body of every request for a sandbox.
	creations []map[string]any
	// files is what the sandbox's file system holds, by path, and uploads each file written
	// into it, with how many runs came before.
	files   map[string][]byte
	uploads []string
	// refuseUpload answers every file written into the sandbox with a 400.
	refuseUpload bool
	// commands are every shell command run in the sandbox, with how many runs came before,
	// and commandExit and commandOutput the exit code and output each gives.
	commands      []string
	commandExit   int
	commandOutput string
	// rawCommands are the request bodies of those commands.
	rawCommands []string
	// steps are the file uploads ("upload") and shell commands ("command") in the order they came.
	steps []string
	// auth is the last Authorization header seen.
	auth string

	// runStatus and runBody are what a code-run answers with.
	deleteStatus int
	runStatus    int
	runBody      string
	// sequence names the next sandbox created.
	sequence int

	// secrets are the organization's secrets by id, and calls every request about a secret
	// or a sandbox's creation, in order, as "<method> <path>".
	secrets map[string]*heldSecret
	calls   []string
	// refuse answers every request with a 500, and refuseSecret the creation of a secret
	// whose name holds it; createStatus, when set, answers a sandbox's creation with it.
	refuse       bool
	refuseSecret string
	// refuseDelete answers the deletion of a secret whose name holds it with a 500.
	refuseDelete string
	// refuseUpdate answers the update of a secret whose name holds it with a 500.
	refuseUpdate string
	createStatus int
	// test is the test the platform serves, when tests share it.
	test string
	// stall, while open, holds every code-run, and arrived hears of each one held.
	stall, arrived chan struct{}
}

// heldSecret is a secret as Daytona keeps it.
type heldSecret struct {
	Name, Value string
	Hosts       []string
}

func newPlatform() *platform {
	stub := &platform{runStatus: http.StatusOK, runBody: `{"result":"12.63\n","exitCode":0}`, files: map[string][]byte{},
		secrets: map[string]*heldSecret{}}
	stub.server = httptest.NewServer(http.HandlerFunc(stub.serve))
	return stub
}

func (p *platform) serve(w http.ResponseWriter, r *http.Request) {
	if strings.HasSuffix(r.URL.Path, "/process/code-run") {
		p.mu.Lock()
		stall, arrived := p.stall, p.arrived
		p.mu.Unlock()
		if stall != nil {
			arrived <- struct{}{}
			<-stall
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.auth = r.Header.Get("Authorization")

	if strings.HasPrefix(r.URL.Path, "/api/secret") || r.Method == http.MethodPost && r.URL.Path == "/api/sandbox" {
		p.calls = append(p.calls, r.Method+" "+r.URL.Path)
	}
	if p.refuse {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"message":"daytona is down"}`))
		return
	}
	switch {
	case strings.HasPrefix(r.URL.Path, "/api/secret"):
		p.secret(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/api/sandbox" && p.createStatus != 0:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(p.createStatus)
		_, _ = w.Write([]byte(`{"message":"no capacity"}`))
	case strings.HasPrefix(r.URL.Path, "/api/socket.io"):
		w.WriteHeader(404)
		return
	case r.Method == http.MethodPost && r.URL.Path == "/api/sandbox":
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		p.creations = append(p.creations, body)
		p.sequence++
		id := "sandbox-" + string(rune('0'+p.sequence))
		p.created = append(p.created, id)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"organizationId":"org","name":"test","user":"root","env":{},"public":false,"networkBlockAll":false,"target":"us","cpu":2,"gpu":0,"memory":4,"disk":10,"id":"` + id + `","state":"started","toolboxProxyUrl":"` + p.server.URL + `/toolbox","labels":{"daytona.io/code-toolbox-language":"python"}}`))

	case strings.HasSuffix(r.URL.Path, "/files/upload-v2"):
		file, _, err := r.FormFile("file")
		if err != nil || p.refuseUpload {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		data, _ := io.ReadAll(file)
		p.files[r.URL.Query().Get("path")] = data
		p.steps = append(p.steps, "upload")
		p.uploads = append(p.uploads, fmt.Sprintf("%s after %d runs", r.URL.Query().Get("path"), len(p.ran)))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"file","path":"` + r.URL.Query().Get("path") + `","type":"file"}`))

	case strings.HasSuffix(r.URL.Path, "/process/execute"):
		var body struct{ Command string }
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		p.rawCommands = append(p.rawCommands, string(raw))
		p.steps = append(p.steps, "command")
		p.commands = append(p.commands, fmt.Sprintf("%s after %d runs", body.Command, len(p.ran)))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"result": p.commandOutput, "exitCode": p.commandExit})

	case strings.HasSuffix(r.URL.Path, "/files/download"):
		data, ok := p.files[r.URL.Query().Get("path")]
		if !ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"message":"no such file"}`))
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(data)

	case r.Method == http.MethodDelete:
		p.deleted = append(p.deleted, r.URL.Path)
		if p.deleteStatus != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(p.deleteStatus)
			_, _ = w.Write([]byte(`{"message":"retry deletion"}`))
			return
		}
		w.WriteHeader(http.StatusNoContent)

	default:
		raw, _ := io.ReadAll(r.Body)
		body := map[string]any{}
		_ = json.Unmarshal(raw, &body)
		body["path"] = r.URL.Path
		body["authorization"] = r.Header.Get("Authorization")
		p.ran = append(p.ran, body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(p.runStatus)
		_, _ = w.Write([]byte(p.runBody))
	}
}

// secret serves Daytona's secret API: create, update and delete. The caller holds mu.
func (p *platform) secret(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name  string   `json:"name"`
		Value *string  `json:"value"`
		Hosts []string `json:"hosts"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	id := strings.TrimPrefix(r.URL.Path, "/api/secret/")
	held := p.secrets[id]
	switch {
	case r.Method == http.MethodPost && p.refuseSecret != "" && strings.Contains(body.Name, p.refuseSecret):
		w.WriteHeader(http.StatusInternalServerError)
		return
	case r.Method == http.MethodPost:
		p.sequence++
		id = "secret-" + string(rune('0'+p.sequence))
		held = &heldSecret{Name: body.Name, Value: *body.Value, Hosts: body.Hosts}
		p.secrets[id] = held
	case held == nil:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"message":"no such secret"}`))
		return
	case r.Method == http.MethodPatch && p.refuseUpdate != "" && strings.Contains(held.Name, p.refuseUpdate):
		w.WriteHeader(http.StatusInternalServerError)
		return
	case r.Method == http.MethodPatch:
		held.Value = *body.Value
	case r.Method == http.MethodDelete && p.refuseDelete != "" && strings.Contains(held.Name, p.refuseDelete):
		w.WriteHeader(http.StatusInternalServerError)
		return
	case r.Method == http.MethodDelete:
		delete(p.secrets, id)
		w.WriteHeader(http.StatusNoContent)
		return
	}
	answer, _ := json.Marshal(map[string]any{"id": id, "name": held.Name, "placeholder": "dtn_secret_" + id,
		"hosts": held.Hosts, "createdAt": "2026-10-20T00:00:00Z", "updatedAt": "2026-10-20T00:00:00Z"})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(answer)
}

func (p *platform) executed() []map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]map[string]any(nil), p.ran...)
}

func (p *platform) requests() []map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]map[string]any(nil), p.creations...)
}

func (p *platform) holds(path string, data []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.files[path] = data
}

func (p *platform) sandboxes() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.created...)
}

// outlasting starts code in box that the platform holds until release, or the test's end, and
// returns once it is held, with the run's error to come.
func (p *platform) outlasting(t *testing.T, box *Sandbox) (ran <-chan error, release func()) {
	stall, arrived, done := make(chan struct{}), make(chan struct{}, 1), make(chan error, 1)
	var once sync.Once
	release = func() { once.Do(func() { close(stall) }) }
	t.Cleanup(release)
	p.mu.Lock()
	p.stall, p.arrived = stall, arrived
	p.mu.Unlock()
	go func() {
		_, err := box.Run(context.Background(), "print(1)", nil)
		done <- err
	}()
	select {
	case <-arrived:
	case err := <-done:
		t.Fatalf("the run ended before its code ran: %v", err)
	}
	return done, release
}

func (p *platform) released() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.deleted...)
}

type DaytonaSuite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	box      *Sandbox
}

func TestDaytonaSuite(t *testing.T) {
	suite.Run(t, new(DaytonaSuite))
}

func (s *DaytonaSuite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)

	box, err := New(Options{
		APIKey:   "key-1",
		APIURL:   s.platform.server.URL + "/api",
		ProxyURL: s.platform.server.URL + "/toolbox",
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.box = box
	s.T().Cleanup(func() { _ = box.Close() })
}

func (s *DaytonaSuite) TestAKeyIsRequired() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{})

	s.ErrorContains(err, apiKeyEnvVar)
}

func (s *DaytonaSuite) TestCodeRunsInASandboxAndItsOutputComesBack() {
	result, err := s.box.Run(s.ctx, "print(84.20 * 0.15)", nil)

	s.Require().NoError(err)
	s.Equal("12.63\n", result.Output)
	s.Zero(result.ExitCode)

	s.Require().Len(s.platform.executed(), 1)
	ran := s.platform.executed()[0]
	s.Equal("print(84.20 * 0.15)", ran["code"])
	s.Equal("python", ran["language"])
	s.Equal("/toolbox/sandbox-1/process/code-run", ran["path"])
	s.Equal("Bearer key-1", ran["authorization"])
}

func (s *DaytonaSuite) TestOneSandboxServesEveryPieceOfCode() {
	// Creating one is the slow part, and a conversation that delegates twice should not
	// wait for it twice.
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	_, err = s.box.Run(s.ctx, "print(2)", nil)
	s.Require().NoError(err)

	s.Len(s.platform.sandboxes(), 1)
	s.Len(s.platform.executed(), 2)
}

func (s *DaytonaSuite) TestNothingIsCreatedUntilThereIsCodeToRun() {
	s.Empty(s.platform.sandboxes(), "a session that never delegates never pays for one")
}

func (s *DaytonaSuite) TestCodeThatFailedIsAResultRatherThanAnError() {
	// The code ran. That it did not work is something the model can read and act on.
	s.platform.mu.Lock()
	s.platform.runBody = `{"result":"NameError: total","exitCode":1}`
	s.platform.mu.Unlock()

	result, err := s.box.Run(s.ctx, "print(total)", nil)

	s.Require().NoError(err)
	s.Equal(1, result.ExitCode)
	s.Equal("NameError: total", result.Output)
}

func (s *DaytonaSuite) TestARefusedRunIsReportedWithWhatDaytonaSaid() {
	s.platform.mu.Lock()
	s.platform.runStatus = http.StatusBadGateway
	s.platform.runBody = `{"message":"the sandbox is gone"}`
	s.platform.mu.Unlock()

	_, err := s.box.Run(s.ctx, "print(1)", nil)

	s.ErrorContains(err, "the sandbox is gone")
}

func (s *DaytonaSuite) TestClosingReleasesTheSandbox() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Require().NoError(s.box.Close())

	s.Equal([]string{"/api/sandbox/sandbox-1"}, s.platform.released())
}

func (s *DaytonaSuite) TestClosingTwiceIsSafe() {
	s.NoError(s.box.Close())
	s.NoError(s.box.Close())

	s.Empty(s.platform.released(), "there was never a sandbox to release")
}

func (s *DaytonaSuite) TestCodeIsRefusedOnceTheSandboxIsClosed() {
	s.Require().NoError(s.box.Close())

	_, err := s.box.Run(s.ctx, "print(1)", nil)

	s.ErrorContains(err, "closed")
}

func (s *DaytonaSuite) TestThereHasToBeSomethingToRun() {
	_, err := s.box.Run(s.ctx, "   ", nil)

	s.ErrorContains(err, "no code")
	s.Empty(s.platform.sandboxes())
}

func (s *DaytonaSuite) TestConcurrentRunsCreateOnlyOneSandbox() {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); _, err := s.box.Run(s.ctx, "print(1)", nil); s.NoError(err) }()
	}
	wg.Wait()
	s.Len(s.platform.sandboxes(), 1)
	s.Len(s.platform.executed(), 8)
	s.NoError(s.box.Close())
}

// TestACloseARunOutlastsReturnsAtItsTimeout is the control: the error is the one of base
// c573d2ba6 (probe: this test there).
func (s *DaytonaSuite) TestACloseARunOutlastsReturnsAtItsTimeout() {
	box := s.configured(Options{Timeout: time.Second})
	ran, release := s.platform.outlasting(s.T(), box)

	s.ErrorIs(box.Close(), context.DeadlineExceeded)

	release()
	s.NoError(<-ran)
}

func (s *DaytonaSuite) TestFailedDeletionRetainsIdentity() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	s.platform.mu.Lock()
	s.platform.deleteStatus = 503
	s.platform.mu.Unlock()
	s.Error(s.box.Close())
	s.Require().NotNil(s.box.box)
	_, err = s.box.Run(s.ctx, "print(2)", nil)
	s.ErrorContains(err, "closed")
	s.platform.mu.Lock()
	s.platform.deleteStatus = 0
	s.platform.mu.Unlock()
	s.NoError(s.box.Close())
	s.Nil(s.box.box)
	s.Len(s.platform.sandboxes(), 1)
}

func (s *DaytonaSuite) configured(config Options) *Sandbox {
	config.APIKey = "key-1"
	config.APIURL = s.platform.server.URL + "/api"
	config.Logger = slog.New(slog.DiscardHandler)
	box, err := New(config)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

func (s *DaytonaSuite) TestWithoutSetupDaytonasOwnSandboxIsUsed() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.Require().Len(s.platform.requests(), 1)
	s.NotContains(s.platform.requests()[0], "buildInfo", "nothing has to be built")
}

func (s *DaytonaSuite) TestASandboxReachesNothingByDefault() {
	_, err := s.box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	created := s.platform.requests()[0]
	s.Equal(true, created["networkBlockAll"])
	s.NotContains(created, "domainAllowList")
	s.NotContains(created, "networkAllowList")
}

func (s *DaytonaSuite) TestANetworkedSandboxReachesTheInternet() {
	box := s.configured(Options{Config: sandbox.Config{Network: true}})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal(false, s.platform.requests()[0]["networkBlockAll"])
}

func (s *DaytonaSuite) TestASandboxReachesOnlyWhatItsAllowListsName() {
	box := s.configured(Options{Config: sandbox.Config{AllowedDomains: []string{"api.example.com", "pypi.org"}}})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal(false, created["networkBlockAll"])
	s.Equal("api.example.com,pypi.org", created["domainAllowList"])
	s.NotContains(created, "networkAllowList", "Daytona refuses a sandbox given two allow lists")
}

func (s *DaytonaSuite) TestASandboxReachesOnlyTheRangesItsAllowListNames() {
	box := s.configured(Options{Config: sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8", "192.168.1.0/24"}}})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	created := s.platform.requests()[0]
	s.Equal("10.0.0.0/8,192.168.1.0/24", created["networkAllowList"])
	s.NotContains(created, "domainAllowList")
}

func (s *DaytonaSuite) TestASetupIsBuiltOnTheImageItNames() {
	box := s.configured(Options{Config: sandbox.Config{
		Image: "python:3.13-slim-bookworm",
		Setup: []string{"apt-get update && apt-get install -y libgl1", "pip install bpy==5.2.2"},
		CPU:   2, MemoryGB: 4,
	}})

	_, err := box.Run(s.ctx, "import bpy", nil)

	s.Require().NoError(err)
	s.Require().Len(s.platform.requests(), 1)
	created := s.platform.requests()[0]
	build, _ := created["buildInfo"].(map[string]any)
	s.Equal("FROM python:3.13-slim-bookworm\n"+
		"RUN apt-get update && apt-get install -y libgl1\n"+
		"RUN pip install bpy==5.2.2", build["dockerfileContent"])
	s.EqualValues(2, created["cpu"])
	s.EqualValues(4, created["memory"])
}

func (s *DaytonaSuite) TestASetupWithoutAnImageStartsFromSlimPython() {
	box := s.configured(Options{Config: sandbox.Config{Setup: []string{"pip install numpy"}}})

	_, err := box.Run(s.ctx, "import numpy", nil)

	s.Require().NoError(err)
	build, _ := s.platform.requests()[0]["buildInfo"].(map[string]any)
	s.Equal("FROM python:3.13-slim-bookworm\nRUN pip install numpy", build["dockerfileContent"])
}

func (s *DaytonaSuite) TestARunMayTakeAsLongAsTheConfigAllows() {
	box := s.configured(Options{Config: sandbox.Config{TimeoutMs: 300_000}})

	_, err := box.Run(s.ctx, "render()", nil)

	s.Require().NoError(err)
	s.EqualValues(300, s.platform.executed()[0]["timeout"], "Daytona is asked for five minutes, in seconds")
}

func (s *DaytonaSuite) TestNoRunMayTakeLongerThanTheCeiling() {
	box := s.configured(Options{Config: sandbox.Config{TimeoutMs: 24 * 3600 * 1000}})

	_, err := box.Run(s.ctx, "render()", nil)

	s.Require().NoError(err)
	s.EqualValues(sandbox.MaxTimeout.Seconds(), s.platform.executed()[0]["timeout"])
}

func (s *DaytonaSuite) TestFilesTheCodeWroteComeBackWithItsOutput() {
	png := []byte("\x89PNG\r\n\x1a\nnot really a picture")
	s.platform.holds("/tmp/render.png", png)

	result, err := s.box.Run(s.ctx, "render()", []string{"/tmp/render.png"})

	s.Require().NoError(err)
	s.Equal([]sandbox.File{{Name: "render.png", MIME: "image/png", Data: png}}, result.Files)
	s.Empty(result.Missing)
	s.Equal("12.63\n", result.Output, "what it printed still comes back")
}

func (s *DaytonaSuite) TestAFileTheCodeDidNotWriteIsReportedMissing() {
	s.platform.holds("/tmp/a.txt", []byte("a"))

	result, err := s.box.Run(s.ctx, "render()", []string{"/tmp/a.txt", "/tmp/render.png"})

	s.Require().NoError(err)
	s.Require().Len(result.Files, 1)
	s.Equal("a.txt", result.Files[0].Name)
	s.Equal("text/plain", result.Files[0].MIME)
	s.Equal([]string{"/tmp/render.png"}, result.Missing)
}

func (s *DaytonaSuite) TestOnlySoManyFilesComeBackFromOneRun() {
	var wanted []string
	for i := range sandbox.MaxFiles + 2 {
		name := "/tmp/" + string(rune('a'+i)) + ".txt"
		s.platform.holds(name, []byte("x"))
		wanted = append(wanted, name)
	}

	result, err := s.box.Run(s.ctx, "write()", wanted)

	s.Require().NoError(err)
	s.Len(result.Files, sandbox.MaxFiles)
	s.Equal(wanted[sandbox.MaxFiles:], result.Missing)
}
