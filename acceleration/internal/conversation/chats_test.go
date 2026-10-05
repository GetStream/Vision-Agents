package conversation

import (
	"bytes"
	"context"
	"encoding/hex"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// twoApps is Chats over a deployment's app and the apps customers were given, each an
// in-memory Chat of its own. An app marked down cannot be reached.
type twoApps struct {
	mu         sync.Mutex
	deployment *chattest.Server
	apps       map[int64]*chattest.Server
	own        map[string]int64
	down       map[int64]bool
	pk         int64
	// readOnly are the customers whose work in the deployment's app can only be read.
	readOnly map[string]bool
}

func newTwoApps(t *testing.T) *twoApps {
	return &twoApps{
		deployment: chattest.NewServer(t), apps: map[int64]*chattest.Server{},
		own: map[string]int64{}, down: map[int64]bool{}, pk: 1, readOnly: map[string]bool{},
	}
}

func (a *twoApps) give(t *testing.T, customer string, app int64) *chattest.Server {
	a.mu.Lock()
	defer a.mu.Unlock()
	server := chattest.NewServer(t)
	a.apps[app], a.own[customer] = server, app
	return server
}

func (a *twoApps) For(_ context.Context, customer string) (*getstream.Stream, int64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if app, ok := a.own[customer]; ok {
		return a.apps[app].Client, app, nil
	}
	return a.deployment.Client, 0, nil
}

func (a *twoApps) ForApp(_ context.Context, customer string, app int64) (*getstream.Stream, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch {
	case (app == 0 || app == a.pk) && a.readOnly[customer]:
		return nil, streamapp.ErrReadOnly
	case app == 0 || app == a.pk:
		return a.deployment.Client, nil
	case a.down[app]:
		return nil, streamapp.ErrStreamAppDisconnected
	case a.own[customer] == app:
		return a.apps[app].Client, nil
	}
	return nil, streamapp.ErrStreamAppMoved
}

func (a *twoApps) ForAppReading(ctx context.Context, customer string, app int64) (*getstream.Stream, error) {
	if app == 0 || app == a.pk {
		return a.deployment.Client, nil
	}
	return a.ForApp(ctx, customer, app)
}

func (a *twoApps) DeploymentApp() int64 { return a.pk }

// replied opens a conversation in an app, writes one exchange into it and waits for it to
// be stored.
func replied(t *testing.T, service *Service, app int64, customer, text string) *Conversation {
	t.Helper()
	c, _, _, err := service.OpenInApp(t.Context(), app, customer, "agent", "", "employee", "", nil)
	require.NoError(t, err)
	require.NoError(t, c.Begin(text))
	c.Observe(agent.ResponseDelta{Text: "Noted."})
	c.Observe(agent.Responded{})
	saved(t, c)
	return c
}

func TestEachCustomersConversationIsWrittenToItsOwnApp(t *testing.T) {
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	service, err := NewForChats(t.TempDir(), apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	acme := replied(t, service, 77, "acme", "acme's question")
	globex := replied(t, service, 0, "globex", "globex's question")

	acmeChannel := strings.TrimPrefix(acme.CID(), "agent:")
	globexChannel := strings.TrimPrefix(globex.CID(), "agent:")
	require.Contains(t, own.Messages(acmeChannel), "acme's question")
	require.Empty(t, apps.deployment.Messages(acmeChannel), "nothing of acme's is written into the deployment's app")
	require.Contains(t, apps.deployment.Messages(globexChannel), "globex's question")
	require.Equal(t, int64(77), acme.StreamApp())
}

func TestAPerAppRecordIsFiledUnderItsCustomer(t *testing.T) {
	// An older binary reads only the top of the outbox and would deliver what it found into
	// the deployment's app, so a record kept in another app is out of its sight.
	apps := newTwoApps(t)
	apps.give(t, "acme", 77)
	root := t.TempDir()
	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	acme := replied(t, service, 77, "acme", "hello")
	globex := replied(t, service, 0, "globex", "hello")

	require.FileExists(t, filepath.Join(root, appsDir, hex.EncodeToString([]byte("acme")),
		strings.TrimPrefix(acme.CID(), "agent:"), "state.json"))
	require.NoFileExists(t, filepath.Join(root, strings.TrimPrefix(acme.CID(), "agent:"), "state.json"))
	require.FileExists(t, filepath.Join(root, strings.TrimPrefix(globex.CID(), "agent:"), "state.json"),
		"the deployment app's records keep the layout every record had")
}

func TestACustomerIdCannotEscapeTheOutbox(t *testing.T) {
	apps := newTwoApps(t)
	root := t.TempDir()
	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	dir := service.recordDir("../../etc", 77, "support-"+uuid.NewString())

	relative, err := filepath.Rel(root, dir)
	require.NoError(t, err)
	require.False(t, strings.HasPrefix(relative, ".."), dir)
}

// pendingRecord writes a record with one reply waiting to be delivered, as a process that
// stopped before delivering it would leave it.
func pendingRecord(t *testing.T, dir, customer string, app int64, text string) string {
	t.Helper()
	id := "support-" + uuid.NewString()
	started := time.Now().UTC()
	finished := started.Add(time.Second)
	record := disk{
		OutboxVersion: 1, CommandLedger: true, CID: "agent:" + id, Customer: customer, Agent: "agent",
		Owner: "employee", StreamApp: app,
		Pending: []operation{{Create: true, Message: Message{
			ID: uuid.NewString(), Role: "assistant", State: "completed", Text: text, StartedAt: started, FinishedAt: &finished,
		}}},
	}
	require.NoError(t, os.MkdirAll(filepath.Join(dir, id), 0o700))
	require.NoError(t, writeJSON(filepath.Join(dir, id, "state.json"), record))
	return id
}

func perApp(root, customer string) string {
	return filepath.Join(root, appsDir, hex.EncodeToString([]byte(customer)))
}

func TestARecoveredConversationDeliversIntoItsOwnApp(t *testing.T) {
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	root := t.TempDir()
	id := pendingRecord(t, perApp(root, "acme"), "acme", 77, "written before the restart")

	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	require.Eventually(t, func() bool { return len(own.Messages(id)) == 1 }, 8*time.Second, 20*time.Millisecond)
	require.Empty(t, apps.deployment.Messages(id))
}

func TestARecordFromBeforeKeepsWritingToTheDeploymentApp(t *testing.T) {
	// acme has an app of its own now, but this was written before it did.
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	root := t.TempDir()
	id := pendingRecord(t, root, "acme", 0, "written in the shared app")

	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	require.Eventually(t, func() bool { return len(apps.deployment.Messages(id)) == 1 }, 8*time.Second, 20*time.Millisecond)
	require.Empty(t, own.Messages(id))
}

func TestOneAppWithoutCredentialsDoesNotHoldUpAnother(t *testing.T) {
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	apps.give(t, "globex", 99)
	apps.down[99] = true
	root := t.TempDir()
	stuck := pendingRecord(t, perApp(root, "globex"), "globex", 99, "waiting for globex's app")
	delivered := pendingRecord(t, perApp(root, "acme"), "acme", 77, "acme's")

	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	require.Eventually(t, func() bool { return len(own.Messages(delivered)) == 1 }, 8*time.Second, 20*time.Millisecond)
	waiting, err := loadDisk(filepath.Join(perApp(root, "globex"), stuck))
	require.NoError(t, err)
	require.Len(t, waiting.Pending, 1, "a parked conversation keeps its pending writes")
	require.Empty(t, apps.deployment.Messages(stuck), "and they are delivered nowhere else")
}

func TestAnOutboxRecordInAnotherCustomersDirectoryIsLeftAlone(t *testing.T) {
	apps := newTwoApps(t)
	apps.give(t, "acme", 77)
	root := t.TempDir()
	misfiled := pendingRecord(t, perApp(root, "acme"), "globex", 77, "not acme's")

	service, err := NewForChats(root, apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	service.mu.Lock()
	_, loaded := service.all[known{"globex", "agent:" + misfiled}]
	service.mu.Unlock()
	require.False(t, loaded)
}

func TestARotatedClientIsUsedOnTheNextWrite(t *testing.T) {
	// The client is resolved on every write, so a credential replaced between two of them
	// is the one the second is made with.
	apps := newTwoApps(t)
	apps.give(t, "acme", 77)
	service, err := NewForChats(t.TempDir(), apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c := replied(t, service, 77, "acme", "first")

	rotated := chattest.NewServer(t)
	apps.mu.Lock()
	apps.apps[77] = rotated
	apps.mu.Unlock()
	_, err = rotated.Client.Chat().GetOrCreateChannel(t.Context(), "agent", strings.TrimPrefix(c.CID(), "agent:"),
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: ptr("agent")}})
	require.NoError(t, err)
	require.NoError(t, c.Begin("second"))
	c.Observe(agent.ResponseDelta{Text: "Again."})
	c.Observe(agent.Responded{})
	saved(t, c)

	require.Contains(t, rotated.Messages(strings.TrimPrefix(c.CID(), "agent:")), "second")
}

func TestATitleIsWrittenWhereTheConversationLives(t *testing.T) {
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	service, err := NewForChats(t.TempDir(), apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c := replied(t, service, 77, "acme", "hello")

	require.NoError(t, service.Describe(t.Context(), "acme", c.CID(), "Ordering", ""))

	channel, _ := own.Channel(strings.TrimPrefix(c.CID(), "agent:"))
	require.Equal(t, "Ordering", channel["name"])
}

func TestHistoryWithNoOutboxRecordReadsTheRecordedApp(t *testing.T) {
	// The outbox is gone, as on a host that keeps it in memory, but the deployment still
	// remembers which app the conversation's session was in.
	apps := newTwoApps(t)
	own := apps.give(t, "acme", 77)
	first, err := NewForChats(t.TempDir(), apps)
	require.NoError(t, err)
	c := replied(t, first, 77, "acme", "kept in acme's app")
	cid := c.CID()
	c.Release()
	first.Close()
	// acme moves on to another app; the conversation stays where it was written.
	apps.mu.Lock()
	apps.own["acme"] = 88
	apps.apps[88] = chattest.NewServer(t)
	apps.own["acme-before"] = 77
	apps.mu.Unlock()

	second, err := NewForChats(t.TempDir(), &pinnedTo{Chats: apps, app: 77, client: own.Client})
	require.NoError(t, err)
	t.Cleanup(second.Close)
	second.SetPins(func(context.Context, string, string) (int64, bool, error) { return 77, true, nil })

	page, err := second.HistoryForCaller(t.Context(), "acme", "", cid, "", "employee")
	require.NoError(t, err)
	var texts []string
	for _, message := range page.Messages {
		texts = append(texts, message.Text)
	}
	require.Contains(t, texts, "kept in acme's app")
}

// pinnedTo answers one app's client for that app, standing in for a source that still
// reaches an app the customer used to act in.
type pinnedTo struct {
	Chats
	app    int64
	client *getstream.Stream
}

func (p *pinnedTo) ForApp(ctx context.Context, customer string, app int64) (*getstream.Stream, error) {
	if app == p.app {
		return p.client, nil
	}
	return p.Chats.ForApp(ctx, customer, app)
}

func ptr[T any](v T) *T { return &v }

func TestAConversationInTheSharedAppIsOnlyReadOnceItCannotBeWrittenThere(t *testing.T) {
	// acme registered an app of its own and the fallback is off: what it said in the
	// deployment's app is still its to read, and nothing more is added there.
	apps := newTwoApps(t)
	service, err := NewForChats(t.TempDir(), apps)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c := replied(t, service, 0, "acme", "said in the shared app")
	c.Release()
	apps.give(t, "acme", 77)
	apps.mu.Lock()
	apps.readOnly["acme"] = true
	apps.mu.Unlock()

	page, err := service.HistoryForCaller(t.Context(), "acme", "agent", c.CID(), "", "employee")
	require.NoError(t, err)
	var texts []string
	for _, message := range page.Messages {
		texts = append(texts, message.Text)
	}
	require.Contains(t, texts, "said in the shared app")

	_, _, _, err = service.OpenInApp(t.Context(), 77, "acme", "agent", c.CID(), "employee", "", nil)
	require.ErrorIs(t, err, streamapp.ErrReadOnly, "resuming it would write there")
	require.Error(t, service.Describe(t.Context(), "acme", c.CID(), "renamed", ""))
}

func TestRollingBackToDeploymentModeKeepsTheDeploymentAppsConversationsOpen(t *testing.T) {
	// App mode wrote this conversation for the deployment's own customer, pinned to the
	// deployment app by its id. Deployment mode, knowing that id, still delivers it there.
	deployment := chattest.NewServer(t)
	root := t.TempDir()
	id := pendingRecord(t, root, "1", 1, "written by app mode")
	clients := streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: deployment.URL, App: 1,
	}), streamapp.ClientsOptions{})

	service, err := NewForChats(root, StreamApps(clients))
	require.NoError(t, err)
	t.Cleanup(service.Close)

	require.Eventually(t, func() bool { return len(deployment.Messages(id)) == 1 }, 8*time.Second, 20*time.Millisecond)
}

// AppsSuite covers conversations kept in customers' own Stream apps, side by side in one
// service: acme's app is 77 and globex's is 99.
type AppsSuite struct {
	suite.Suite
	apps    *twoApps
	acme    *chattest.Server
	globex  *chattest.Server
	service *Service
	logs    *logged
}

// logged is what the service logged, written from its conversations' goroutines.
type logged struct {
	mu  sync.Mutex
	out bytes.Buffer
}

func (l *logged) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.out.Write(p)
}

// count is how many lines carry text.
func (l *logged) count(text string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Count(l.out.String(), text)
}

func (l *logged) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.out.String()
}

func TestAppsSuite(t *testing.T) { suite.Run(t, new(AppsSuite)) }

func (s *AppsSuite) SetupTest() {
	s.apps = newTwoApps(s.T())
	s.acme = s.apps.give(s.T(), "acme", 77)
	s.globex = s.apps.give(s.T(), "globex", 99)
	service, err := NewForChats(s.T().TempDir(), s.apps)
	s.Require().NoError(err)
	s.logs = &logged{}
	service.logger = slog.New(slog.NewTextHandler(s.logs, nil))
	s.service = service
}

// parkedReply opens a conversation in globex's app, takes the app out of reach and begins a
// reply, waiting until its writes have been refused once.
func (s *AppsSuite) parkedReply() *Conversation {
	c, _, _, err := s.service.OpenInApp(s.T().Context(), 99, "globex", "agent", "", "employee", "", nil)
	s.Require().NoError(err)
	s.apps.mu.Lock()
	s.apps.down[99] = true
	s.apps.mu.Unlock()
	s.Require().NoError(c.Begin("waiting for globex's app"))
	s.Require().Eventually(func() bool { return current(c).Error != "" }, 8*time.Second, 20*time.Millisecond)
	return c
}

func (s *AppsSuite) TearDownTest() {
	s.service.Close()
}

func (s *AppsSuite) TestTwoCustomersEachKeepTheirOwnConversationUnderTheSameChannelId() {
	acme := replied(s.T(), s.service, 77, "acme", "acme's question")
	id := strings.TrimPrefix(acme.CID(), "agent:")
	// globex made a channel under the same id in its own app, which Stream allows: an id is
	// unique only within one app.
	_, err := s.globex.Client.Chat().GetOrCreateChannel(s.T().Context(), "agent", id, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: ptr("agent"),
			Members: []getstream.ChannelMemberRequest{{UserID: "agent"}, {UserID: "employee"}},
			Custom: map[string]any{CustomerField: "globex", "support_agent_id": "agent", "support_owner_id": "employee",
				TriggerField: SessionCommandTrigger}}})
	s.Require().NoError(err)

	globex, _, _, err := s.service.OpenInApp(s.T().Context(), 99, "globex", "agent", acme.CID(), "employee", "", nil)
	s.Require().NoError(err, "acme holding the id first does not shut globex out of its own")
	s.Require().NoError(globex.Begin("globex's question"))
	globex.Observe(agent.ResponseDelta{Text: "Noted."})
	globex.Observe(agent.Responded{})
	saved(s.T(), globex)
	acme.Release()
	reopened, _, _, err := s.service.OpenInApp(s.T().Context(), 77, "acme", "agent", acme.CID(), "employee", "", nil)
	s.Require().NoError(err)

	s.Same(acme, reopened, "acme reopens its own conversation")
	s.Equal(int64(99), globex.StreamApp())
	s.Contains(s.globex.Messages(id), "globex's question")
	s.NotContains(s.acme.Messages(id), "globex's question")
}

func (s *AppsSuite) TestAConversationWhoseAppCannotBeWrittenIsLoggedAsParkedOnce() {
	c := s.parkedReply()
	// Stopping tries its writes once more, still refused.
	s.service.Close()

	s.Equal(1, s.logs.count(`msg="parked a conversation`), s.logs.String())
	s.Contains(s.logs.String(), "level=WARN")
	s.Contains(s.logs.String(), "customer=globex cid="+c.CID()+" stream_app=99")
	s.Contains(s.logs.String(), streamapp.ErrStreamAppDisconnected.Error())
}

func (s *AppsSuite) TestAParkedConversationIsLoggedOnceItsWritesGoThrough() {
	c := s.parkedReply()
	s.apps.mu.Lock()
	s.apps.down[99] = false
	s.apps.mu.Unlock()
	// Stopping delivers what was waiting, without waiting out the parked retry.
	s.service.Close()

	s.Contains(s.globex.Messages(strings.TrimPrefix(c.CID(), "agent:")), "waiting for globex's app")
	s.Equal(1, s.logs.count(`msg="unparked a conversation`), s.logs.String())
	s.Contains(s.logs.String(), "customer=globex cid="+c.CID()+" stream_app=99")
}
