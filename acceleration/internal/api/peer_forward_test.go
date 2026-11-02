//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// forwardDelivery hands a scheduled delivery for a stored task's running run, in the task's
// session on other, to it the way a peer does, presenting credential, and returns the run's
// prompt, which the agent is to hear, and what other answered.
func (s *ScheduledTasksSuite) forwardDelivery(other *httptest.Server, credential []byte) (string, *node.ForwardResponse, error) {
	return s.forwardChanged(other, credential, nil)
}

// forwardChanged is forwardDelivery with change applied to the payload before it is sent.
func (s *ScheduledTasksSuite) forwardChanged(other *httptest.Server, credential []byte, change func(map[string]any)) (string, *node.ForwardResponse, error) {
	return s.forwardStored(other, credential, nil, change)
}

// forwardStored is forwardChanged with edit applied to the task before it is stored; the
// payload then names the edited task's owner but the session the task was made in. change
// sees the payload with the task's id and its run's in it.
func (s *ScheduledTasksSuite) forwardStored(other *httptest.Server, credential []byte, edit func(*store.ScheduledTask), change func(map[string]any)) (string, *node.ForwardResponse, error) {
	return s.forwardStoredTo(other, other, credential, edit, change)
}

// forwardStoredTo is forwardStored for a session opened on holding, handed to to.
func (s *ScheduledTasksSuite) forwardStoredTo(holding, to *httptest.Server, credential []byte, edit func(*store.ScheduledTask), change func(map[string]any)) (string, *node.ForwardResponse, error) {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := user.on(holding).createSession(CreateSessionRequest{ConfigId: &config})
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID, APIKey: s.app.key,
		CallerKind: "authenticated", SessionID: opened.Id, Title: "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   time.Now().UTC().Add(48 * time.Hour).Truncate(time.Second), Timezone: "UTC",
	}
	if edit != nil {
		edit(&task)
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 25))
	now := time.Now()
	run := store.ScheduledTaskRun{
		TaskID: task.ID, CustomerID: task.CustomerID, ScheduledFor: now, Trigger: store.ScheduledTaskTriggerManual,
		Prompt: "peer-delivery-" + s.utils.uuid(), State: store.ScheduledTaskRunRunning, Attempts: 1, StartedAt: &now,
	}
	s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &run))

	payload := map[string]any{
		"customer_id": s.customerID(),
		"session_id":  opened.Id,
		"user_id":     task.UserID,
		"caller_kind": task.CallerKind,
		"task_id":     task.ID,
		"run_id":      run.ID,
		"command_id":  "st_" + run.ID,
		"text":        "<scheduled_task>" + run.Prompt + "</scheduled_task>",
		"due":         now,
	}
	if change != nil {
		change(payload)
	}
	answer, err := s.handOff(to, credential, payload)
	return run.Prompt, answer, err
}

// handOff posts payload to to's peer path the way a peer does, presenting credential.
func (s *ScheduledTasksSuite) handOff(to *httptest.Server, credential []byte, payload any) (*node.ForwardResponse, error) {
	body, err := json.Marshal(payload)
	s.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, scheduledPeerPath, bytes.NewReader(body))
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return node.NewForwarder(nil, credential).Ask(ctx, to.Listener.Addr().String(), request, body)
}

// suiteCredential is what a node holding the suite's keyring presents to its peers.
func (s *ScheduledTasksSuite) suiteCredential() []byte {
	credential, _ := s.sealer.Derive(node.CredentialPurpose)
	return credential
}

// refused checks that a forwarded delivery was turned away and never reached the session.
func (s *ScheduledTasksSuite) refused(text string, err error) {
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)
	s.Never(func() bool { return s.model().heard(text) }, time.Second, 50*time.Millisecond,
		"the refused delivery reached the session")
}

// refusedAsNoTask checks that an authenticated peer's delivery was answered as no such
// task and never reached the session.
func (s *ScheduledTasksSuite) refusedAsNoTask(text string, answer *node.ForwardResponse, err error) {
	s.Require().NoError(err)
	s.Equal(int32(http.StatusNotFound), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"code":"`+codeScheduledTaskNotFound+`"`)
	s.Never(func() bool { return s.model().heard(text) }, time.Second, 50*time.Millisecond,
		"the refused delivery reached the session")
}

// delivers checks that a forwarded delivery was answered and reached the session.
func (s *ScheduledTasksSuite) delivers(text string, answer *node.ForwardResponse, err error) {
	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Eventually(func() bool { return s.model().heard(text) }, 10*time.Second, 50*time.Millisecond,
		"the delivery never reached the session")
}

func (s *ScheduledTasksSuite) TestAForwardWithoutTheDeploymentsCredentialIsRefused() {
	other, _ := s.otherRouter(nil)

	text, _, err := s.forwardDelivery(other, nil)
	s.refused(text, err)
}

func (s *ScheduledTasksSuite) TestAForwardWithAnotherDeploymentsCredentialIsRefused() {
	other, _ := s.otherRouter(nil)
	another, err := auth.NewSealer("another deployment")
	s.Require().NoError(err)
	credential, _ := another.Derive(node.CredentialPurpose)

	text, _, err := s.forwardDelivery(other, credential)
	s.refused(text, err)
}

func (s *ScheduledTasksSuite) TestAPeerHoldingTheKeyringDelivers() {
	other, _ := s.otherRouter(nil)

	s.delivers(s.forwardDelivery(other, s.suiteCredential()))
}

func (s *ScheduledTasksSuite) TestAPeerNamingAnotherUserThanTheStoredTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["user_id"] = s.utils.uuid()
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingAnotherSessionThanAServerTasksIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardStored(other, s.suiteCredential(), func(task *store.ScheduledTask) {
		task.CallerKind = string(auth.KindServer)
		task.SessionID = s.utils.uuid()
	}, nil))
}

func (s *ScheduledTasksSuite) TestAPeerNamingAServerTasksOwnSessionDelivers() {
	other, _ := s.otherRouter(nil)

	s.delivers(s.forwardStored(other, s.suiteCredential(), func(task *store.ScheduledTask) {
		task.CallerKind = string(auth.KindServer)
	}, nil))
}

func (s *ScheduledTasksSuite) TestAPeerNamingChannelsTheStoredTaskLacksDeliversInTheConversation() {
	other, _ := s.otherRouter(nil)

	s.delivers(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["channels"] = []string{"sms"}
	}))
}

// textedTask has a task stored for delivery by text to writer, on an agent with an SMS line
// whose user writes from writer.
func (s *ScheduledTasksSuite) textedTask(writer string) func(*store.ScheduledTask) {
	return func(task *store.ScheduledTask) {
		number := s.lineNumber()
		s.seedSMSLine(number)
		config := s.agentWithScheduling(func(r *AgentConfigRequest) {
			r.Channels = &AgentChannels{Sms: &ChannelLineRequest{Number: number}}
		})
		s.Require().NoError(s.store.SaveChannelIdentity(context.Background(), &store.ChannelIdentity{
			CustomerID: s.customerID(), ConfigID: config, Kind: "sms", Address: writer, UserID: task.UserID,
		}))
		task.ConfigID = config
		task.Channels = []string{"sms"}
	}
}

// textsRecordedFor waits for the texts recorded as sent for user and answers them.
func (s *ScheduledTasksSuite) textsRecordedFor(user string) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ? AND user_id = ? AND provider = ?", s.customerID(), user, "telnyx").
			Scan(s.T().Context())
		return err == nil && len(rows) > 0
	}, 10*time.Second, 50*time.Millisecond, "no text was recorded for the task's user")
	return rows
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoConfigDeliversAStoredSMSTaskByText() {
	other, _ := s.otherRouter(nil)
	writer := s.writerNumber()

	_, answer, err := s.forwardStored(other, s.suiteCredential(), s.textedTask(writer), nil)
	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"delivered_to":"sms"`)
	s.Eventually(func() bool { return len(s.sink.to(writer)) > 0 }, 10*time.Second, 50*time.Millisecond,
		"the stored task's text never went out")
}

// The peer reads the task's row to check whose it is, so it takes what the row holds from
// there too: a payload naming another key or other categories cannot bill the task's texts
// to them.
func (s *ScheduledTasksSuite) TestAPeerNamingAnotherKeyAndCategoriesTextsUnderTheStoredOnes() {
	other, _ := s.otherRouter(nil)
	writer := s.writerNumber()
	var stored store.ScheduledTask

	_, answer, err := s.forwardStored(other, s.suiteCredential(), func(task *store.ScheduledTask) {
		s.textedTask(writer)(task)
		task.UsageCategories = []string{"project:reminders"}
		stored = *task
	}, func(payload map[string]any) {
		payload["api_key"] = "forged-key"
		payload["usage_categories"] = []string{"forged"}
		payload["conversation_id"] = s.utils.uuid()
		payload["timezone"] = "Pacific/Kiritimati"
	})

	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"delivered_to":"sms"`)
	for _, row := range s.textsRecordedFor(stored.UserID) {
		s.Equal(s.app.key, row.APIKey, "the text was billed to the payload's key")
		s.Equal([]string{"project:reminders"}, row.UsageCategories)
	}
}

// The peer says what the run's row holds, not the words it was handed.
func (s *ScheduledTasksSuite) TestAPeerHandedOtherWordsSaysTheRunsPrompt() {
	other, _ := s.otherRouter(nil)
	forged := "forged-" + s.utils.uuid()

	prompt, answer, err := s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["text"] = "<scheduled_task>" + forged + "</scheduled_task>"
	})

	s.delivers(prompt, answer, err)
	s.False(s.model().heard(forged), "the agent was told the payload's words")
}

// A run is the task's to deliver only while it runs: a finished one was carried out already.
func (s *ScheduledTasksSuite) TestAPeerNamingAFinishedRunOfTheTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		finished := store.ScheduledTaskRun{
			TaskID: payload["task_id"].(string), CustomerID: s.customerID(), ScheduledFor: time.Now(),
			Trigger: store.ScheduledTaskTriggerManual, Prompt: "done", State: store.ScheduledTaskRunDelivered,
		}
		s.Require().NoError(s.store.InsertScheduledTaskRun(context.Background(), &finished))
		payload["run_id"] = finished.ID
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingARunThatIsNotStoredIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["run_id"] = s.utils.uuid()
	}))
}

// A run of another user's task is not this task's occurrence: taking it would ring this
// task's call on that run, which then never rings for its own task.
func (s *ScheduledTasksSuite) TestAPeerNamingAnotherTasksRunIsRefusedAndLeavesThatRunAlone() {
	s.placingCalls()
	other, peer := s.otherRouter(nil)
	peer.phone = s.router.phone
	mine := s.phoneTask(s.data.createUser().on(other), []string{"phone"}, time.Now())
	theirs := s.phoneTask(s.data.createUser().on(other), []string{"phone"}, time.Now())
	theirRun := s.runOf(theirs, time.Now(), "")

	answer, err := s.handOff(other, s.suiteCredential(), s.deliveryOf(mine, theirRun.ID))

	s.Require().NoError(err)
	s.Equal(int32(http.StatusNotFound), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"code":"`+codeScheduledTaskNotFound+`"`)
	s.Empty(s.runsOf(theirs)[0].SessionID, "the other task's run gave its call away")
}

// A call's grace counts from when the run came due, not from the due time the payload names.
func (s *ScheduledTasksSuite) TestAPeerHandedALaterDueMissesACallTheRunIsTooLateFor() {
	s.placingCalls()
	other, peer := s.otherRouter(nil)
	peer.phone = s.router.phone
	due := time.Now().Add(-16 * time.Minute)
	task := s.phoneTask(s.data.createUser().on(other), []string{"phone"}, due)
	run := s.runOf(task, due, "")
	delivery := s.deliveryOf(task, run.ID)
	delivery.Due = time.Now()

	answer, err := s.handOff(other, s.suiteCredential(), delivery)

	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"missed":true`)
	s.Empty(s.runsOf(task)[0].SessionID, "a call was placed past its grace")
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoUserIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "user_id")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingTheServerForAUsersTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["caller_kind"] = string(auth.KindServer)
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoCallerKindIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "caller_kind")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingATaskThatIsNotStoredIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["task_id"] = s.utils.uuid()
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "task_id")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoCustomerIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "customer_id")
	}))
}

// A node with no store has no task to check a peer's word against: it is no such task, and
// nothing is delivered. A node with no store opens no session, so the one the delivery
// names is held by another node.
func (s *ScheduledTasksSuite) TestANodeWithNoStoreRefusesAPeersDelivery() {
	holding, _ := s.otherRouter(nil)
	logged := &loggedText{}
	other, _, _ := s.otherNodeServingFrom(nil, nil, s.sealer, slog.New(slog.NewTextHandler(logged, nil)))

	s.refusedAsNoTask(s.forwardStoredTo(holding, other, s.suiteCredential(), nil, nil))
	s.NotContains(logged.String(), "a request panicked", "the refusal went on to read a store there is none of")
}

// A node that could not read the task says so, so the router that handed it over tries
// again, and nothing is delivered.
func (s *ScheduledTasksSuite) TestANodeThatCouldNotReadTheTaskAnswersAnError() {
	closed, err := store.Open(testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN")))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	logged := &loggedText{}
	other, _, _ := s.otherNodeServingFrom(closed, nil, s.sealer, slog.New(slog.NewTextHandler(logged, nil)))

	text, answer, err := s.forwardDelivery(other, s.suiteCredential())

	s.Require().NoError(err)
	s.Equal(int32(http.StatusInternalServerError), answer.GetStatus(), "body: %s", answer.GetBody())
	var answered struct{ Error struct{ Code string } }
	s.Require().NoError(json.Unmarshal(answer.GetBody(), &answered), "more than the one error was answered: %s", answer.GetBody())
	s.Equal("internal_error", answered.Error.Code)
	s.Contains(logged.String(), "could not read a peer's scheduled task")
	s.Never(func() bool { return s.model().heard(text) }, time.Second, 50*time.Millisecond,
		"a delivery whose task could not be read reached the session")
}

func (s *ScheduledTasksSuite) TestAPeerStillOnTheOlderKeyDeliversToARotatedNode() {
	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: suiteKEK, 2: "the suite's next key"})
	s.Require().NoError(err)
	other, _ := s.otherRouterSealedBy(nil, rotated, slog.New(slog.DiscardHandler))

	s.delivers(s.forwardDelivery(other, s.suiteCredential()))
}

func (s *ScheduledTasksSuite) TestANodeWithNoKeyringRefusesEveryForward() {
	var logged bytes.Buffer
	other, _ := s.otherRouterSealedBy(nil, nil, slog.New(slog.NewTextHandler(&logged, nil)))
	s.Contains(logged.String(), "setting=ROUTER_AUTH_KEK", "the node never said which setting it lacks")

	text, _, err := s.forwardDelivery(other, s.suiteCredential())
	s.refused(text, err)
	text, _, err = s.forwardDelivery(other, nil)
	s.refused(text, err)
}

func (s *ScheduledTasksSuite) TestTheUnansweredPeerPathIsRefusedWithoutAPeer() {
	body, err := json.Marshal(map[string]string{"customer_id": s.customerID(), "session_id": s.utils.uuid()})
	s.Require().NoError(err)

	status, answer := s.deliver(unansweredPeerPath, string(body), "")

	s.Equal(http.StatusNotFound, status, "body: %s", answer)
}
