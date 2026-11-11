//go:build integration

package api

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"encoding/xml"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// DocumentsSuite covers a file sent with a question: the agent reads it on that turn and every
// turn after, a conversation carried on or forked by the same person reads it again, and nobody
// else's session ever does.
type DocumentsSuite struct {
	RouterSuite
}

func TestDocumentsSuite(t *testing.T) {
	runSuite(t, new(DocumentsSuite))
}

func (s *DocumentsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.forgetNoted()
}

func (s *DocumentsSuite) TestTheFilesTextGoesAheadOfTheConversationAndTheQuestionStaysTheQuestion() {
	opened := s.opening(s.client)
	phrase := "the returns window is 45 days " + s.utils.uuid()
	question := "How long is the returns window? " + s.utils.uuid()

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: question, Files: &[]FileSource{s.docx("returns.docx", phrase)}}, &answering))

	s.Equal(question, value(answering.Said), "what was asked is the question, not the file")
	request := s.asked(question)
	read := documentsIn(request)
	s.Require().Len(read, 1, "the file goes with the reply")
	s.Contains(read[0].Content, phrase)
	s.Contains(read[0].Content, `"name":"returns.docx"`)
	s.True(read[0].Document, "the file is marked so a reply's screen leaves it out")
	s.Equal(read[0], request.Input[0], "the file goes ahead of the conversation")
}

func (s *DocumentsSuite) TestTheFileIsReadAgainOnEveryLaterTurn() {
	opened := s.opening(s.client)
	phrase := "the night shift starts at 22:00 " + s.utils.uuid()
	s.ask(s.client, opened, "What does this say? "+s.utils.uuid(), s.docx("shifts.docx", phrase))

	later := "And when does it end? " + s.utils.uuid()
	s.ask(s.client, opened, later)

	request := s.asked(later)
	read := documentsIn(request)
	s.Require().Len(read, 1, "a turn sent no file still reads the one sent before")
	s.Contains(read[0].Content, phrase)
}

func (s *DocumentsSuite) TestAFileAloneIsAnsweredAsAQuestionAboutIt() {
	opened := s.opening(s.client)
	phrase := "quarterly safety audit " + s.utils.uuid()

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Files: &[]FileSource{s.docx("audit.docx", phrase)}}, &answering))

	request := s.asked("Say briefly what it is about.")
	s.Contains(documentsIn(request)[0].Content, phrase)
}

func (s *DocumentsSuite) TestAFileIsShownOnThePersonsMessage() {
	opened := s.opening(s.client)
	command := s.utils.uuid()

	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: "Summarise this " + s.utils.uuid(),
			Files: &[]FileSource{s.docx("handbook.docx", "staff handbook "+s.utils.uuid())}}, nil))

	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	var message map[string]any
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(channel) {
			if stored["id"] == command {
				message = stored
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the person's message was never written")
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"title":"handbook.docx"`)
	s.NotContains(string(attachments), `"type":"image"`, "a document is shown as a file, not a picture")
}

func (s *DocumentsSuite) TestARetryOfAQuestionWithALargeFileIsAnsweredOnce() {
	opened := s.opening(s.client)
	question := "Summarise the manual " + s.utils.uuid()
	request := CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: question,
		Files: &[]FileSource{s.docx("manual.docx", strings.Repeat("all work and no play ", 5000))}}
	first, payload := s.sent(s.client, opened, request)
	s.Require().Equal(http.StatusAccepted, first, string(payload))
	s.asked(question)

	again, payload := s.client.call(http.MethodPost, s.responses(opened), request)

	s.Equal(http.StatusAccepted, again, "the file its first sending pinned does not count against its retry: %s", payload)
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	turns := 0
	for _, model := range s.noted {
		for _, sent := range model.replies() {
			if said := harness.WithoutNotes(sent.Input); len(said) > 0 && strings.HasSuffix(said[len(said)-1].Content, question) {
				turns++
			}
		}
	}
	s.Equal(1, turns, "the retry starts no second turn")
}

func (s *DocumentsSuite) TestAnotherUserCannotSendAFileToSomeoneElsesSession() {
	opened := s.opening(s.client)
	stranger := s.data.createUser()

	status, _ := stranger.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: "What is in here?", Files: &[]FileSource{s.docx("notes.docx", "private notes")}})

	s.Equal(http.StatusNotFound, status, "another person's session is not there to them")
}

func (s *DocumentsSuite) TestAForkByThePersonCarriesTheirFile() {
	opened := s.opening(s.client)
	phrase := "the lease ends in March " + s.utils.uuid()
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("lease.docx", phrase))

	var forked Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{}, &forked))
	question := "When does the lease end? " + s.utils.uuid()
	s.ask(s.client, forked, question)

	request := s.asked(question)
	read := documentsIn(request)
	s.Require().Len(read, 1, "the person's fork reads the file they sent")
	s.Contains(read[0].Content, phrase)
}

func (s *DocumentsSuite) TestAForkOfAForkStillCarriesTheFile() {
	opened := s.opening(s.client)
	phrase := "the deposit is two months " + s.utils.uuid()
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("lease.docx", phrase))

	var forked, again Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{}, &forked))
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+forked.Id+"/fork",
		ForkSessionRequest{}, &again))
	question := "What is the deposit? " + s.utils.uuid()
	s.ask(s.client, again, question)

	request := s.asked(question)
	s.Require().Len(documentsIn(request), 1, "the first fork kept the file, so its own fork carries it")
}

func (s *DocumentsSuite) TestAForkOfAForkWithoutItsMessagesCarriesNoFile() {
	opened := s.opening(s.client)
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("payslip.docx", "net pay 2,400 "+s.utils.uuid()))
	var forked, bare Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{}, &forked))

	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+forked.Id+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)}, &bare))
	question := "What does the file say? " + s.utils.uuid()
	s.ask(s.serverClient, bare, question)

	s.Empty(documentsIn(s.asked(question)), "a fork that carries no messages carries no file either, whoever forks it")
}

func (s *DocumentsSuite) TestAForksFileIsReadOnceAfterTheForkIsReopened() {
	opened := s.opening(s.client)
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("lease.docx", "the lease ends in March "+s.utils.uuid()))
	var forked Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{}, &forked))
	s.ask(s.client, forked, "Anything else? "+s.utils.uuid())
	s.client.stopSession(forked.Id)

	question := "When does it end? " + s.utils.uuid()
	s.ask(s.client, forked, question)

	s.Len(documentsIn(s.asked(question)), 1, "the reopened fork reads its file once, not once per opening")
}

func (s *DocumentsSuite) TestABackendCannotForkAPersonsConversationWithTheirFile() {
	opened := s.opening(s.client)
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("medical.docx", "blood type O negative "+s.utils.uuid()))

	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork", ForkSessionRequest{})

	s.Equal(http.StatusBadRequest, status, "a person's conversation is theirs to fork, file and all")
}

func (s *DocumentsSuite) TestAReopenedConversationReadsItsFileAgain() {
	opened := s.opening(s.client)
	phrase := "the warranty covers two years " + s.utils.uuid()
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("warranty.docx", phrase))
	s.client.stopSession(opened.Id)

	question := "How long is the warranty? " + s.utils.uuid()
	s.ask(s.client, opened, question)

	request := s.asked(question)
	read := documentsIn(request)
	s.Require().Len(read, 1, "the conversation carried on reads the file it was sent")
	s.Contains(read[0].Content, phrase)
}

func (s *DocumentsSuite) TestAScannedPageIsReadFromItsPicture() {
	opened := s.opening(s.client)
	question := "What does the scan say? " + s.utils.uuid()

	s.ask(s.client, opened, question, s.testdata("scanned.pdf"))

	read := documentsIn(s.asked(question))
	s.Require().Len(read, 1)
	s.Contains(read[0].Content, "Drain the oil.", "a page that is only a picture is read from it")
	s.NotContains(read[0].Content, "only a picture, which was not read")
}

func (s *DocumentsSuite) TestAPageWithAChartHasTheChartRead() {
	opened := s.opening(s.client)
	question := "What does the chart show? " + s.utils.uuid()

	s.ask(s.client, opened, question, s.testdata("figures.pdf"))

	read := documentsIn(s.asked(question))
	s.Require().Len(read, 1)
	s.Contains(read[0].Content, "Oil changes rise each week.", "a chart is read as words the model can answer from")
	s.NotContains(read[0].Content, "which were not read")
}

func (s *DocumentsSuite) TestAPictureSentAsAFileIsRefused() {
	opened := s.opening(s.client)
	picture := "data:image/png;base64," + base64.StdEncoding.EncodeToString(drawn.Images[0].Data)

	status, failure := s.refused(opened, CreateResponseRequest{Text: "What is this?", Files: &[]FileSource{{Url: picture}}})

	s.Equal(http.StatusUnsupportedMediaType, status)
	s.Equal("unsupported_media_type", failure.Code)
}

func (s *DocumentsSuite) TestAnEncryptedFileIsRefusedAsUnreadable() {
	opened := s.opening(s.client)
	encrypted, err := os.ReadFile("../knowledge/extract/testdata/encrypted.docx")
	s.Require().NoError(err)

	status, failure := s.refused(opened, CreateResponseRequest{Text: "What is this?", Files: &[]FileSource{{
		Url: "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(encrypted)}}})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeDocumentUnreadable, failure.Code)
	s.Contains(failure.Message, "password")
}

func (s *DocumentsSuite) TestAFileWithAVoiceNoteIsRefused() {
	opened := s.opening(s.client)

	status, failure := s.refused(opened, CreateResponseRequest{Files: &[]FileSource{s.docx("a.docx", "text")},
		Audio: &AudioSource{Url: "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wavOf("hello"))}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "separate responses")
}

func (s *DocumentsSuite) TestAFileDoesNotLiftTheLimitOnTheRestOfTheRequest() {
	opened := s.opening(s.client)

	status, _ := s.refused(opened, CreateResponseRequest{Text: strings.Repeat("a", 2<<20),
		Files: &[]FileSource{s.docx("a.docx", "text")}})

	s.Equal(http.StatusRequestEntityTooLarge, status, "the room made for an inline file is the file's alone")
}

func (s *DocumentsSuite) TestAFileOverTheConversationsRoomIsRefusedBeforeAnyTurn() {
	opened := s.opening(s.client)
	question := "What does this say? " + s.utils.uuid()

	status, failure := s.refused(opened, CreateResponseRequest{Text: question,
		Files: &[]FileSource{s.docx("long.docx", strings.Repeat("all work and no play ", 7000))}})

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Equal(codeDocumentTooLarge, failure.Code)
	s.never(question)
}

func (s *DocumentsSuite) TestTwoFilesThatFitAloneButNotTogetherRefuseTheSecond() {
	opened := s.opening(s.client)
	half := strings.Repeat("all work and no play ", 3500)
	s.ask(s.client, opened, "Read this "+s.utils.uuid(), s.docx("first.docx", half))

	status, failure := s.refused(opened, CreateResponseRequest{Text: "And this", Files: &[]FileSource{s.docx("second.docx", half)}})

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Equal(codeDocumentTooLarge, failure.Code)
	s.Contains(failure.Message, "room for")
}

func (s *DocumentsSuite) TestAFileThatCannotBeScreenedIsRefusedRatherThanRead() {
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPut, "/v1/policies/app", map[string]any{"prompt_injection": true}, nil))
	opened := s.opening(s.client)
	question := "What does this say? " + s.utils.uuid()

	// The suite's classifier answers none of the screen's questions, so no verdict is reached.
	status, _ := s.refused(opened, CreateResponseRequest{Text: question, Files: &[]FileSource{s.docx("a.docx", "some text")}})

	s.Equal(http.StatusServiceUnavailable, status)
	s.never(question)
}

// opening is a text conversation of as's, kept in Chat, under an agent config on a model the
// suite notes: a conversation reopened from its row is on its config's model.
func (s *DocumentsSuite) opening(as *testClient) Session {
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "documents-" + s.utils.uuid(), Llm: pointerTo("noted/noted-model"), Mode: pointerTo(AgentModeText),
	}, &config))
	return as.createSession(CreateSessionRequest{ConfigId: &config.Id})
}

func (s *DocumentsSuite) responses(opened Session) string {
	return "/v1/agents/sessions/" + opened.Id + "/responses"
}

// ask asks a question as as, with the files given, and waits for the model to be asked it.
func (s *DocumentsSuite) ask(as *testClient, opened Session, question string, files ...FileSource) {
	request := CreateResponseRequest{Text: question, RequestId: pointerTo(s.utils.uuid())}
	if len(files) > 0 {
		request.Files = &files
	}
	status, payload := s.sent(as, opened, request)
	s.Require().Equal(http.StatusAccepted, status, string(payload))
	s.asked(question)
}

// refused sends a request the router refuses, and returns how.
func (s *DocumentsSuite) refused(opened Session, request CreateResponseRequest) (int, ErrorDetail) {
	request.RequestId = pointerTo(s.utils.uuid())
	status, payload := s.sent(s.client, opened, request)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

// asked is the request the model was sent to answer question, once it was.
func (s *DocumentsSuite) asked(question string) llm.ResponseParams {
	s.lastAsked(question)
	request, _ := s.askedWith(question)
	return request
}

// sent sends request as as once the session takes it: a session answers one question at a
// time, so the next waits for the one before it, as an SDK does.
func (s *DocumentsSuite) sent(as *testClient, opened Session, request CreateResponseRequest) (int, []byte) {
	var status int
	var payload []byte
	s.Require().Eventually(func() bool {
		status, payload = as.call(http.MethodPost, s.responses(opened), request)
		return !strings.Contains(string(payload), "a response is already running")
	}, settleFor, 50*time.Millisecond, "the session never took the question")
	return status, payload
}

// never checks that no model was asked question.
func (s *DocumentsSuite) never(question string) {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	for _, model := range s.noted {
		for _, request := range model.requests() {
			for _, message := range request.Input {
				s.NotContains(message.Content, question, "a refused file starts no turn")
			}
		}
	}
}

// docx is a Word document holding text, sent as a data URI.
func (s *DocumentsSuite) docx(name, text string) FileSource {
	var body bytes.Buffer
	s.Require().NoError(xml.EscapeText(&body, []byte(text)))
	var archive bytes.Buffer
	writer := zip.NewWriter(&archive)
	part, err := writer.Create("word/document.xml")
	s.Require().NoError(err)
	_, err = part.Write([]byte(`<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>` +
		body.String() + `</w:t></w:r></w:p></w:body></w:document>`))
	s.Require().NoError(err)
	s.Require().NoError(writer.Close())
	return FileSource{Url: "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(archive.Bytes()), Name: &name}
}

// testdata is a file of the extractor's test data, sent as a data URI under its own name.
func (s *DocumentsSuite) testdata(name string) FileSource {
	data, err := os.ReadFile("../knowledge/extract/testdata/" + name)
	s.Require().NoError(err)
	return FileSource{Url: "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(data), Name: &name}
}

// documentsIn are the files a request was sent.
func documentsIn(request llm.ResponseParams) []llm.Message {
	return slices.DeleteFunc(slices.Clone(request.Input), func(message llm.Message) bool { return !message.Document })
}
