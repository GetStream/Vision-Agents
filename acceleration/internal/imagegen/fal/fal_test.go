package fal

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
)

const model = "alibaba/qwen-image-3/text-to-image"

// queue stands in for FAL's queue: a submission, a status that is in progress for as many
// polls as it is told to be, a result and a cancellation.
type queue struct {
	server *httptest.Server

	mu sync.Mutex
	// pending is how many polls answer IN_PROGRESS before the job is COMPLETED.
	pending int
	// submitStatus, when set, is what a submission is refused with.
	submitStatus int
	// job, when set, is what a submission answers with instead of this queue's own
	// addresses.
	job map[string]string
	// result is what the finished job answers with, and resultStatus its status.
	result       []byte
	resultStatus int

	submitted []map[string]any
	keys      []string
	polls     int
	cancelled int
}

func newQueue() *queue {
	q := &queue{resultStatus: http.StatusOK}
	q.server = httptest.NewServer(http.HandlerFunc(q.serve))
	return q
}

func (q *queue) serve(w http.ResponseWriter, r *http.Request) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.keys = append(q.keys, r.Header.Get("Authorization"))

	requests := q.server.URL + "/alibaba/qwen-image-3/requests/r1"
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/"+model:
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		q.submitted = append(q.submitted, body)
		if q.submitStatus != 0 {
			http.Error(w, `{"detail":"no"}`, q.submitStatus)
			return
		}
		job := q.job
		if job == nil {
			job = map[string]string{
				"request_id":   "r1",
				"status_url":   requests + "/status",
				"response_url": requests,
				"cancel_url":   requests + "/cancel",
			}
		}
		_ = json.NewEncoder(w).Encode(job)
	case r.Method == http.MethodGet && r.URL.Path == "/alibaba/qwen-image-3/requests/r1/status":
		q.polls++
		status := "COMPLETED"
		if q.polls <= q.pending {
			status = "IN_PROGRESS"
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"status": status})
	case r.Method == http.MethodGet && r.URL.Path == "/alibaba/qwen-image-3/requests/r1":
		w.WriteHeader(q.resultStatus)
		_, _ = w.Write(q.result)
	case r.Method == http.MethodPut && r.URL.Path == "/alibaba/qwen-image-3/requests/r1/cancel":
		q.cancelled++
		_, _ = io.WriteString(w, `{"status":"CANCELLATION_REQUESTED"}`)
	default:
		http.NotFound(w, r)
	}
}

// finishWith sets what the finished job answers with.
func (q *queue) finishWith(result map[string]any) {
	encoded, _ := json.Marshal(result)
	q.mu.Lock()
	defer q.mu.Unlock()
	q.result = encoded
}

func (q *queue) seen() (submitted []map[string]any, polls, cancelled int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.submitted, q.polls, q.cancelled
}

type FALSuite struct {
	suite.Suite
	ctx      context.Context
	queue    *queue
	provider *Provider
}

func TestFALSuite(t *testing.T) {
	suite.Run(t, new(FALSuite))
}

func (s *FALSuite) SetupSuite() {
	// Three seconds between polls is right for FAL and far too slow for a test.
	pollInterval = 5 * time.Millisecond
}

func (s *FALSuite) SetupTest() {
	s.ctx = context.Background()
	s.queue = newQueue()
	s.T().Cleanup(s.queue.server.Close)

	provider, err := New(Options{APIKey: "test-key", Model: model, BaseURL: s.queue.server.URL})
	s.Require().NoError(err)
	s.provider = provider
}

// picture is an encoded picture of the given size.
func picture(format string, width, height int) []byte {
	var encoded bytes.Buffer
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	if format == imagegen.FormatJPEG {
		_ = jpeg.Encode(&encoded, canvas, nil)
	} else {
		_ = png.Encode(&encoded, canvas)
	}
	return encoded.Bytes()
}

func inline(format string, data []byte) map[string]any {
	return map[string]any{"url": "data:image/" + format + ";base64," + base64.StdEncoding.EncodeToString(data)}
}

func (s *FALSuite) TestAKeyIsRequired() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{Model: model})
	s.ErrorContains(err, apiKeyEnvVar)
}

func (s *FALSuite) TestAModelThatIsNotAFALEndpointIsRefused() {
	_, err := New(Options{APIKey: "test-key", Model: "qwen"})
	s.ErrorContains(err, "not a FAL model")
}

func (s *FALSuite) TestAJobIsSubmittedPolledAndItsPicturesReturned() {
	s.queue.pending = 2
	s.queue.finishWith(map[string]any{
		"images": []any{inline("png", picture("png", 1024, 768)), inline("png", picture("png", 1024, 768))},
		"seed":   7,
	})
	seed := int64(7)

	drawn, err := s.provider.Generate(s.ctx, imagegen.Request{
		Prompt: "A yellow watering can", NegativePrompt: "text",
		Width: 1024, Height: 768, N: 2, Seed: &seed,
	})
	s.Require().NoError(err)

	s.Require().Len(drawn.Images, 2)
	for _, one := range drawn.Images {
		s.Equal("image/png", one.MediaType)
		s.Equal(1024, one.Width)
		s.Equal(768, one.Height)
		s.Require().NotNil(one.Seed)
		s.EqualValues(7, *one.Seed)
	}

	submitted, polls, cancelled := s.queue.seen()
	s.Require().Len(submitted, 1)
	s.Equal("A yellow watering can", submitted[0]["prompt"])
	s.Equal("text", submitted[0]["negative_prompt"])
	s.Equal(map[string]any{"width": 1024.0, "height": 768.0}, submitted[0]["image_size"])
	s.EqualValues(2, submitted[0]["num_images"])
	s.EqualValues(7, submitted[0]["seed"])
	s.Equal("png", submitted[0]["output_format"])
	s.Equal(true, submitted[0]["sync_mode"], "the picture must come back in the answer rather than behind a link")
	s.Equal(true, submitted[0]["enable_safety_checker"])
	s.Equal(3, polls, "two polls in progress and one completed")
	s.Zero(cancelled, "a job that finished is not cancelled")
	for _, key := range s.queue.keys {
		s.Equal("Key test-key", key)
	}
}

func (s *FALSuite) TestAnAspectRatioIsSentAsFALsNameForIt() {
	s.queue.finishWith(map[string]any{"images": []any{inline("png", picture("png", 16, 9))}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field", AspectRatio: "16:9"})
	s.Require().NoError(err)

	submitted, _, _ := s.queue.seen()
	s.Require().Len(submitted, 1)
	s.Equal("landscape_16_9", submitted[0]["image_size"])
}

func (s *FALSuite) TestJPEGIsAskedForAndReturnedAsJPEG() {
	s.queue.finishWith(map[string]any{"images": []any{inline("jpeg", picture("jpeg", 64, 64))}})

	drawn, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field", Format: imagegen.FormatJPEG})
	s.Require().NoError(err)

	s.Require().Len(drawn.Images, 1)
	s.Equal("image/jpeg", drawn.Images[0].MediaType)
	submitted, _, _ := s.queue.seen()
	s.Equal("jpeg", submitted[0]["output_format"])
}

func (s *FALSuite) TestWhatFALCannotDrawIsRefusedBeforeAnythingIsSubmitted() {
	for _, request := range []imagegen.Request{
		{Prompt: "a field", Width: 4096, Height: 4096},
		{Prompt: "a field", Width: 256, Height: 256},
		{Prompt: "a field", AspectRatio: "21:9"},
	} {
		_, err := s.provider.Generate(s.ctx, request)

		s.Equal(imagegen.UnsupportedOption, imagegen.CodeOf(err), "%+v", request)
		s.False(imagegen.Accepted(err), "nothing was submitted, so the request can go elsewhere")
	}
	submitted, _, _ := s.queue.seen()
	s.Empty(submitted)
}

func (s *FALSuite) TestARefusedSubmissionWasNeverAccepted() {
	s.queue.submitStatus = http.StatusServiceUnavailable

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.False(imagegen.Accepted(err))
	s.ErrorContains(err, "503")
}

func (s *FALSuite) TestAQueueAddressAnywhereElseIsNeverCalled() {
	base := s.queue.server.URL
	host := strings.TrimPrefix(base, "http://")
	for _, address := range []string{
		"http://example.com/alibaba/qwen-image-3/requests/r1/status",
		"https://" + host + "/alibaba/qwen-image-3/requests/r1/status",
		"http://user:secret@" + host + "/alibaba/qwen-image-3/requests/r1/status",
		base + "/alibaba/qwen-image-3/requests/r1/status?token=x",
		base + "/alibaba/other-model/requests/r1/status",
		base + "/alibaba/qwen-image-3/../../other/requests/r1/status",
		base + "/alibaba/qwen-image-30/requests/r1/status",
	} {
		s.queue.job = map[string]string{
			"request_id":   "r1",
			"status_url":   address,
			"response_url": base + "/alibaba/qwen-image-3/requests/r1",
			"cancel_url":   base + "/alibaba/qwen-image-3/requests/r1/cancel",
		}

		_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

		s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err), address)
		s.True(imagegen.Accepted(err), "the job was submitted, so it is not asked of anybody else")
	}
	_, polls, _ := s.queue.seen()
	s.Zero(polls)
}

func (s *FALSuite) TestAPictureBehindALinkIsNeverFetched() {
	s.queue.finishWith(map[string]any{"images": []any{map[string]any{"url": s.queue.server.URL + "/files/picture.png"}}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "inline png")
}

func (s *FALSuite) TestAPictureThatDoesNotDecodeIsRefused() {
	whole := picture("png", 64, 64)
	for _, data := range [][]byte{whole[:40], []byte("not a picture at all")} {
		s.queue.finishWith(map[string]any{"images": []any{inline("png", data)}})

		_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

		s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
		s.ErrorContains(err, "does not decode")
	}
}

func (s *FALSuite) TestAPictureLabelledPNGThatIsAJPEGIsRefused() {
	s.queue.finishWith(map[string]any{"images": []any{inline("png", picture("jpeg", 64, 64))}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "image/jpeg")
}

func (s *FALSuite) TestAPictureLargerThanTheCapIsRefused() {
	oversize := append(picture("png", 64, 64), make([]byte, imagegen.MaxBytes)...)
	s.queue.finishWith(map[string]any{"images": []any{inline("png", oversize)}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "larger than")
}

func (s *FALSuite) TestFewerPicturesThanWereAskedForIsAFailure() {
	s.queue.finishWith(map[string]any{"images": []any{inline("png", picture("png", 64, 64))}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field", N: 2})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "asked for 2 pictures and got 1")
}

func (s *FALSuite) TestAPictureTheSafetyCheckerFlaggedIsContentFiltered() {
	s.queue.finishWith(map[string]any{
		"images":            []any{inline("png", picture("png", 64, 64))},
		"has_nsfw_concepts": []bool{true},
	})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ContentFiltered, imagegen.CodeOf(err))
	s.True(imagegen.Accepted(err))
}

func (s *FALSuite) TestAContentPolicyRefusalIsContentFiltered() {
	s.queue.resultStatus = http.StatusUnprocessableEntity
	s.queue.result = []byte(`{"detail":[{"type":"content_policy_violation","msg":"flagged"}]}`)

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ContentFiltered, imagegen.CodeOf(err))
}

func (s *FALSuite) TestAJobThatRanOutOfTimeIsCancelled() {
	s.queue.pending = 1_000_000
	ctx, cancel := context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()

	_, err := s.provider.Generate(ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.Timeout, imagegen.CodeOf(err))
	s.True(imagegen.Accepted(err))
	_, polls, cancelled := s.queue.seen()
	s.Positive(polls)
	s.Equal(1, cancelled, "a job nobody is waiting for is not left drawing")
}

func (s *FALSuite) TestAJobTheCallerStoppedWaitingForIsCancelled() {
	s.queue.pending = 1_000_000
	ctx, cancel := context.WithCancel(s.ctx)
	time.AfterFunc(30*time.Millisecond, cancel)

	_, err := s.provider.Generate(ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.Cancelled, imagegen.CodeOf(err))
	_, _, cancelled := s.queue.seen()
	s.Equal(1, cancelled)
}
