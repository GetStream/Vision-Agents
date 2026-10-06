package agent

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// TestLiveEOTNativeScoreGatesPipeline runs only when EOT_LIVE_URL and
// EOT_LIVE_TOKEN_FILE are set. EOT_LIVE_PCM_PATH may select raw PCM16LE; otherwise the
// test uses a synthetic 16 kHz signal. The direct call verifies the pinned service
// contract; the same EOTClient then scores pipeline audio while semantic models are stubbed.
func (s *AgentSuite) TestLiveEOTNativeScoreGatesPipeline() {
	t := s.T()
	endpoint := strings.TrimSpace(os.Getenv("EOT_LIVE_URL"))
	if endpoint == "" {
		t.Skip("set EOT_LIVE_URL to run the authenticated native EOT smoke test")
	}
	tokenFile := strings.TrimSpace(os.Getenv("EOT_LIVE_TOKEN_FILE"))
	if tokenFile == "" {
		t.Fatal("EOT_LIVE_TOKEN_FILE is required for the authenticated live test")
	}
	info, err := os.Stat(tokenFile)
	if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
		t.Fatal("EOT_LIVE_TOKEN_FILE must name a nonempty regular token file")
	}

	samples := liveEOTSamples(t)
	pcm := pcm16LEBytes(samples)
	client, err := NewEOTClient(endpoint, tokenFile)
	s.Require().NoError(err)
	transport := &liveEOTTransport{base: http.DefaultTransport}
	client.client.Transport = transport

	requestID := newLiveEOTRequestID(t)
	started := time.Now()
	score, err := client.Score(context.Background(), requestID, pcm)
	directLatency := time.Since(started)
	s.Require().NoError(err, "authenticated native EOT score failed")
	s.Require().Equal(len(samples), score.Samples)
	s.Require().True(validProbability(score.Probability))
	direct := liveEOTExchangeAt(t, transport, 0)
	assertLiveEOTExchange(t, direct, len(samples))
	if score.Probability != direct.probability {
		t.Fatal("EOTClient score did not match its validated live response")
	}
	t.Logf("EOT_LIVE direct request=%s score=%.8f samples=%d latency_ms=%.3f", sanitizedLiveRequestID(requestID), score.Probability, score.Samples, float64(directLatency)/float64(time.Millisecond))

	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.DiscardHandler))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })
	s.eot = client
	s.eotMode = EOTModePrimary
	s.join(false)
	s.flow.mu.Lock()
	s.flow.delay = 100 * time.Millisecond
	s.flow.mu.Unlock()
	participant := stt.Participant{ID: "live-caller", UserID: "live-caller", Name: "Caller"}
	s.edge.inbound <- InboundAudio{
		Participant: participant,
		Audio:       audio.PcmData{Samples: samples, SampleRate: eotSampleRate, Channels: 1},
	}
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) == len(pcm)
	}, "pipeline did not retain the supplied live-smoke PCM window")
	s.eventually(func() bool { return len(s.ears.transcribed()) == 1 }, "pipeline did not feed PCM through its cadence harness")

	// Start with the production default. This observes how the actual model score
	// behaves at 0.5; 0.5 is a smoke-test threshold, not a calibration claim.
	s.agent.mu.Lock()
	s.agent.options.EOTThreshold = 0.5
	s.agent.mu.Unlock()
	s.says(participant, "please help me with a booking")

	s.eventually(func() bool { return liveEOTExchangeReady(transport, 1) }, "pipeline did not complete its first live EOT request")
	firstPipeline := liveEOTExchangeAt(t, transport, 1)
	assertLiveEOTExchange(t, firstPipeline, len(samples))
	if math.Abs(firstPipeline.probability-score.Probability) > 1e-6 {
		t.Fatal("first pipeline score did not match the direct native score")
	}
	s.Empty(s.flow.requests(), "primary EOT must not invoke semantic flow before using the native score")
	if score.Probability < 0.5 {
		s.eventually(func() bool {
			s.agent.converse.mu.Lock()
			defer s.agent.converse.mu.Unlock()
			_, waiting := s.agent.converse.waiting[participant.ID]
			return waiting
		}, "the native score below 0.5 did not leave the caller waiting")
		s.Empty(s.voice.spoken(), "a native Wait decision must not release the semantic response")
		if !hasLiveDecision(s.reported(), ActWait) {
			t.Fatal("pipeline did not report its native-score Wait decision at 0.5")
		}
		s.Empty(s.flow.requests(), "a successful native Wait decision must bypass semantic flow")
		if score.Probability == 0 {
			t.Log("EOT_LIVE pipeline reached only the Wait branch at the exact zero-score boundary")
			return
		}

		// For an interior low score, p/2 is a valid threshold below that same
		// native score. The normal cadence retry then exercises Respond.
		s.agent.mu.Lock()
		s.agent.options.EOTThreshold = score.Probability / 2
		s.agent.mu.Unlock()
		s.eventually(func() bool { return liveEOTExchangeReady(transport, 2) }, "cadence retry did not complete another live EOT request")
		secondPipeline := liveEOTExchangeAt(t, transport, 2)
		assertLiveEOTExchange(t, secondPipeline, len(samples))
		if math.Abs(secondPipeline.probability-score.Probability) > 1e-6 {
			t.Fatal("cadence retry did not score the same native PCM window")
		}
		s.eventually(func() bool { return said(s.voice.spoken()) != "" }, "the native-score Respond decision did not release the semantic reply")
		s.Contains(said(s.voice.spoken()), "Hello there.")
		s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "pipeline did not complete exactly one semantic response")
		if !hasLiveDecision(s.reported(), ActAnswer) {
			t.Fatal("pipeline did not report its native-score answer decision")
		}
		s.Empty(s.flow.requests(), "a successful native Respond decision must bypass semantic flow")
		t.Logf("EOT_LIVE pipeline wait_request=%s wait_score=%.8f respond_request=%s respond_score=%.8f samples=%d threshold=%.8f",
			sanitizedLiveRequestID(firstPipeline.requestID), firstPipeline.probability,
			sanitizedLiveRequestID(secondPipeline.requestID), secondPipeline.probability, len(samples), score.Probability/2)
		return
	}

	s.eventually(func() bool { return said(s.voice.spoken()) != "" }, "the native score at or above 0.5 did not release the semantic reply")
	s.Contains(said(s.voice.spoken()), "Hello there.")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "pipeline did not complete exactly one semantic response")
	if !hasLiveDecision(s.reported(), ActAnswer) {
		t.Fatal("pipeline did not report its native-score answer decision at 0.5")
	}
	s.Empty(s.flow.requests(), "a successful native Respond decision must bypass semantic flow")
	if score.Probability == 1 {
		t.Log("EOT_LIVE pipeline reached only the Respond branch at the exact one-score boundary")
		return
	}

	// The first request naturally responded at 0.5. To cover the other reachable
	// branch, submit the same full window again and set a valid threshold just above
	// the measured score; this does not imply that 0.5 is optimally calibrated.
	s.agent.mu.Lock()
	oldRing := s.agent.audioHistory[participant.ID]
	delete(s.agent.audioHistory, participant.ID)
	s.agent.mu.Unlock()
	if oldRing != nil {
		oldRing.clear()
	}
	s.edge.inbound <- InboundAudio{
		Participant: participant,
		Audio:       audio.PcmData{Samples: samples, SampleRate: eotSampleRate, Channels: 1},
	}
	s.eventually(func() bool { return len(s.agent.eotAudioSnapshot(participant.ID)) == len(pcm) }, "the repeated native smoke PCM window was not retained")
	waitThreshold := (1 + score.Probability) / 2
	s.agent.mu.Lock()
	s.agent.options.EOTThreshold = waitThreshold
	s.agent.mu.Unlock()
	s.says(participant, "and one more thing")
	s.eventually(func() bool { return liveEOTExchangeReady(transport, 2) }, "second native pipeline request did not complete")
	secondPipeline := liveEOTExchangeAt(t, transport, 2)
	assertLiveEOTExchange(t, secondPipeline, len(samples))
	if math.Abs(secondPipeline.probability-score.Probability) > 1e-6 {
		t.Fatal("second pipeline did not score the same native PCM window")
	}
	s.eventually(func() bool {
		s.agent.converse.mu.Lock()
		defer s.agent.converse.mu.Unlock()
		_, waiting := s.agent.converse.waiting[participant.ID]
		return waiting
	}, "the native score below the raised threshold did not leave the caller waiting")
	if countOf[Responded](s.reported()) != 1 || !hasLiveDecision(s.reported(), ActWait) {
		t.Fatal("second pipeline turn did not report a native-score Wait decision")
	}
	s.Empty(s.flow.requests(), "a successful native Wait decision must bypass semantic flow")
	t.Logf("EOT_LIVE pipeline respond_request=%s respond_score=%.8f wait_request=%s wait_score=%.8f samples=%d threshold=%.8f",
		sanitizedLiveRequestID(firstPipeline.requestID), firstPipeline.probability,
		sanitizedLiveRequestID(secondPipeline.requestID), secondPipeline.probability, len(samples), waitThreshold)
}

type liveEOTExchange struct {
	requestID      string
	authenticated  bool
	contentType    string
	requestSamples int
	statusCode     int
	decoded        bool
	responseID     string
	model          string
	release        string
	probability    float64
	wait           float64
	sampleRate     int
	samples        int
	windowSamples  int
}

type liveEOTTransport struct {
	base http.RoundTripper
	mu   sync.Mutex
	all  []liveEOTExchange
}

func (tr *liveEOTTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	contentLength := request.ContentLength
	exchange := liveEOTExchange{
		requestID:      request.Header.Get("X-Request-ID"),
		authenticated:  strings.HasPrefix(request.Header.Get("Authorization"), "Bearer ") && len(request.Header.Get("Authorization")) > len("Bearer "),
		contentType:    request.Header.Get("Content-Type"),
		requestSamples: int(contentLength / 2),
	}
	response, err := tr.base.RoundTrip(request)
	if err != nil {
		tr.append(exchange)
		return nil, err
	}
	if response != nil {
		exchange.statusCode = response.StatusCode
	}
	index := tr.append(exchange)
	if response != nil && response.Body != nil {
		response.Body = &liveEOTResponseBody{ReadCloser: response.Body, owner: tr, index: index}
	}
	return response, nil
}

func (tr *liveEOTTransport) append(exchange liveEOTExchange) int {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.all = append(tr.all, exchange)
	return len(tr.all) - 1
}

func (tr *liveEOTTransport) finish(index int, raw []byte) {
	var response eotResponse
	decoded := json.Unmarshal(raw, &response) == nil
	tr.mu.Lock()
	defer tr.mu.Unlock()
	if index < 0 || index >= len(tr.all) {
		return
	}
	exchange := &tr.all[index]
	exchange.decoded = decoded
	if !decoded {
		return
	}
	exchange.responseID = liveStringValue(response.RequestID)
	exchange.model = liveStringValue(response.Model)
	exchange.release = liveStringValue(response.Release)
	exchange.probability = liveFloatValue(response.Probability)
	exchange.wait = liveFloatValue(response.Wait)
	exchange.sampleRate = liveIntValue(response.SampleRate)
	exchange.samples = liveIntValue(response.Samples)
	exchange.windowSamples = liveIntValue(response.WindowSamples)
}

func (tr *liveEOTTransport) snapshot() []liveEOTExchange {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return append([]liveEOTExchange(nil), tr.all...)
}

type liveEOTResponseBody struct {
	io.ReadCloser
	owner *liveEOTTransport
	index int
	once  sync.Once
	raw   []byte
}

func (body *liveEOTResponseBody) Read(destination []byte) (int, error) {
	n, err := body.ReadCloser.Read(destination)
	if n > 0 && len(body.raw) < 16*1024 {
		remaining := 16*1024 - len(body.raw)
		body.raw = append(body.raw, destination[:min(n, remaining)]...)
	}
	if err != nil {
		body.finish()
	}
	return n, err
}

func (body *liveEOTResponseBody) Close() error {
	err := body.ReadCloser.Close()
	body.finish()
	return err
}

func (body *liveEOTResponseBody) finish() {
	body.once.Do(func() { body.owner.finish(body.index, body.raw) })
}

func liveEOTExchangeReady(transport *liveEOTTransport, index int) bool {
	exchanges := transport.snapshot()
	return index >= 0 && index < len(exchanges) && exchanges[index].decoded
}

func liveEOTExchangeAt(t *testing.T, transport *liveEOTTransport, index int) liveEOTExchange {
	t.Helper()
	exchanges := transport.snapshot()
	if index < 0 || index >= len(exchanges) {
		t.Fatalf("live EOT exchange %d was not observed", index)
	}
	return exchanges[index]
}

func assertLiveEOTExchange(t *testing.T, exchange liveEOTExchange, samples int) {
	t.Helper()
	if exchange.requestID == "" || exchange.responseID != exchange.requestID {
		t.Fatal("live EOT response did not echo the request ID")
	}
	if !exchange.authenticated {
		t.Fatal("live EOT request did not carry bearer authentication")
	}
	if exchange.contentType != "audio/pcm;rate=16000;channels=1;format=s16le" || exchange.requestSamples != samples {
		t.Fatal("live EOT request audio format or sample window was incorrect")
	}
	if exchange.statusCode != http.StatusOK || !exchange.decoded {
		t.Fatal("live EOT service did not return a valid response")
	}
	if exchange.model != eotModel || exchange.release != eotRelease {
		t.Fatal("live EOT response did not match the pinned blend model and release")
	}
	if !validProbability(exchange.probability) || !validProbability(exchange.wait) || math.Abs(exchange.wait-(1-exchange.probability)) > 1e-6 {
		t.Fatal("live EOT response probabilities were invalid")
	}
	if exchange.sampleRate != eotSampleRate || exchange.samples != samples || exchange.windowSamples != samples {
		t.Fatal("live EOT response window metadata did not match the submitted PCM")
	}
}

func liveEOTSamples(t *testing.T) []int16 {
	t.Helper()
	path := strings.TrimSpace(os.Getenv("EOT_LIVE_PCM_PATH"))
	if path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal("EOT_LIVE_PCM_PATH could not be read")
		}
		if len(raw)%2 != 0 || len(raw)/2 < eotMinSamples || len(raw)/2 > eotMaxSamples {
			t.Fatal("EOT_LIVE_PCM_PATH must be raw mono PCM16LE with 320 to 256000 samples")
		}
		samples := make([]int16, len(raw)/2)
		for i := range samples {
			samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return samples
	}

	samples := make([]int16, eotMaxSamples)
	for i := range samples {
		seconds := float64(i) / float64(eotSampleRate)
		envelope := 0.55 + 0.45*math.Sin(2*math.Pi*0.37*seconds)
		carrier := 190*math.Sin(2*math.Pi*220*seconds) + 75*math.Sin(2*math.Pi*510*seconds)
		samples[i] = int16(5000 * envelope * carrier / 265)
	}
	return samples
}

func pcm16LEBytes(samples []int16) []byte {
	pcm := make([]byte, len(samples)*2)
	for i, sample := range samples {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
	}
	return pcm
}

func newLiveEOTRequestID(t *testing.T) string {
	t.Helper()
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		t.Fatal("could not create a unique live EOT request ID")
	}
	return "eot-live-" + hex.EncodeToString(nonce[:])
}

func sanitizedLiveRequestID(requestID string) string {
	digest := sha256.Sum256([]byte(requestID))
	return hex.EncodeToString(digest[:6])
}

func hasLiveDecision(events []Event, kind ActionKind) bool {
	for _, event := range events {
		if decision, ok := event.(Decided); ok && decision.Kind == string(kind) {
			return true
		}
	}
	return false
}

func liveStringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func liveFloatValue(value *float64) float64 {
	if value == nil {
		return math.NaN()
	}
	return *value
}

func liveIntValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}
