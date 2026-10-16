package agent

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// InterruptionClient is an optional acoustic fast path for overlapping speech.
// Errors and low scores leave the existing transcript/controller path in charge.
type InterruptionClient struct {
	endpoint, tokenFile string
	threshold           float64
	http                *http.Client
}

func NewInterruptionClient(endpoint, tokenFile string, threshold float64, timeout time.Duration) (*InterruptionClient, error) {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("interruption: invalid endpoint")
	}
	if !(threshold > 0 && threshold < 1) || timeout <= 0 {
		return nil, errors.New("interruption: invalid threshold or timeout")
	}
	if u.Path == "" || u.Path == "/" {
		u.Path = "/v1/interruption"
	}
	return &InterruptionClient{endpoint: u.String(), tokenFile: tokenFile, threshold: threshold,
		http: &http.Client{Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}, nil
}

func (c *InterruptionClient) score(ctx context.Context, id string, pcm []byte) (bool, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewReader(pcm))
	if err != nil {
		return false, errors.New("interruption: request failed")
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Request-ID", id)
	if c.tokenFile != "" {
		raw, err := os.ReadFile(c.tokenFile)
		if err != nil {
			return false, errors.New("interruption: credentials unavailable")
		}
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(string(raw)))
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false, errors.New("interruption: endpoint unavailable")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return false, errors.New("interruption: endpoint rejected request")
	}
	var result struct {
		Probability *float64 `json:"probability"`
		ID          string   `json:"request_id"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&result); err != nil || result.ID != id || result.Probability == nil {
		return false, errors.New("interruption: invalid response")
	}
	p := *result.Probability
	if math.IsNaN(p) || p < 0 || p > 1 {
		return false, errors.New("interruption: invalid probability")
	}
	return p >= c.threshold, nil
}

type adaptiveInterruptions struct {
	mu       sync.Mutex
	speakers map[string]*adaptiveSpeaker
}

type adaptiveSpeaker struct {
	prefix, window             []int16
	reply                      string
	serial                     uint64
	samples, scored            int
	lastVoice                  time.Time
	active, inflight, finished bool
}

// append observes the audio clock, not transcript arrival times. A quiet gap or
// different reply retires the utterance; a backchannel never retires later audio.
func (s *adaptiveSpeaker) append(reply string, voiced bool, pcm []int16, now time.Time) ([]byte, uint64) {
	if s.reply != reply || (s.active && now.Sub(s.lastVoice) >= 300*time.Millisecond) {
		s.reply, s.active, s.inflight, s.finished = reply, false, false, false
		s.window, s.samples, s.scored = nil, 0, 0
		s.serial++
	}
	if voiced {
		s.lastVoice = now
	}
	if reply != "" && voiced && !s.active {
		s.active = true
		s.window = append([]int16(nil), s.prefix...)
	}
	if s.active && !s.finished {
		remaining := 2*eotSampleRate - s.samples
		pcmPart := pcm[:min(len(pcm), remaining)]
		s.window = append(s.window, pcmPart...)
		s.samples += len(pcmPart)
	}
	s.prefix = append(s.prefix, pcm...)
	if len(s.prefix) > eotSampleRate {
		s.prefix = append(s.prefix[:0], s.prefix[len(s.prefix)-eotSampleRate:]...)
	}
	if !s.active || s.finished || s.inflight || s.samples < eotSampleRate/5 || s.samples-s.scored < eotSampleRate/10 {
		return nil, s.serial
	}
	s.inflight, s.scored = true, s.samples
	body := make([]byte, 2*len(s.window))
	for i, v := range s.window {
		binary.LittleEndian.PutUint16(body[2*i:], uint16(v))
	}
	return body, s.serial
}

func (a *Agent) observeInterruption(participant stt.Participant, pcm audio.PcmData, now time.Time) {
	if a.options.Interruption == nil || a.voiced == nil || pcm.SampleRate != eotSampleRate || pcm.Channels != 1 || participant.ID == "" {
		return
	}
	state := a.floor()
	reply := state.Speaking
	if state.Quiet || state.Unheard || strings.HasPrefix(reply, backchannelPrefix) {
		reply = ""
	}
	if state.LastParticipant.ID != "" && state.LastParticipant.ID != participant.ID {
		reply = ""
	}
	voiced := a.voiced.lastVoiced(participant.ID).Equal(now)
	a.adaptive.mu.Lock()
	defer a.adaptive.mu.Unlock()
	if a.adaptive.speakers == nil {
		a.adaptive.speakers = map[string]*adaptiveSpeaker{}
	}
	s := a.adaptive.speakers[participant.ID]
	if s == nil {
		if len(a.adaptive.speakers) >= maxEOTParticipants {
			return
		}
		s = &adaptiveSpeaker{}
		a.adaptive.speakers[participant.ID] = s
	}
	body, serial := s.append(reply, voiced, pcm.Samples, now)
	if body == nil {
		return
	}
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return
	}
	a.running.Add(1)
	a.mu.Unlock()
	go func() {
		defer a.running.Done()
		id := fmt.Sprintf("%s/%s/%d/%d", reply, participant.ID, serial, now.UnixNano())
		stop, err := a.options.Interruption.score(a.ctx, id, body)
		a.adaptive.mu.Lock()
		if s.serial != serial || s.reply != reply {
			a.adaptive.mu.Unlock()
			return
		}
		s.inflight = false
		if err != nil || s.scored >= 2*eotSampleRate {
			s.finished = true
		}
		var stopped interruption
		var ok bool
		if err == nil && stop && a.ctx.Err() == nil {
			s.finished = true
			stopped, ok = a.stopPlayback(participant, reply, now, "adaptive", "audio")
		}
		a.adaptive.mu.Unlock()
		if ok {
			a.finishInterruptedTurn(stopped)
		}
	}()
}
