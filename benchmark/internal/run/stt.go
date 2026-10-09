package run

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
)

// sttRate is the rate the router's transcription socket takes by default.
const sttRate = 16_000

// The pace a call arrives at, the room tone that ends the turn, and how long a transcript
// that has gone quiet is waited on before the clip is called settled.
const (
	sttChunk    = 100 * time.Millisecond
	sttQuiet    = 2 * time.Second
	sttSettled  = 1500 * time.Millisecond
	sttPatience = 10 * time.Second
)

// STTConfig is one STT bench: a clip set, transcribed by each target in turn.
type STTConfig struct {
	Root string
	// Manifest is JSONL of id, reference, and audio (a 16-bit PCM WAV, relative to the
	// manifest) or hypothesis. With no targets, hypotheses are scored as given.
	Manifest string
	// Rows are clips given directly, such as ScenarioClips, in place of a manifest.
	Rows           []STTClipRow
	Targets        []string
	Out            string
	NetworkProfile string
	Logger         *slog.Logger
}

// STTClipRow is one line of a clip manifest.
type STTClipRow struct {
	ID         string `json:"id"`
	Audio      string `json:"audio,omitempty"`
	Reference  string `json:"reference"`
	Hypothesis string `json:"hypothesis,omitempty"`
}

// STT transcribes every clip with every target and writes clips.jsonl, summary.json and
// report.md to cfg.Out. A clip that could not be transcribed is kept, with its error, and
// counted as returning nothing.
func STT(ctx context.Context, cfg STTConfig) (report.Summary, error) {
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	var raw []byte
	var rows []STTClipRow
	var err error
	if len(cfg.Rows) > 0 {
		rows = cfg.Rows
		if raw, err = json.Marshal(rows); err != nil {
			return report.Summary{}, err
		}
	} else {
		if raw, err = os.ReadFile(cfg.Manifest); err != nil {
			return report.Summary{}, err
		}
		if rows, err = parseClipRows(raw, filepath.Dir(cfg.Manifest), len(cfg.Targets) > 0); err != nil {
			return report.Summary{}, err
		}
	}
	dataset, err := datasetHash(raw, rows)
	if err != nil {
		return report.Summary{}, err
	}

	started := time.Now().UTC()
	commit, dirty := gitState(cfg.Root)
	sum := report.Summary{
		SchemaVersion:      report.SchemaVersion,
		BenchmarkVersion:   report.BenchmarkVersion,
		MethodologyVersion: report.MethodologyVersion,
		Kind:               report.KindSTT,
		System:             "stt",
		RunID:              "stt-" + started.Format("20060102T150405Z"),
		Started:            started,
		Manifest: report.RunManifest{
			GitCommit:         commit,
			GitDirty:          dirty,
			DatasetHash:       dataset,
			Target:            strings.Join(cfg.Targets, ","),
			Transport:         "router-stt-socket",
			GoVersion:         runtime.Version(),
			NetworkProfile:    cfg.NetworkProfile,
			NormalizerVersion: score.NormalizerVersion,
			Command:           append([]string{"voicebench"}, os.Args[1:]...),
		},
	}

	var clips []report.STTClip
	if len(cfg.Targets) == 0 {
		for _, row := range rows {
			clips = append(clips, scoreGiven(row))
		}
		sum.STT = []report.STTSummary{report.SummarizeSTT("given", clips)}
	} else {
		client, err := stream.NewClient(stream.Backend{})
		if err != nil {
			return report.Summary{}, err
		}
		for _, target := range cfg.Targets {
			var targetClips []report.STTClip
			for _, row := range rows {
				clip := transcribeClip(ctx, client, target, row)
				if clip.Error != "" {
					logger.Warn("stt clip failed", "target", target, "clip", row.ID, "error", clip.Error)
				}
				targetClips = append(targetClips, clip)
			}
			clips = append(clips, targetClips...)
			sum.STT = append(sum.STT, report.SummarizeSTT(target, targetClips))
		}
	}
	return sum, writeBench(cfg.Out, sum, clips, report.STTMarkdown(sum))
}

func parseClipRows(raw []byte, dir string, streamed bool) ([]STTClipRow, error) {
	var rows []STTClipRow
	seen := map[string]bool{}
	for i, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row STTClipRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("stt: line %d: %w", i+1, err)
		}
		if row.ID == "" {
			row.ID = fmt.Sprint(i + 1)
		}
		if seen[row.ID] {
			return nil, fmt.Errorf("stt: line %d: duplicate id %q", i+1, row.ID)
		}
		seen[row.ID] = true
		switch {
		case streamed && row.Audio == "":
			return nil, fmt.Errorf("stt: line %d: a target needs audio to stream", i+1)
		case !streamed && row.Hypothesis == "":
			return nil, fmt.Errorf("stt: line %d: no target given, so the line needs a hypothesis", i+1)
		}
		if row.Audio != "" && !filepath.IsAbs(row.Audio) {
			row.Audio = filepath.Join(dir, row.Audio)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("stt: empty manifest")
	}
	return rows, nil
}

// datasetHash fingerprints the manifest and every clip it streams, so a re-recorded clip
// under the same name is a different dataset.
func datasetHash(manifest []byte, rows []STTClipRow) (string, error) {
	h := sha256.New()
	h.Write(manifest)
	for _, row := range rows {
		if row.Audio == "" {
			continue
		}
		clip, err := os.ReadFile(row.Audio)
		if err != nil {
			return "", fmt.Errorf("stt: clip %s: %w", row.ID, err)
		}
		h.Write(clip)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func scoreGiven(row STTClipRow) report.STTClip {
	return report.STTClip{
		ID:         row.ID,
		Target:     "given",
		Reference:  row.Reference,
		Hypothesis: row.Hypothesis,
		Returned:   strings.TrimSpace(row.Hypothesis) != "",
		Raw:        score.ScoreWER(row.Reference, row.Hypothesis, false),
		Normalized: score.ScoreWER(row.Reference, row.Hypothesis, true),
	}
}

// heard is one transcript and when it arrived, in milliseconds from the first audio sent.
type heard struct {
	AtMs     int
	Text     string
	Final    bool
	Provider string
	Model    string
	Error    string
}

// speech is where the voice is in a clip, in milliseconds from its start.
type speech struct{ StartMs, EndMs int }

// transcribeClip streams one clip at the pace a call delivers it, follows it with the
// silence that ends a turn, and records every transcript and when it arrived.
func transcribeClip(ctx context.Context, client *stream.Client, target string, row STTClipRow) report.STTClip {
	failed := func(err error) report.STTClip {
		clip := scoreClip(row, target, speech{}, nil)
		clip.Error = err.Error()
		return clip
	}
	pcm, err := audio.ReadWAV(row.Audio)
	if err != nil {
		return failed(err)
	}
	pcm = audio.Resample(pcm, sttRate)
	voice := speechIn(pcm.Samples)

	// Interims are counted as the provider sends them by default, not asked for: asking narrows
	// routing to the providers that declare the option, which is not the target as named.
	transcriber, err := client.Router("").STT().Realtime(ctx, &acceleration.SttOptions{Target: &target})
	if err != nil {
		return failed(err)
	}
	defer transcriber.Close()

	started := time.Now()
	arrivals := make(chan heard, 256)
	// scored stops the reader once the clip is scored, so a provider still talking cannot
	// leave it blocked on a channel nobody reads.
	scored := make(chan struct{})
	defer close(scored)
	go func() {
		defer close(arrivals)
		for t := range transcriber.Transcripts() {
			select {
			case arrivals <- heard{AtMs: int(time.Since(started).Milliseconds()), Text: t.Text, Final: t.Final,
				Provider: t.Provider, Model: t.Model, Error: t.Error}:
			case <-scored:
				return
			}
		}
	}()
	sent := make(chan error, 1)
	go func() { sent <- sendPaced(ctx, transcriber, pcm.Samples, started) }()

	var timeline []heard
	sending := true
	settled := time.NewTimer(time.Hour)
	defer settled.Stop()
	deadline := time.After(time.Duration(len(pcm.Samples))*time.Second/sttRate + sttQuiet + sttPatience)
	for {
		select {
		case h, open := <-arrivals:
			if !open {
				return scoreClip(row, target, voice, timeline)
			}
			timeline = append(timeline, h)
			if !sending {
				settled.Reset(sttSettled)
			}
		case err := <-sent:
			if err != nil {
				return failed(err)
			}
			sending = false
			settled.Reset(sttSettled)
		case <-settled.C:
			return scoreClip(row, target, voice, timeline)
		case <-deadline:
			clip := scoreClip(row, target, voice, timeline)
			if !clip.Returned {
				clip.Error = fmt.Sprintf("no settled transcript within %s of the audio", sttQuiet+sttPatience)
			}
			return clip
		case <-ctx.Done():
			return failed(ctx.Err())
		}
	}
}

// sendPaced sends the clip and then room tone, a chunk at a time on the clock a call keeps.
func sendPaced(ctx context.Context, transcriber *stream.Transcriber, samples []int16, started time.Time) error {
	quiet := make([]int16, int(sttQuiet.Milliseconds())*sttRate/1000)
	size := int(sttChunk.Milliseconds()) * sttRate / 1000
	all := append(append([]int16(nil), samples...), quiet...)
	for i, start := 0, 0; start < len(all); i, start = i+1, start+size {
		if wait := time.Until(started.Add(time.Duration(i) * sttChunk)); wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		chunk := all[start:min(start+size, len(all))]
		pcm := make([]byte, 2*len(chunk))
		for j, sample := range chunk {
			binary.LittleEndian.PutUint16(pcm[2*j:], uint16(sample))
		}
		if err := transcriber.Send(pcm); err != nil {
			return fmt.Errorf("sending audio: %w", err)
		}
	}
	return nil
}

// speechIn is where the voice is in a clip. Timing from the file's edges instead would
// credit the provider for room tone it was given for free.
func speechIn(samples []int16) speech {
	spans := audio.DetectSpeech(samples, sttRate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) == 0 {
		return speech{EndMs: len(samples) * 1000 / sttRate}
	}
	return speech{StartMs: spans[0].StartMs, EndMs: spans[len(spans)-1].EndMs}
}

// scoreClip reads a clip's timeline: what was heard first, what settled last, and what
// arrived while the caller was still speaking. The settled transcripts, in order, are the
// hypothesis.
func scoreClip(row STTClipRow, target string, voice speech, timeline []heard) report.STTClip {
	clip := report.STTClip{ID: row.ID, Target: target, Reference: row.Reference, Timed: true}
	var finals []string
	first := true
	for _, h := range timeline {
		if h.Error != "" {
			if clip.Error == "" {
				clip.Error = h.Error
			}
			continue
		}
		if h.Provider != "" {
			clip.Provider, clip.Model = h.Provider, h.Model
		}
		if strings.TrimSpace(h.Text) == "" {
			continue
		}
		if first {
			clip.ToFirstWordsMs = h.AtMs - voice.StartMs
			first = false
		}
		if h.AtMs < voice.EndMs {
			clip.WhileSpeaking++
		}
		if h.Final {
			finals = append(finals, strings.TrimSpace(h.Text))
			clip.ToSettleMs = h.AtMs - voice.EndMs
		}
	}
	clip.Returned = len(finals) > 0
	clip.Hypothesis = strings.Join(finals, " ")
	clip.Raw = score.ScoreWER(row.Reference, clip.Hypothesis, false)
	clip.Normalized = score.ScoreWER(row.Reference, clip.Hypothesis, true)
	return clip
}

// writeBench writes a component bench's per-clip JSONL, summary and markdown report.
func writeBench[Clip any](out string, sum report.Summary, clips []Clip, markdown string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(out, "clips.jsonl"))
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, clip := range clips {
		if err := enc.Encode(clip); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(sum, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "summary.json"), append(raw, '\n'), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "report.md"), []byte(markdown), 0o644)
}

// ScenarioClips is an STT clip set made of the scenarios' caller lines: names, numbers,
// addresses and corrections the agents have to hear, each with its script as the reference.
// The clips are the caller audio a call plays, synthesized and cached on first use, so this is
// synthetic speech. ids narrows it to a scenario set; nil takes every scenario.
func ScenarioClips(root string, ids []string) ([]STTClipRow, error) {
	inSet := map[string]bool{}
	for _, id := range ids {
		inSet[id] = true
	}
	var rows []STTClipRow
	seen := map[string]bool{}
	for _, pack := range scenario.Packs() {
		scenarios, err := scenario.LoadPack(filepath.Join(root, "scenarios", pack))
		if err != nil {
			return nil, err
		}
		for _, sc := range scenarios {
			if ids != nil && !inSet[sc.ID] {
				continue
			}
			for i, text := range sc.SpeechTexts() {
				if seen[text] {
					continue
				}
				seen[text] = true
				if _, err := synth.LoadOrSynth(root, "", text); err != nil {
					return nil, fmt.Errorf("stt: caller audio for %s: %w", sc.ID, err)
				}
				rows = append(rows, STTClipRow{
					ID:        fmt.Sprintf("%s#%d", sc.ID, i+1),
					Audio:     synth.CachePath(root, "", text),
					Reference: text,
				})
			}
		}
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("stt: no caller lines in that scenario set")
	}
	return rows, nil
}
