package run

import (
	"context"
	"crypto/sha256"
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
)

// ttsPatience is how long one line is waited on, from sending the text to its last audio.
const ttsPatience = 30 * time.Second

// TTSConfig is one TTS bench: a text corpus, spoken by each target in turn.
type TTSConfig struct {
	Root string
	// Corpus is JSONL of id and text. Empty speaks every scenario's agent reply lines, the
	// names, times and confirmations an agent actually has to say.
	Corpus         string
	Targets        []string
	Voice          string
	Out            string
	NetworkProfile string
	Logger         *slog.Logger
}

// TTSLine is one line of a text corpus.
type TTSLine struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// TTS speaks every line with every target, then scores the audio for latency, round-trip
// WER through the scoring ASR, and health. It writes clips.jsonl, summary.json and
// report.md to cfg.Out. A line that could not be spoken or transcribed is kept, with its
// error, and counted as failed.
func TTS(ctx context.Context, cfg TTSConfig) (report.Summary, error) {
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	if len(cfg.Targets) == 0 {
		return report.Summary{}, fmt.Errorf("tts: at least one target is required")
	}
	lines, err := loadCorpus(cfg.Root, cfg.Corpus)
	if err != nil {
		return report.Summary{}, err
	}
	canonical, err := json.Marshal(lines)
	if err != nil {
		return report.Summary{}, err
	}
	corpusHash := sha256.Sum256(canonical)

	started := time.Now().UTC()
	commit, dirty := gitState(cfg.Root)
	sum := report.Summary{
		SchemaVersion:      report.SchemaVersion,
		BenchmarkVersion:   report.BenchmarkVersion,
		MethodologyVersion: report.MethodologyVersion,
		Kind:               report.KindTTS,
		System:             "tts",
		RunID:              "tts-" + started.Format("20060102T150405Z"),
		Started:            started,
		Manifest: report.RunManifest{
			GitCommit:         commit,
			GitDirty:          dirty,
			DatasetHash:       hex.EncodeToString(corpusHash[:]),
			Target:            strings.Join(cfg.Targets, ","),
			TargetVoice:       cfg.Voice,
			Transport:         "router-tts-socket",
			GoVersion:         runtime.Version(),
			NetworkProfile:    cfg.NetworkProfile,
			ScoringASR:        score.ScoringASR,
			NormalizerVersion: score.NormalizerVersion,
			Command:           append([]string{"voicebench"}, os.Args[1:]...),
		},
	}

	client, err := stream.NewClient(stream.Backend{})
	if err != nil {
		return report.Summary{}, err
	}
	var clips []report.TTSClip
	for _, target := range cfg.Targets {
		var targetClips []report.TTSClip
		for _, line := range lines {
			clip := speakLine(ctx, client, target, cfg.Voice, line)
			if clip.Error != "" {
				logger.Warn("tts line failed", "target", target, "line", line.ID, "error", clip.Error)
			}
			targetClips = append(targetClips, clip)
		}
		clips = append(clips, targetClips...)
		sum.TTS = append(sum.TTS, report.SummarizeTTS(target, targetClips))
	}
	return sum, writeBench(cfg.Out, sum, clips, report.TTSMarkdown(sum))
}

// loadCorpus reads a JSONL corpus, or gathers the scenarios' agent reply lines when none is
// given. Repeated text is spoken once.
func loadCorpus(root, path string) ([]TTSLine, error) {
	var lines []TTSLine
	seen := map[string]bool{}
	add := func(line TTSLine) {
		line.Text = strings.TrimSpace(line.Text)
		if line.Text == "" || seen[line.Text] {
			return
		}
		seen[line.Text] = true
		lines = append(lines, line)
	}
	if path == "" {
		for _, pack := range scenario.Packs() {
			scenarios, err := scenario.LoadPack(filepath.Join(root, "scenarios", pack))
			if err != nil {
				return nil, err
			}
			for _, sc := range scenarios {
				for i, reply := range sc.AgentReplies {
					add(TTSLine{ID: fmt.Sprintf("%s#%d", sc.ID, i+1), Text: reply})
				}
			}
		}
	} else {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		for i, text := range strings.Split(string(raw), "\n") {
			if strings.TrimSpace(text) == "" {
				continue
			}
			var line TTSLine
			if err := json.Unmarshal([]byte(text), &line); err != nil {
				return nil, fmt.Errorf("tts: line %d: %w", i+1, err)
			}
			if line.ID == "" {
				line.ID = fmt.Sprint(i + 1)
			}
			add(line)
		}
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("tts: the corpus has no lines")
	}
	return lines, nil
}

// spoken is what one utterance came back as, and when.
type spoken struct {
	Samples     []int16
	Rate        int
	TTFBMs      int
	SynthesisMs int
	Error       string
}

// speakLine sends one line to be spoken and gathers its audio, then scores it.
func speakLine(ctx context.Context, client *stream.Client, target, voice string, line TTSLine) report.TTSClip {
	options := &acceleration.TtsOptions{Target: &target}
	if voice != "" {
		options.Voice = &voice
	}
	speaker, err := client.Router("").TTS().Realtime(ctx, options)
	if err != nil {
		return scoreSpoken(line, target, spoken{Error: err.Error()}, nil)
	}
	defer speaker.Close()

	sent := time.Now()
	if err := speaker.Speak(line.Text); err != nil {
		return scoreSpoken(line, target, spoken{Error: err.Error()}, nil)
	}
	out := gather(ctx, speaker.Audio(), sent)
	return scoreSpoken(line, target, out, transcribeSpoken)
}

// gather reads one utterance's audio until it is done, the socket closes, or it is waited
// on too long.
func gather(ctx context.Context, chunks <-chan stream.Audio, sent time.Time) spoken {
	var out spoken
	deadline := time.After(ttsPatience)
	for {
		select {
		case chunk, open := <-chunks:
			if !open {
				if out.Error == "" && out.SynthesisMs == 0 {
					out.Error = "the socket closed before the utterance finished"
				}
				return out
			}
			switch {
			case chunk.Error != "":
				out.Error = chunk.Error
				return out
			case chunk.Done:
				out.SynthesisMs = int(time.Since(sent).Milliseconds())
				return out
			case len(chunk.Samples) > 0:
				if out.Samples == nil {
					out.TTFBMs = int(time.Since(sent).Milliseconds())
					out.Rate = chunk.SampleRate
				}
				pcm := audio.FromPCM16LE(chunk.Samples, chunk.SampleRate).Samples
				if chunk.Channels == 2 {
					mono := make([]int16, len(pcm)/2)
					for i := range mono {
						mono[i] = pcm[2*i]
					}
					pcm = mono
				}
				out.Samples = append(out.Samples, pcm...)
			}
		case <-deadline:
			out.Error = fmt.Sprintf("the utterance did not finish within %s", ttsPatience)
			return out
		case <-ctx.Done():
			out.Error = ctx.Err().Error()
			return out
		}
	}
}

func transcribeSpoken(pcm audio.PCM) (string, error) {
	heard, err := score.TranscribeDeepgram(pcm)
	return heard.Text, err
}

// scoreSpoken grades one utterance: its latency and real-time factor, its health, and the
// round trip of its audio back to text through transcribe. A nil transcribe skips the
// round trip, for an utterance that never came back.
func scoreSpoken(line TTSLine, target string, out spoken, transcribe func(audio.PCM) (string, error)) report.TTSClip {
	clip := report.TTSClip{ID: line.ID, Target: target, Text: line.Text, Error: out.Error}
	pcm := audio.PCM{Rate: out.Rate, Samples: out.Samples}
	clip.Health = audio.MeasureHealth(pcm.Samples, pcm.Rate)
	clip.Grade, clip.Issues = report.GradeHealth(clip.Health)
	if len(out.Samples) == 0 || out.Rate <= 0 {
		return clip
	}
	clip.Returned = true
	clip.TTFBMs = out.TTFBMs
	clip.SynthesisMs = out.SynthesisMs
	clip.AudioMs = len(out.Samples) * 1000 / out.Rate
	if clip.AudioMs > 0 && out.SynthesisMs > 0 {
		clip.RTF = float64(out.SynthesisMs) / float64(clip.AudioMs)
	}
	if transcribe == nil {
		return clip
	}
	heard, err := transcribe(pcm)
	if err != nil {
		if clip.Error == "" {
			clip.Error = "scoring asr: " + err.Error()
		}
		return clip
	}
	clip.Heard = heard
	clip.Raw = score.ScoreWER(line.Text, heard, false)
	clip.Normalized = score.ScoreWER(line.Text, heard, true)
	return clip
}
