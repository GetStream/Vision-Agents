package stt

import (
	"context"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const mediaProbeTimeout = 5 * time.Second

// ProbeDuration reads audio duration from a local recording with ffprobe when the
// provider did not report it. It never changes the bytes or refuses a recording.
func ProbeDuration(ctx context.Context, recording Recording) (int64, bool) {
	if len(recording.Audio) == 0 {
		return 0, false
	}
	probe, err := exec.LookPath("ffprobe")
	if err != nil {
		return 0, false
	}

	suffix := filepath.Ext(filepath.Base(strings.ReplaceAll(recording.Filename, "\\", "/")))
	if len(suffix) > 12 || strings.ContainsAny(suffix, `/\:;\r\n`) {
		suffix = ""
	}
	file, err := os.CreateTemp("", "va-stt-media-*"+suffix)
	if err != nil {
		return 0, false
	}
	path := file.Name()
	defer os.Remove(path)
	if _, err := file.Write(recording.Audio); err != nil {
		_ = file.Close()
		return 0, false
	}
	if err := file.Close(); err != nil {
		return 0, false
	}

	probeCtx, cancel := context.WithTimeout(ctx, mediaProbeTimeout)
	defer cancel()
	command := exec.CommandContext(
		probeCtx,
		probe,
		"-v", "error",
		"-protocol_whitelist", "file,pipe",
		"-select_streams", "a:0",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		path,
	)
	command.Stderr = io.Discard
	output, err := command.Output()
	if err != nil {
		return 0, false
	}
	seconds, err := strconv.ParseFloat(strings.TrimSpace(string(output)), 64)
	if err != nil || math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0 || seconds > float64(math.MaxInt64)/1000 {
		return 0, false
	}
	return int64(math.Round(seconds * 1000)), true
}
