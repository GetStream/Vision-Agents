// Package video turns a recorded clip into frames a vision model can be shown.
//
// None of the models routed here take a video whole, OpenAI's included: what they take is
// a sequence of images, and what makes it a video to them is being told when each was
// taken. So a clip is sampled into evenly spaced JPEGs, each carrying its timestamp, and
// goes down the same path an attached image does.
//
// The router fetches a clip by URL itself, which makes it a way into whatever network it
// runs on. The fetch therefore refuses any address that is not public, checked on the
// address actually dialled so that neither a redirect nor a DNS answer changed after the
// check can get around it.
package video

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/exec"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// MaxBytes bounds one clip, fetched or sent inline.
	MaxBytes = 50 << 20
	// DefaultFrames is how many frames a clip is sampled into when the caller names none.
	DefaultFrames = 8
	// MaxFrames is the most one clip may be sampled into.
	MaxFrames = 32

	fetchTimeout = 60 * time.Second
	// frameWidth is as wide as a frame is kept. Vision models downscale anything larger,
	// so the extra pixels are bytes on the wire and nothing else.
	frameWidth = 1024
)

// ErrUnavailable is what a deployment without ffmpeg says, rather than failing halfway.
var ErrUnavailable = errors.New("video: this deployment cannot read video (ffmpeg is not installed)")

// readOnlyAClip are the input options of every ffprobe and ffmpeg run. ffmpeg picks the
// format from the bytes, and some formats (an HLS playlist, a concat list) name other
// files that it then opens, so a caller could have it read any file the router can
// reach. The formats are two demuxers, each by all the names `ffmpeg -demuxers` lists
// for it: mov for MP4, MOV and 3GP, matroska for MKV and WebM
// (https://ffmpeg.org/ffmpeg-formats.html#Format-Options). The one protocol is the local
// file the clip was written to (https://ffmpeg.org/ffmpeg-protocols.html).
var readOnlyAClip = []string{"-format_whitelist", "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm", "-protocol_whitelist", "file"}

// errPrivate is a fetch that would have reached somewhere that is not the public internet.
var errPrivate = errors.New("video: the URL resolves to an address that is not public")

// Frame is one moment of a clip.
type Frame struct {
	At    time.Duration
	Image llm.ImagePart
}

// Frames samples a clip into count evenly spaced frames, and says how long the clip is.
// A clip with fewer frames than count gives the frames it has, each once.
//
// source is an HTTP(S) URL or a base64 data URI of a video.
func Frames(ctx context.Context, source string, count int) ([]Frame, time.Duration, error) {
	return frames(ctx, publicClient, source, count)
}

func frames(ctx context.Context, client *http.Client, source string, count int) ([]Frame, time.Duration, error) {
	if count == 0 {
		count = DefaultFrames
	}
	if count < 1 || count > MaxFrames {
		return nil, 0, stack.Wrap(fmt.Errorf("video: max_frames must be between 1 and %d", MaxFrames))
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, 0, stack.Wrap(ErrUnavailable)
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		return nil, 0, stack.Wrap(ErrUnavailable)
	}

	clip, err := load(ctx, client, source)
	if err != nil {
		return nil, 0, stack.Wrap(err)
	}
	file, err := os.CreateTemp("", "clip-*")
	if err != nil {
		return nil, 0, stack.Wrap(err)
	}
	defer os.Remove(file.Name())
	_, err = file.Write(clip)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return nil, 0, stack.Wrap(err)
	}

	length, err := duration(ctx, file.Name())
	if err != nil {
		return nil, 0, stack.Wrap(err)
	}
	times, err := frameTimes(ctx, file.Name())
	if err != nil {
		return nil, 0, stack.Wrap(err)
	}
	picked := pick(times, length, count)
	sampled := make([]Frame, 0, len(picked))
	for _, slot := range picked {
		jpeg, err := frameAt(ctx, file.Name(), slot.seek)
		if err != nil {
			return nil, 0, stack.Wrap(err)
		}
		sampled = append(sampled, Frame{At: slot.at, Image: llm.ImagePart{MIME: "image/jpeg", Data: jpeg, Video: true}})
	}
	return sampled, length, nil
}

// slot is one sample of a clip: the moment it stands for, and where to seek to get the
// frame showing then.
type slot struct{ at, seek time.Duration }

// pick chooses the frames to show of a clip whose frames show at times, which are sorted.
//
// A slice's moment is its middle rather than its start, so the first frame is not the
// black one a clip so often opens on and the last is not past the end. The frame wanted
// is the one showing then, the last that began by it. ffmpeg's -ss gives the first frame
// that begins at or after the time asked, so a moment inside the last frame's display
// time, past where it began, finds no frame at all (a one second clip at 4 frames a
// second has none after 750ms). Seeking to the middle of the gap before the wanted frame
// finds that frame and no other. Two moments in the same frame are one sample, which is
// how a clip of fewer frames than asked gives the frames it has.
func pick(times []time.Duration, length time.Duration, count int) []slot {
	var picked []slot
	last := -1
	for i := range count {
		at := time.Duration(float64(length) * (float64(i) + 0.5) / float64(count))
		frame := max(sort.Search(len(times), func(j int) bool { return times[j] > at })-1, 0)
		if frame == last {
			continue
		}
		last = frame
		var seek time.Duration
		if frame > 0 {
			seek = (times[frame-1] + times[frame]) / 2
		}
		picked = append(picked, slot{at: at, seek: seek})
	}
	return picked
}

// load reads a clip's bytes out of a data URI or off the network.
func load(ctx context.Context, client *http.Client, source string) ([]byte, error) {
	if rest, ok := strings.CutPrefix(source, "data:"); ok {
		mime, encoded, found := strings.Cut(rest, ";base64,")
		if !found || !strings.HasPrefix(mime, "video/") {
			return nil, stack.Wrap(errors.New("video: a data URI must be base64 with a video/ media type"))
		}
		if base64.StdEncoding.DecodedLen(len(encoded)) > MaxBytes {
			return nil, stack.Wrap(fmt.Errorf("video: a clip is at most %d MB", MaxBytes>>20))
		}
		clip, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("video: the data URI is not valid base64: %w", err))
		}
		return clip, nil
	}

	address, err := url.Parse(source)
	if err != nil || (address.Scheme != "https" && address.Scheme != "http") || address.Hostname() == "" || address.User != nil {
		return nil, stack.Wrap(errors.New("video: url must be an absolute HTTP(S) URL without credentials, or a data URI"))
	}
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address.String(), nil)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request.Header.Set("User-Agent", egress.UserAgent)
	response, err := client.Do(request)
	if err != nil {
		if errors.Is(err, errPrivate) {
			return nil, stack.Wrap(errPrivate)
		}
		return nil, stack.Wrap(fmt.Errorf("video: fetching the clip: %w", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, stack.Wrap(fmt.Errorf("video: fetching the clip: %s", response.Status))
	}
	clip, err := io.ReadAll(io.LimitReader(response.Body, MaxBytes+1))
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("video: fetching the clip: %w", err))
	}
	if len(clip) > MaxBytes {
		return nil, stack.Wrap(fmt.Errorf("video: a clip is at most %d MB", MaxBytes>>20))
	}
	return clip, nil
}

// duration is how long the clip at path plays for.
func duration(ctx context.Context, path string) (time.Duration, error) {
	args := append([]string{"-v", "error"}, readOnlyAClip...)
	out, err := exec.CommandContext(ctx, "ffprobe", append(args,
		"-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path)...).Output()
	seconds, parseErr := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil || parseErr != nil || seconds <= 0 {
		return 0, stack.Wrap(errors.New("video: not a video ffmpeg can read"))
	}
	return time.Duration(seconds * float64(time.Second)), nil
}

// frameTimes is when each frame of the clip at path begins to show, in order, counted from
// the clip's start as ffmpeg's -ss counts. ffprobe's lines are read as they come, so a clip
// of millions of frames costs the router only the times, not ffprobe's whole listing.
func frameTimes(ctx context.Context, path string) ([]time.Duration, error) {
	errNotAVideo := errors.New("video: not a video ffmpeg can read")
	args := append([]string{"-v", "error", "-select_streams", "v:0"}, readOnlyAClip...)
	command := exec.CommandContext(ctx, "ffprobe", append(args,
		"-show_entries", "format=start_time:packet=pts_time", "-of", "csv", path)...)
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if err := command.Start(); err != nil {
		return nil, stack.Wrap(errNotAVideo)
	}
	var start time.Duration
	var times []time.Duration
	lines := bufio.NewScanner(stdout)
	for lines.Scan() {
		kind, value, _ := strings.Cut(lines.Text(), ",")
		seconds, err := strconv.ParseFloat(value, 64)
		switch {
		case err != nil:
		case kind == "format":
			start = time.Duration(seconds * float64(time.Second))
		case kind == "packet":
			times = append(times, time.Duration(seconds*float64(time.Second)))
		}
	}
	if lines.Err() != nil {
		// ffprobe blocked on a full pipe would never exit; closing it ends its writes.
		_ = stdout.Close()
	}
	// Wait closes the pipe, so it comes after the last read. It runs even when the read
	// failed, or ffprobe is never reaped.
	waitErr := command.Wait()
	if lines.Err() != nil || waitErr != nil || len(times) == 0 {
		return nil, stack.Wrap(errNotAVideo)
	}
	for i := range times {
		times[i] -= start
	}
	slices.Sort(times)
	return times, nil
}

// seekTime is a moment as ffmpeg's -ss takes it, to the microsecond.
func seekTime(at time.Duration) string {
	return strconv.FormatFloat(at.Seconds(), 'f', 6, 64)
}

// frameAt is the first frame showing from a moment of the clip on, as a JPEG.
func frameAt(ctx context.Context, path string, at time.Duration) ([]byte, error) {
	var stderr bytes.Buffer
	args := append([]string{"-v", "error"}, readOnlyAClip...)
	command := exec.CommandContext(ctx, "ffmpeg", append(args,
		"-ss", seekTime(at), "-i", path,
		"-frames:v", "1", "-vf", fmt.Sprintf("scale='min(%d,iw)':-2", frameWidth),
		"-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-")...)
	command.Stderr = &stderr
	jpeg, err := command.Output()
	if err != nil || len(jpeg) == 0 {
		return nil, stack.Wrap(fmt.Errorf("video: reading the frame at %s: %s", at, strings.TrimSpace(stderr.String())))
	}
	return jpeg, nil
}

// publicClient fetches only from the public internet, and never through a proxy, which
// would dial on its behalf where the check below cannot see.
var publicClient = &http.Client{
	Transport: &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second,
			Control: publicOnly,
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 20 * time.Second,
	},
}

// publicOnly refuses to connect to an address that is not public, by the same list as
// connector traffic (egress.IsPublic).
func publicOnly(_, address string, _ syscall.RawConn) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return err
	}
	ip, err := netip.ParseAddr(host)
	if err != nil || !egress.IsPublic(ip) {
		return errPrivate
	}
	return nil
}
