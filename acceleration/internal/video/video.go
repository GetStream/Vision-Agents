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
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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

// errPrivate is a fetch that would have reached somewhere that is not the public internet.
var errPrivate = errors.New("video: the URL resolves to an address that is not public")

// Frame is one moment of a clip.
type Frame struct {
	At    time.Duration
	Image llm.ImagePart
}

// Frames samples a clip into count evenly spaced frames, and says how long the clip is.
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
		return nil, 0, fmt.Errorf("video: max_frames must be between 1 and %d", MaxFrames)
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return nil, 0, ErrUnavailable
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		return nil, 0, ErrUnavailable
	}

	clip, err := load(ctx, client, source)
	if err != nil {
		return nil, 0, err
	}
	file, err := os.CreateTemp("", "clip-*")
	if err != nil {
		return nil, 0, err
	}
	defer os.Remove(file.Name())
	_, err = file.Write(clip)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return nil, 0, err
	}

	length, err := duration(ctx, file.Name())
	if err != nil {
		return nil, 0, err
	}
	sampled := make([]Frame, 0, count)
	for i := range count {
		// The middle of each slice rather than its start, so the first frame is not the
		// black one a clip so often opens on and the last is not past the end.
		at := time.Duration(float64(length) * (float64(i) + 0.5) / float64(count))
		jpeg, err := frameAt(ctx, file.Name(), at)
		if err != nil {
			return nil, 0, err
		}
		sampled = append(sampled, Frame{At: at, Image: llm.ImagePart{MIME: "image/jpeg", Data: jpeg}})
	}
	return sampled, length, nil
}

// load reads a clip's bytes out of a data URI or off the network.
func load(ctx context.Context, client *http.Client, source string) ([]byte, error) {
	if rest, ok := strings.CutPrefix(source, "data:"); ok {
		mime, encoded, found := strings.Cut(rest, ";base64,")
		if !found || !strings.HasPrefix(mime, "video/") {
			return nil, errors.New("video: a data URI must be base64 with a video/ media type")
		}
		if base64.StdEncoding.DecodedLen(len(encoded)) > MaxBytes {
			return nil, fmt.Errorf("video: a clip is at most %d MB", MaxBytes>>20)
		}
		clip, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("video: the data URI is not valid base64: %w", err)
		}
		return clip, nil
	}

	address, err := url.Parse(source)
	if err != nil || (address.Scheme != "https" && address.Scheme != "http") || address.Hostname() == "" || address.User != nil {
		return nil, errors.New("video: url must be an absolute HTTP(S) URL without credentials, or a data URI")
	}
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address.String(), nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		if errors.Is(err, errPrivate) {
			return nil, errPrivate
		}
		return nil, fmt.Errorf("video: fetching the clip: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("video: fetching the clip: %s", response.Status)
	}
	clip, err := io.ReadAll(io.LimitReader(response.Body, MaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("video: fetching the clip: %w", err)
	}
	if len(clip) > MaxBytes {
		return nil, fmt.Errorf("video: a clip is at most %d MB", MaxBytes>>20)
	}
	return clip, nil
}

// duration is how long the clip at path plays for.
func duration(ctx context.Context, path string) (time.Duration, error) {
	out, err := exec.CommandContext(ctx, "ffprobe", "-v", "error",
		"-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path).Output()
	seconds, parseErr := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil || parseErr != nil || seconds <= 0 {
		return 0, errors.New("video: not a video ffmpeg can read")
	}
	return time.Duration(seconds * float64(time.Second)), nil
}

// frameAt is the frame showing at a moment of the clip, as a JPEG.
func frameAt(ctx context.Context, path string, at time.Duration) ([]byte, error) {
	var stderr bytes.Buffer
	command := exec.CommandContext(ctx, "ffmpeg", "-v", "error",
		"-ss", strconv.FormatFloat(at.Seconds(), 'f', 3, 64), "-i", path,
		"-frames:v", "1", "-vf", fmt.Sprintf("scale='min(%d,iw)':-2", frameWidth),
		"-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-")
	command.Stderr = &stderr
	jpeg, err := command.Output()
	if err != nil || len(jpeg) == 0 {
		return nil, fmt.Errorf("video: reading the frame at %s: %s", at, strings.TrimSpace(stderr.String()))
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
			Control: func(_, address string, _ syscall.RawConn) error {
				host, _, err := net.SplitHostPort(address)
				if err != nil {
					return err
				}
				ip, err := netip.ParseAddr(host)
				if err != nil || !public(ip) {
					return errPrivate
				}
				return nil
			},
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 20 * time.Second,
	},
}

// sharedAddressSpace is carrier-grade NAT, which is not private by Go's reckoning but is
// no more the public internet than 10/8 is.
var sharedAddressSpace = netip.MustParsePrefix("100.64.0.0/10")

func public(ip netip.Addr) bool {
	ip = ip.Unmap()
	return ip.IsGlobalUnicast() && !ip.IsPrivate() && !sharedAddressSpace.Contains(ip)
}
