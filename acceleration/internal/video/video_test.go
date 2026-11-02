package video

import (
	"bytes"
	"encoding/base64"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type VideoSuite struct {
	suite.Suite
	clip []byte
	webm []byte
}

func TestVideo(t *testing.T) {
	suite.Run(t, new(VideoSuite))
}

// SetupSuite records the four second clips the sampling tests read, an mp4 and a webm.
func (s *VideoSuite) SetupSuite() {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		s.T().Skip("ffmpeg not available to record the clip")
	}
	s.clip = s.record("clip.mp4", "-pix_fmt", "yuv420p")
	s.webm = s.record("clip.webm")
}

func (s *VideoSuite) record(name string, options ...string) []byte {
	path := filepath.Join(s.T().TempDir(), name)
	args := append([]string{"-v", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10"}, options...)
	out, err := exec.Command("ffmpeg", append(args, path)...).CombinedOutput()
	s.Require().NoError(err, string(out))
	clip, err := os.ReadFile(path)
	s.Require().NoError(err)
	return clip
}

func (s *VideoSuite) dataURI() string {
	return "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(s.clip)
}

// canary is a named pipe in dir that nothing may open, and says whether something did. A
// pipe rather than a file because opening one for reading waits for a writer, so the open
// is seen even when nothing is read.
func (s *VideoSuite) canary(dir string) (string, func() bool) {
	path := filepath.Join(dir, "canary.ts")
	s.Require().NoError(syscall.Mkfifo(path, 0o600))
	reached := make(chan struct{})
	go func() {
		writer, err := os.OpenFile(path, os.O_WRONLY, 0)
		if err != nil {
			return
		}
		close(reached)
		_ = writer.Close()
	}()
	s.T().Cleanup(func() {
		// Opening the read end lets the writer still waiting above return.
		if reader, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0); err == nil {
			_ = reader.Close()
		}
	})
	return path, func() bool {
		select {
		case <-reached:
			return true
		default:
			return false
		}
	}
}

// hlsPlaylist is an HLS playlist whose one segment is the file at path.
func hlsPlaylist(path string) []byte {
	return []byte("#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.0,\nfile://" + path + "\n#EXT-X-ENDLIST\n")
}

// concatList is an ffmpeg concat list naming a file beside it, as the concat demuxer's
// default safe mode allows.
func concatList(name string) []byte {
	return []byte("ffconcat version 1.0\nfile " + name + "\n")
}

func referenceURI(list []byte) string {
	return "data:video/mp2t;base64," + base64.StdEncoding.EncodeToString(list)
}

// served is a server on this machine handing out the clip, and whether it ever did.
func (s *VideoSuite) served() (string, *atomic.Bool) {
	var hit atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hit.Store(true)
		_, _ = w.Write(s.clip)
	}))
	s.T().Cleanup(server.Close)
	return server.URL + "/clip.mp4", &hit
}

func (s *VideoSuite) TestAClipIsSampledIntoEvenlySpacedFrames() {
	frames, length, err := Frames(s.T().Context(), s.dataURI(), 4)

	s.Require().NoError(err)
	s.InDelta(4*time.Second, length, float64(100*time.Millisecond))
	s.Require().Len(frames, 4)
	for i, frame := range frames {
		want := time.Duration(float64(length) * (float64(i) + 0.5) / 4)
		s.Equal(want, frame.At)
		s.Equal("image/jpeg", frame.Image.MIME)
		s.True(bytes.HasPrefix(frame.Image.Data, []byte{0xFF, 0xD8}), "frame %d is not a JPEG", i)
		s.NoError(frame.Image.Validate())
	}
}

func (s *VideoSuite) TestAWebMClipIsSampled() {
	frames, length, err := Frames(s.T().Context(), "data:video/webm;base64,"+base64.StdEncoding.EncodeToString(s.webm), 2)

	s.Require().NoError(err)
	s.InDelta(4*time.Second, length, float64(100*time.Millisecond))
	s.Require().Len(frames, 2)
	for i, frame := range frames {
		s.True(bytes.HasPrefix(frame.Image.Data, []byte{0xFF, 0xD8}), "frame %d is not a JPEG", i)
	}
}

func (s *VideoSuite) TestAnHLSPlaylistIsRefusedWithoutOpeningWhatItNames() {
	canary, opened := s.canary(s.T().TempDir())

	frames, _, err := Frames(s.T().Context(), referenceURI(hlsPlaylist(canary)), 2)

	s.ErrorContains(err, "not a video ffmpeg can read")
	s.Nil(frames)
	s.False(opened(), "ffmpeg opened the file the playlist names")
}

func (s *VideoSuite) TestAConcatListIsRefusedWithoutOpeningWhatItNames() {
	dir := s.T().TempDir()
	// The clip's temporary file goes beside the canary, where the list's relative name points.
	s.T().Setenv("TMPDIR", dir)
	_, opened := s.canary(dir)

	frames, _, err := Frames(s.T().Context(), referenceURI(concatList("canary.ts")), 2)

	s.ErrorContains(err, "not a video ffmpeg can read")
	s.Nil(frames)
	s.False(opened(), "ffmpeg opened the file the list names")
}

func (s *VideoSuite) TestAFrameIsNotReadThroughAnHLSPlaylist() {
	dir := s.T().TempDir()
	canary, opened := s.canary(dir)
	playlist := filepath.Join(dir, "playlist.m3u8")
	s.Require().NoError(os.WriteFile(playlist, hlsPlaylist(canary), 0o600))

	jpeg, err := frameAt(s.T().Context(), playlist, 0)

	s.Error(err)
	s.Nil(jpeg)
	s.False(opened(), "ffmpeg opened the file the playlist names")
}

func (s *VideoSuite) TestAFrameIsNotReadThroughAConcatList() {
	dir := s.T().TempDir()
	_, opened := s.canary(dir)
	list := filepath.Join(dir, "list.ffconcat")
	s.Require().NoError(os.WriteFile(list, concatList("canary.ts"), 0o600))

	jpeg, err := frameAt(s.T().Context(), list, 0)

	s.Error(err)
	s.Nil(jpeg)
	s.False(opened(), "ffmpeg opened the file the list names")
}

// The path is always a temporary file; these prove nothing else is opened in its place.
func (s *VideoSuite) TestTheLengthIsReadOnlyFromAFile() {
	address, hit := s.served()

	_, err := duration(s.T().Context(), address)

	s.ErrorContains(err, "not a video ffmpeg can read")
	s.False(hit.Load(), "ffprobe fetched the URL")
}

func (s *VideoSuite) TestAFrameIsReadOnlyFromAFile() {
	address, hit := s.served()

	jpeg, err := frameAt(s.T().Context(), address, 0)

	s.Error(err)
	s.Nil(jpeg)
	s.False(hit.Load(), "ffmpeg fetched the URL")
}

func (s *VideoSuite) TestAClipNamingNoCountGetsTheDefault() {
	frames, _, err := Frames(s.T().Context(), s.dataURI(), 0)

	s.Require().NoError(err)
	s.Len(frames, DefaultFrames)
}

func (s *VideoSuite) TestAClipIsFetchedFromAURL() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(s.clip)
	}))
	defer server.Close()

	frames, _, err := frames(s.T().Context(), server.Client(), server.URL+"/clip.mp4", 2)

	s.Require().NoError(err)
	s.Len(frames, 2)
}

func (s *VideoSuite) TestAURLOnThisMachineIsRefused() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(s.clip)
	}))
	defer server.Close()

	_, _, err := Frames(s.T().Context(), server.URL+"/clip.mp4", 2)

	s.ErrorIs(err, errPrivate)
}

func (s *VideoSuite) TestTheCloudMetadataAddressIsRefused() {
	_, _, err := Frames(s.T().Context(), "http://169.254.169.254/latest/meta-data/", 2)

	s.ErrorIs(err, errPrivate)
}

func (s *VideoSuite) TestTooManyFramesAreRefused() {
	_, _, err := Frames(s.T().Context(), s.dataURI(), MaxFrames+1)

	s.ErrorContains(err, "max_frames")
}

func (s *VideoSuite) TestSomethingThatIsNotAVideoIsRefused() {
	_, _, err := Frames(s.T().Context(), "data:video/mp4;base64,"+base64.StdEncoding.EncodeToString([]byte("not a video")), 2)

	s.ErrorContains(err, "not a video")
}

func (s *VideoSuite) TestADataURIThatIsNotAVideoIsRefused() {
	_, _, err := Frames(s.T().Context(), "data:image/png;base64,iVBORw0KGgo=", 2)

	s.ErrorContains(err, "video/")
}

// PublicOnlySuite needs no ffmpeg, so it runs where VideoSuite is skipped.
type PublicOnlySuite struct {
	suite.Suite
}

func TestPublicOnly(t *testing.T) {
	suite.Run(t, new(PublicOnlySuite))
}

func (s *PublicOnlySuite) TestOnlyThePublicInternetIsPublic() {
	for address, want := range map[string]bool{
		"8.8.8.8":          true,
		"2001:4860::8888":  true,
		"10.0.0.1":         false,
		"172.16.0.1":       false,
		"192.168.1.1":      false,
		"100.64.1.1":       false,
		"127.0.0.1":        false,
		"169.254.169.254":  false,
		"0.0.0.0":          false,
		"::1":              false,
		"fd00::1":          false,
		"fe80::1":          false,
		"::ffff:127.0.0.1": false,
		"::ffff:10.1.2.3":  false,
		"0.1.2.3":          false, // "this network", 0.0.0.0/8
		"192.0.0.8":        false, // IETF protocol assignments, 192.0.0.0/24
		"198.19.255.254":   false, // benchmarking, 198.18.0.0/15
		"240.0.0.1":        false, // reserved, 240.0.0.0/4
		"64:ff9b::a00:1":   false, // NAT64 of 10.0.0.1
		"2002:a00:1::1":    false, // 6to4 of 10.0.0.1
		"2001:0:a00:1::1":  false, // Teredo, 2001::/32
	} {
		err := publicOnly("tcp", net.JoinHostPort(address, "443"), nil)
		if want {
			s.NoError(err, address)
		} else {
			s.ErrorIs(err, errPrivate, address)
		}
	}
}

func (s *PublicOnlySuite) TestTheFetchRefusesAServerOnThisMachine() {
	served := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		served = true
		_, _ = w.Write([]byte("clip"))
	}))
	defer server.Close()

	_, err := load(s.T().Context(), publicClient, server.URL+"/clip.mp4")

	s.ErrorIs(err, errPrivate)
	s.False(served, "the request reached the server")
}
