package video

import (
	"bytes"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type VideoSuite struct {
	suite.Suite
	clip []byte
}

func TestVideo(t *testing.T) {
	suite.Run(t, new(VideoSuite))
}

// SetupSuite records a four second clip, which is what every sampling test reads.
func (s *VideoSuite) SetupSuite() {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		s.T().Skip("ffmpeg not available to record the clip")
	}
	path := filepath.Join(s.T().TempDir(), "clip.mp4")
	out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi",
		"-i", "testsrc=duration=4:size=320x240:rate=10", "-pix_fmt", "yuv420p", path).CombinedOutput()
	s.Require().NoError(err, string(out))
	s.clip, err = os.ReadFile(path)
	s.Require().NoError(err)
}

func (s *VideoSuite) dataURI() string {
	return "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(s.clip)
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

func (s *VideoSuite) TestOnlyThePublicInternetIsPublic() {
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
	} {
		s.Equal(want, public(netip.MustParseAddr(address)), address)
	}
}
