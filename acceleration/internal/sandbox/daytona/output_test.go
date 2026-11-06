package daytona

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
)

// OutputSuite runs code whose output is far larger than what comes back of it, on a Daytona in
// process, and reads cut answers as the SDK would.
type OutputSuite struct {
	suite.Suite
	ctx     context.Context
	daytona *daytonatest.Server
	box     *Sandbox
}

func TestOutputSuite(t *testing.T) {
	suite.Run(t, new(OutputSuite))
}

func (s *OutputSuite) SetupTest() {
	s.ctx = context.Background()
	s.daytona = daytonatest.Start(s.T())
	box, err := New(Options{APIKey: "key-1", APIURL: s.daytona.APIURL(), Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.box = box
	s.T().Cleanup(func() { _ = box.Close() })
}

// printed is a code run's answer whose code printed output times over and exited with exit. The
// exit code comes after the output, so a reader that stopped at the output would lose it.
func printed(output string, times, exit int) func(io.Writer) {
	encoded, _ := json.Marshal(output)
	once := encoded[1 : len(encoded)-1]
	return func(w io.Writer) {
		_, _ = io.WriteString(w, `{"result":"`)
		// Written in blocks, so a large answer is quick to send.
		block := []byte(strings.Repeat(string(once), max(1, (64<<10)/len(once))))
		for left := times; left > 0; left -= len(block) / len(once) {
			_, _ = w.Write(block[:min(left, len(block)/len(once))*len(once)])
		}
		_, _ = fmt.Fprintf(w, `","exitCode":%d}`, exit)
	}
}

// cutAnswer is what the SDK reads of body, a code run's answer.
func cutAnswer(body string) (string, error) {
	read, err := io.ReadAll(&cutOutput{body: io.NopCloser(strings.NewReader(body)), left: maxAnswerBytes})
	return string(read), err
}

func (s *OutputSuite) TestAnOutputFarOverTheBoundComesBackCutWithItsExitCode() {
	line := "row 0123456789 é\n"
	times := (256 << 20) / len(line)
	s.daytona.Answers(printed(line, times, 3))
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	result, err := s.box.Run(s.ctx, "print(rows)", nil)

	runtime.ReadMemStats(&after)
	s.Require().NoError(err)
	s.Equal(3, result.ExitCode)
	s.LessOrEqual(len(result.Output), sandbox.MaxOutputBytes)
	s.True(strings.HasSuffix(result.Output, sandbox.OutputCut), "a cut output ends with the marker")
	kept := strings.TrimSuffix(result.Output, sandbox.OutputCut)
	s.True(strings.HasPrefix(strings.Repeat(line, sandbox.MaxOutputBytes/len(line)+1), kept), "what is kept is the output's start")
	s.GreaterOrEqual(len(kept), sandbox.MaxOutputBytes-len(sandbox.OutputCut)-utf8.UTFMax, "no more is dropped than a character")
	s.True(utf8.ValidString(result.Output))
	// The answer is 256 MiB: read whole, as the SDK reads it, it would be allocated several times.
	s.Less(after.TotalAlloc-before.TotalAlloc, uint64(32<<20), "the router held the answer whole")
}

func (s *OutputSuite) TestTheBoundIsTheToolResultCapAndTheMarkerSaysIt() {
	s.Equal(core.MaxResultBytes, sandbox.MaxOutputBytes, "a run's output is cut where every other tool result is")
	s.Contains(sandbox.OutputCut, fmt.Sprintf("more than %d KiB", sandbox.MaxOutputBytes>>10))
}

func (s *OutputSuite) TestAnOutputUnderTheBoundComesBackAsPrinted() {
	// JSON escapes each of these; the expectation is the output as printed, as the base reads it.
	unit := "<b>a & b</b> \"q\" \\ \t\x00 é 😀  \n"
	output := strings.Repeat(unit, sandbox.MaxOutputBytes/len(unit))
	s.daytona.Answers(printed(output, 1, 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal(output, result.Output)
	s.Zero(result.ExitCode)
}

func (s *OutputSuite) TestAnOutputOfExactlyTheBoundIsNotCut() {
	// Each < is six bytes on the wire, <, so this is the most a fitting output takes.
	output := strings.Repeat("<", sandbox.MaxOutputBytes)
	s.daytona.Answers(printed(output, 1, 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal(output, result.Output)
}

func (s *OutputSuite) TestAnOutputOneByteOverTheBoundIsCut() {
	s.daytona.Answers(printed(strings.Repeat("a", sandbox.MaxOutputBytes+1), 1, 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal(strings.Repeat("a", sandbox.MaxOutputBytes-len(sandbox.OutputCut))+sandbox.OutputCut, result.Output)
}

func (s *OutputSuite) TestAnOutputOverTheBoundWithEscapedQuotesIsCut() {
	// A reader taking \" for the end of the output would pass the rest of it uncut.
	line := `say "hi" \ ` + "\n"
	s.daytona.Answers(printed(line, 2*sandbox.MaxOutputBytes/len(line), 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.LessOrEqual(len(result.Output), sandbox.MaxOutputBytes)
	s.True(strings.HasSuffix(result.Output, sandbox.OutputCut), "a cut output ends with the marker")
	kept := strings.TrimSuffix(result.Output, sandbox.OutputCut)
	s.True(strings.HasPrefix(strings.Repeat(line, sandbox.MaxOutputBytes/len(line)+1), kept), "what is kept is the output's start")
}

func (s *OutputSuite) TestAnOutputWhoseEscapesOutgrowWhatIsHeldIsCut() {
	// Twice the bound of six-byte escapes, \u003c, after five bytes that put the end of what is
	// held five bytes into one: \u003 is the longest escape cut short.
	s.daytona.Answers(printed("aaaaa"+strings.Repeat("<", 2*sandbox.MaxOutputBytes), 1, 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal("aaaaa"+strings.Repeat("<", sandbox.MaxOutputBytes-len(sandbox.OutputCut)-5)+sandbox.OutputCut, result.Output)
}

func (s *OutputSuite) TestACutNeverSplitsACharacter() {
	s.daytona.Answers(printed("😀", sandbox.MaxOutputBytes, 0))

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal(strings.Repeat("😀", (sandbox.MaxOutputBytes-len(sandbox.OutputCut))/4)+sandbox.OutputCut, result.Output)
}

func (s *OutputSuite) TestACutNeverSplitsAnEscapedSurrogatePair() {
	// The 12 bytes of one character, as a toolbox that escapes all but ASCII would send it.
	s.daytona.Answers(func(w io.Writer) {
		_, _ = io.WriteString(w, `{"result":"`+strings.Repeat(`😀`, sandbox.MaxOutputBytes)+`","exitCode":0}`)
	})

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal(strings.Repeat("😀", (sandbox.MaxOutputBytes-len(sandbox.OutputCut))/4)+sandbox.OutputCut, result.Output)
}

func (s *OutputSuite) TestAnAnswerFarOverItsBoundBesidesTheOutputFailsTheRun() {
	// The answer never ends: it is written until the router hangs up.
	ended := make(chan struct{})
	s.daytona.Answers(func(w io.Writer) {
		defer close(ended)
		_, _ = io.WriteString(w, `{"result":"ok\n","exitCode":0,"padding":"`)
		block := []byte(strings.Repeat("x", 64<<10))
		for {
			if _, err := w.Write(block); err != nil {
				return
			}
		}
	})

	_, err := s.box.Run(s.ctx, "print(it)", nil)

	s.ErrorContains(err, "too large to read")
	s.Eventually(func() bool {
		select {
		case <-ended:
			return true
		default:
			return false
		}
	}, 10*time.Second, 10*time.Millisecond, "the router left the answer it stopped reading open")
}

func (s *OutputSuite) TestOnlyTheTopLevelResultIsTheOutput() {
	nested := strings.Repeat("n", 2*sandbox.MaxOutputBytes)
	body := `{"artifacts":{"charts":[{"result":"` + nested + `","elements":[[1,2],{"a":"result"}]}]},` +
		`"result":"` + strings.Repeat("a", 2*sandbox.MaxOutputBytes) + `","exitCode":0}`

	read, err := cutAnswer(body)

	s.Require().NoError(err)
	var answer struct {
		Artifacts struct {
			Charts []struct {
				Result string `json:"result"`
			} `json:"charts"`
		} `json:"artifacts"`
		Result string `json:"result"`
	}
	s.Require().NoError(json.Unmarshal([]byte(read), &answer))
	s.Equal(nested, answer.Artifacts.Charts[0].Result, "a result nested in another field is not cut")
	s.Equal(strings.Repeat("a", sandbox.MaxOutputBytes-len(sandbox.OutputCut))+sandbox.OutputCut, answer.Result)
}

func (s *OutputSuite) TestAValueThatSaysResultIsNotAKey() {
	body := `{"kind":"result","note":"` + strings.Repeat("a", 2*sandbox.MaxOutputBytes) + `","exitCode":0}`

	read, err := cutAnswer(body)

	s.Require().NoError(err)
	s.Equal(body, read)
}

func (s *OutputSuite) TestARepeatedResultIsReadAsTheSDKReadsIt() {
	// Go's JSON decode keeps the last of two values of one key, as the base does.
	s.daytona.Answers(func(w io.Writer) {
		_, _ = io.WriteString(w, `{"result":"`+strings.Repeat("a", 2*maxHeldBytes)+`","result":"short\n","exitCode":0}`)
	})

	result, err := s.box.Run(s.ctx, "print(it)", nil)

	s.Require().NoError(err)
	s.Equal("short\n", result.Output)
}

func (s *OutputSuite) TestAnOutputUnderAnotherSpellingOfResultIsCut() {
	// Go's JSON decode takes "Result", and a key written with an escape, for "result", and keeps
	// the last: the reader does not know them, so Run cuts what the SDK decoded.
	long := strings.Repeat("a", 2*sandbox.MaxOutputBytes)
	cut := strings.Repeat("a", sandbox.MaxOutputBytes-len(sandbox.OutputCut)) + sandbox.OutputCut
	for name, answer := range map[string]string{
		"capitalised":   `{"result":"x","Result":"` + long + `","exitCode":0}`,
		"escaped key":   `{"res\u0075lt":"` + long + `","exitCode":0}`,
		"upper escaped": `{"result":"x","R\u0045SULT":"` + long + `","exitCode":0}`,
	} {
		s.Run(name, func() {
			s.daytona.Answers(func(w io.Writer) { _, _ = io.WriteString(w, answer) })

			result, err := s.box.Run(s.ctx, "print(it)", nil)

			s.Require().NoError(err)
			s.Equal(cut, result.Output)
		})
	}
}

func (s *OutputSuite) TestAKeyThatOnlyStartsWithResultIsNotTheOutput() {
	body := `{"results":"` + strings.Repeat("a", 2*sandbox.MaxOutputBytes) + `","exitCode":0}`

	read, err := cutAnswer(body)

	s.Require().NoError(err)
	s.Equal(body, read)
}

func (s *OutputSuite) TestAnEscapedQuoteOrBackslashDoesNotEndItsString() {
	// One escaped quote, then an escaped backslash: a reader taking \" for the end of the string,
	// or the quote after \\ for part of it, would read the output's key as a value.
	body := `{"note":"say \"hi \\","result":"` + strings.Repeat("a", 2*sandbox.MaxOutputBytes) + `"}`

	read, err := cutAnswer(body)

	s.Require().NoError(err)
	var answer struct{ Note, Result string }
	s.Require().NoError(json.Unmarshal([]byte(read), &answer))
	s.Equal(`say "hi \`, answer.Note)
	s.Equal(strings.Repeat("a", sandbox.MaxOutputBytes-len(sandbox.OutputCut))+sandbox.OutputCut, answer.Result)
}

func (s *OutputSuite) TestAnOutputThatIsNotAJSONStringPassesAsItCame() {
	// The SDK's own decode refuses it, as it would were the answer read whole.
	body := `{"result":"a\x","exitCode":0}`

	read, err := cutAnswer(body)

	s.Require().NoError(err)
	s.Equal(body, read)
}

func (s *OutputSuite) TestAnOutputTooLongToHoldThatIsNotAJSONStringFails() {
	_, err := cutAnswer(`{"result":"\x` + strings.Repeat("a", 2*maxHeldBytes) + `"}`)

	s.ErrorIs(err, errBadOutput)
}

func (s *OutputSuite) TestAnAnswerOfExactlyItsBoundIsReadWhole() {
	body := `{"result":"ok","exitCode":0}`
	answer := &cutOutput{body: io.NopCloser(strings.NewReader(body)), left: int64(len(body))}

	read, err := io.ReadAll(answer)

	s.Require().NoError(err)
	s.Equal(body, string(read))
}

func (s *OutputSuite) TestAnAnswerOneByteOverItsBoundFails() {
	body := `{"result":"ok","exitCode":0}`
	answer := &cutOutput{body: io.NopCloser(strings.NewReader(body)), left: int64(len(body)) - 1}

	_, err := io.ReadAll(answer)

	s.ErrorIs(err, errAnswerTooLarge)
}

func (s *OutputSuite) TestADownloadOfExactlyItsBoundIsReadWhole() {
	body := strings.Repeat("x", 100)

	read, err := io.ReadAll(&capped{body: io.NopCloser(strings.NewReader(body)), left: 100})

	s.Require().NoError(err)
	s.Equal(body, string(read))
}

func (s *OutputSuite) TestADownloadOneByteOverItsBoundFails() {
	_, err := io.ReadAll(&capped{body: io.NopCloser(strings.NewReader(strings.Repeat("x", 101))), left: 100})

	s.ErrorIs(err, errAnswerTooLarge)
}
