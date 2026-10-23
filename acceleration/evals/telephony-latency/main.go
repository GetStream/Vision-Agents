// Command telephony-latency measures how much delay a phone call adds, by carrier and by
// city, and how much of it the calling carrier adds against the called one.
//
//	telephony-latency provision -vendors twilio,telnyx -per-city 2
//	telephony-latency run -host 203.0.113.7 -region us-east1 -runs 10
//	telephony-latency report -out report.md results-us-east1.jsonl results-us-west1.jsonl
//	telephony-latency release
//
// Provision buys numbers in the ten most populous US cities and records them in a file.
// Run is the bench: a small SIP endpoint on a public address, one per region. Every number
// is pointed at it, then every number calls every other one in its city. The callee's
// carrier delivers the call here and the caller's carrier bridges its own side here too,
// so both ends of the call are on one machine. A timestamp sent as tones on one leg and
// decoded off the other is the latency, each way.
//
// See .factory/evals/telephony-latency.md for what is measured and why.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/locate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/twilio"
)

// minGap is the shortest time between stamps on a call. The two directions go half a gap
// apart, and each needs room for a 360 ms stamp and the quiet after it.
const minGap = time.Second

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	var err error
	switch os.Args[1] {
	case "provision":
		err = provision(ctx, os.Args[2:], log)
	case "run":
		err = run(ctx, os.Args[2:], log)
	case "report":
		err = report(ctx, os.Args[2:], log)
	case "release":
		err = release(ctx, os.Args[2:], log)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: telephony-latency <provision|run|report|release> [flags]")
}

func provision(ctx context.Context, arguments []string, log *slog.Logger) error {
	flags := flag.NewFlagSet("provision", flag.ExitOnError)
	numbers := flags.String("numbers", "numbers.json", "file the bought numbers are recorded in")
	vendorList := flags.String("vendors", "twilio,telnyx", "vendors to buy from")
	cityList := flags.String("cities", "", "cities to buy in, comma separated (default: all ten)")
	perCity := flags.Int("per-city", 2, "numbers per vendor per city; two lets a vendor call itself")
	_ = flags.Parse(arguments)

	providers, err := newProviders(list(*vendorList))
	if err != nil {
		return err
	}
	cities, err := pickCities(list(*cityList))
	if err != nil {
		return err
	}
	lines, err := Provision(ctx, providers, cities, *perCity, *numbers, log)
	log.Info("telephony-latency: numbers on file", "count", len(lines), "file", *numbers)
	return err
}

func run(ctx context.Context, arguments []string, log *slog.Logger) error {
	flags := flag.NewFlagSet("run", flag.ExitOnError)
	numbers := flags.String("numbers", "numbers.json", "file of numbers from provision")
	hostFlag := flags.String("host", "", "public IPv4 address carriers reach this machine at (required)")
	listen := flags.String("listen", "0.0.0.0:5060", "UDP address to take SIP on")
	rtpRange := flags.String("rtp-ports", "20000-20999", "UDP ports for media, as low-high")
	region := flags.String("region", "", "where this bench runs, e.g. us-east1; recorded with every call")
	vendorList := flags.String("vendors", "", "only these vendors' numbers (default: all on file)")
	cityList := flags.String("cities", "", "only these cities' numbers (default: all on file)")
	scope := flags.String("scope", string(ScopeCity), "city: numbers call others in their city; all: every number calls every other")
	runs := flags.Int("runs", 10, "calls per pair of numbers")
	parallel := flags.Int("parallel", 4, "calls at once")
	stamps := flags.Int("stamps", 5, "timestamps sent each way on a call")
	gap := flags.Duration("gap", 2*time.Second, "time between timestamps on a call")
	patience := flags.Duration("patience", 30*time.Second, "how long a leg may take to arrive")
	out := flags.String("out", "", "results file, appended to (default: results-<region>.jsonl)")
	skipInbound := flags.Bool("skip-inbound", false, "leave the numbers pointed where they are")
	_ = flags.Parse(arguments)

	host, err := netip.ParseAddr(*hostFlag)
	if err != nil || !host.Is4() {
		return fmt.Errorf("-host must be this machine's public IPv4 address, got %q", *hostFlag)
	}
	if *gap < minGap {
		return fmt.Errorf("-gap must be at least %s, room for a stamp each way and the quiet after them", minGap)
	}
	if Scope(*scope) != ScopeCity && Scope(*scope) != ScopeAll {
		return fmt.Errorf("-scope must be %s or %s", ScopeCity, ScopeAll)
	}
	low, high, err := portRange(*rtpRange)
	if err != nil {
		return err
	}
	rtpPorts, err := newPorts(low, high)
	if err != nil {
		return err
	}
	sipPort, err := listenPort(*listen)
	if err != nil {
		return err
	}
	if *out == "" {
		*out = "results-" + or(*region, "local") + ".jsonl"
	}

	all, err := LoadLines(*numbers)
	if err != nil {
		return err
	}
	lines := all.Only(list(*vendorList), list(*cityList))
	if len(lines) < 2 {
		return fmt.Errorf("%s has %d matching numbers; provision some first", *numbers, len(lines))
	}
	vendorNames := map[string]bool{}
	for _, line := range lines {
		vendorNames[line.Vendor] = true
	}
	providers, err := newProviders(sortedKeys(vendorNames))
	if err != nil {
		return err
	}

	var telnyxCaller *TelnyxCaller
	if vendorNames["telnyx"] {
		if telnyxCaller, err = NewTelnyxCaller(&http.Client{Timeout: 30 * time.Second}); err != nil {
			return err
		}
	}

	endpoint, err := NewEndpoint(*listen, host, rtpPorts, log)
	if err != nil {
		return err
	}
	defer endpoint.Close()

	if !*skipInbound {
		for _, line := range lines {
			bridge := phone.Bridge{URI: fmt.Sprintf("sip:%s@%s:%d", line.E164, host, sipPort)}
			if err := providers[line.Vendor].ConfigureInbound(ctx, phone.Inbound{E164: line.E164, Bridge: bridge}); err != nil {
				return fmt.Errorf("point %s at the bench: %w", line.E164, err)
			}
		}
		log.Info("telephony-latency: numbers point at the bench", "count", len(lines))
	}

	file, err := os.OpenFile(*out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open %s: %w", *out, err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)

	prober := &Prober{
		Endpoint:  endpoint,
		Providers: providers,
		Telnyx:    telnyxCaller,
		Host:      host,
		Port:      sipPort,
		Stamps:    *stamps,
		Gap:       *gap,
		Patience:  *patience,
	}
	tasks := Plan(lines, Scope(*scope), *runs)
	log.Info("telephony-latency: starting", "calls", len(tasks), "region", *region, "out", *out)

	done, failed := 0, 0
	var writeErr error
	Execute(ctx, tasks, *parallel, func(ctx context.Context, task Task) Result {
		result := prober.Probe(ctx, task.Caller, task.Callee)
		result.Region = *region
		result.Run = task.Run
		return result
	}, func(result Result) {
		if ctx.Err() != nil && result.Error != "" {
			// Cut short by stopping the run, which says nothing about the carriers.
			return
		}
		done++
		if result.Error != "" {
			failed++
		}
		if err := encoder.Encode(result); err != nil && writeErr == nil {
			writeErr = fmt.Errorf("write %s: %w", *out, err)
		}
		log.Info("telephony-latency: call",
			"done", fmt.Sprintf("%d/%d", done, len(tasks)),
			"caller", result.Caller.Vendor+" "+result.Caller.City,
			"callee", result.Callee.Vendor+" "+result.Callee.City,
			"forward_ms", result.ForwardMs, "backward_ms", result.BackwardMs, "err", result.Error)
	})
	log.Info("telephony-latency: finished", "calls", done, "failed", failed, "out", *out)
	return errors.Join(writeErr, ctx.Err())
}

func report(ctx context.Context, arguments []string, log *slog.Logger) error {
	flags := flag.NewFlagSet("report", flag.ExitOnError)
	out := flags.String("out", "report.md", "where to write the report")
	withPlaces := flags.Bool("locate", true, "look up where each vendor's addresses are")
	_ = flags.Parse(arguments)
	if flags.NArg() == 0 {
		return errors.New("report needs one or more results files")
	}

	var results []Result
	for _, path := range flags.Args() {
		read, err := readResults(path)
		if err != nil {
			return err
		}
		results = append(results, read...)
	}

	places := map[string]locate.Place{}
	if *withPlaces {
		locator, err := locate.NewLocator(ctx, &http.Client{Timeout: 30 * time.Second})
		if err != nil {
			log.Warn("telephony-latency: some cloud ranges are missing", "err", err)
		}
		for _, result := range results {
			for _, address := range []string{result.CallerMedia, result.CalleeMedia, result.CallerSignal, result.CalleeSignal} {
				parsed, err := netip.ParseAddrPort(address)
				if err != nil {
					continue
				}
				ip := parsed.Addr()
				if _, done := places[ip.String()]; done {
					continue
				}
				place, err := locator.Locate(ctx, ip)
				if err != nil {
					log.Warn("telephony-latency: locate", "ip", ip, "err", err)
				}
				places[ip.String()] = place
			}
		}
	}

	if err := os.WriteFile(*out, []byte(Render(results, places)), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", *out, err)
	}
	log.Info("telephony-latency: report written", "calls", len(results), "out", *out)
	return nil
}

func release(ctx context.Context, arguments []string, log *slog.Logger) error {
	flags := flag.NewFlagSet("release", flag.ExitOnError)
	numbers := flags.String("numbers", "numbers.json", "file of numbers from provision")
	_ = flags.Parse(arguments)

	lines, err := LoadLines(*numbers)
	if err != nil {
		return err
	}
	vendorNames := map[string]bool{}
	for _, line := range lines {
		vendorNames[line.Vendor] = true
	}
	providers, err := newProviders(sortedKeys(vendorNames))
	if err != nil {
		return err
	}
	return Release(ctx, providers, *numbers, log)
}

// newProviders builds the named vendors from their environment variables.
func newProviders(names []string) (map[string]phone.Provider, error) {
	providers := map[string]phone.Provider{}
	for _, name := range names {
		var provider phone.Provider
		var err error
		switch name {
		case "twilio":
			provider, err = twilio.New(twilio.Options{})
		case "telnyx":
			provider, err = telnyx.New(telnyx.Options{})
		default:
			return nil, fmt.Errorf("%q is not a vendor the bench runs; it knows twilio and telnyx", name)
		}
		if err != nil {
			return nil, err
		}
		providers[name] = provider
	}
	return providers, nil
}

func pickCities(names []string) ([]City, error) {
	if len(names) == 0 {
		return Cities, nil
	}
	var picked []City
	for _, name := range names {
		found := false
		for _, city := range Cities {
			if strings.EqualFold(city.Name, name) {
				picked = append(picked, city)
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("%q is not one of the bench's cities", name)
		}
	}
	return picked, nil
}

func readResults(path string) ([]Result, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	var results []Result
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for line := 1; scanner.Scan(); line++ {
		if len(strings.TrimSpace(scanner.Text())) == 0 {
			continue
		}
		var result Result
		if err := json.Unmarshal(scanner.Bytes(), &result); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		results = append(results, result)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return results, nil
}

func portRange(text string) (int, int, error) {
	lowText, highText, ok := strings.Cut(text, "-")
	low, lowErr := strconv.Atoi(lowText)
	high, highErr := strconv.Atoi(highText)
	if !ok || lowErr != nil || highErr != nil {
		return 0, 0, fmt.Errorf("-rtp-ports must be low-high, got %q", text)
	}
	return low, high, nil
}

func listenPort(listen string) (int, error) {
	_, portText, err := net.SplitHostPort(listen)
	if err != nil {
		return 0, fmt.Errorf("-listen must be host:port, got %q", listen)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		return 0, fmt.Errorf("-listen must be host:port, got %q", listen)
	}
	return port, nil
}

func list(text string) []string {
	var items []string
	for item := range strings.SplitSeq(text, ",") {
		if item = strings.TrimSpace(item); item != "" {
			items = append(items, item)
		}
	}
	return items
}
