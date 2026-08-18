// Command phone buys and operates the numbers agents answer on.
//
//	phone vendors
//	phone search -vendor twilio -country US -area 512
//	phone buy -vendor twilio -number +15125551234 -tag project=support
//	phone attach -number +15125551234 -call support-line
//	phone dial -from +15125551234 -to +15550001111
//	phone transfer -from +15125551234 -to +15550002222 -call support-line
//	phone press -vendor telnyx -call-id v3:abc -digits 1
//	phone list
//	phone release -number +15125551234
//
// Stream's SIP is inbound only today, so dialling out is the vendor placing the call and
// bridging it into a trunk the agent is already on. A transfer is the same thing aimed at
// a call that already exists: the human becomes a third party on it rather than the caller
// being handed anywhere.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"text/tabwriter"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	postgresEnvVar    = "ROUTER_POSTGRES_DSN"
	redisEnvVar       = "ROUTER_REDIS_ADDR"
	phoneConfigEnvVar = "ROUTER_PHONE_CONFIG"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	command := os.Args[1]
	arguments := os.Args[2:]

	if err := run(command, arguments); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr,
		"usage: phone <vendors|search|buy|attach|dial|transfer|press|list|release> [flags]")
}

func run(command string, arguments []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	switch command {
	case "vendors":
		return listVendors()
	case "search":
		return search(ctx, arguments)
	case "buy":
		return buy(ctx, arguments)
	case "attach":
		return attach(ctx, arguments)
	case "dial":
		return dial(ctx, arguments)
	case "transfer":
		return transfer(ctx, arguments)
	case "press":
		return press(ctx, arguments)
	case "list":
		return list(ctx, arguments)
	case "release":
		return release(ctx, arguments)
	default:
		usage()
		return fmt.Errorf("unknown command %q", command)
	}
}

// listVendors says which vendors are known, which are implemented and which are ready to
// use, which is the first question when nothing works.
func listVendors() error {
	registry, err := registry()
	if err != nil {
		return err
	}

	out := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(out, "VENDOR\tSTATUS\tCAPABILITIES")
	for _, vendor := range registry.Vendors() {
		fmt.Fprintf(out, "%s\t%s\t%s\n", vendor.Vendor, status(vendor), capabilities(vendor))
	}
	return out.Flush()
}

func status(vendor phone.Vendor) string {
	if !vendor.Implemented {
		return "not implemented"
	}
	if missing := vendor.Missing(); len(missing) > 0 {
		return "needs " + strings.Join(missing, ", ")
	}
	return "ready"
}

func capabilities(vendor phone.Vendor) string {
	rendered := make([]string, 0, len(vendor.Capabilities))
	for _, capability := range vendor.Capabilities {
		rendered = append(rendered, string(capability))
	}
	return strings.Join(rendered, ",")
}

func search(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("search", flag.ExitOnError)
	vendor := flags.String("vendor", "twilio", "vendor to search at")
	country := flags.String("country", "US", "ISO 3166-1 alpha-2 country code")
	area := flags.String("area", "", "area code")
	contains := flags.String("contains", "", "digits the number must contain")
	limit := flags.Int("limit", 10, "how many to show")
	if err := flags.Parse(arguments); err != nil {
		return err
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	offered, err := service.Search(ctx, *vendor, phone.Search{
		Country:      *country,
		AreaCode:     *area,
		Contains:     *contains,
		Limit:        *limit,
		Capabilities: []phone.Capability{phone.Voice},
	})
	if err != nil {
		return err
	}
	if len(offered) == 0 {
		fmt.Println("no numbers offered")
		return nil
	}

	out := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(out, "NUMBER\tCOUNTRY\tWHERE\tPER MONTH")
	for _, number := range offered {
		fmt.Fprintf(out, "%s\t%s\t%s\t%s\n",
			number.E164, number.Country, where(number), dollars(number.MonthlyCostMicros))
	}
	return out.Flush()
}

func buy(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("buy", flag.ExitOnError)
	vendor := flags.String("vendor", "twilio", "vendor to buy from")
	number := flags.String("number", "", "number in +15551234567 form")
	customer := flags.String("customer", "demo", "customer the number is billed to")
	var tags routing.TagsFlag
	flags.Var(&tags, "tag", "cost label as key=value, repeat for several")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *number == "" {
		return errors.New("a number is required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	bought, err := service.Buy(ctx, phone.Purchase{
		Vendor: *vendor,
		E164:   *number,
		Owner:  routing.Owner{CustomerID: *customer, Tags: tags.Tags},
	})
	if err != nil {
		return err
	}

	fmt.Printf("bought %s from %s for %s per month\n",
		bought.E164, bought.Vendor, dollars(bought.MonthlyCostMicros))
	fmt.Printf("attach it with: phone attach -number %s\n", bought.E164)
	return nil
}

func attach(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("attach", flag.ExitOnError)
	number := flags.String("number", "", "number to point at an agent")
	customer := flags.String("customer", "demo", "customer that holds the number")
	call := flags.String("call", "", "call every caller joins, empty gives each their own")
	callType := flags.String("call-type", "", "stream call type, empty means default")
	allowed := flags.String("allowed-ips", "", "comma separated vendor signalling IPs or CIDRs")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *number == "" {
		return errors.New("a number is required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	attached, err := service.Attach(ctx, phone.Attachment{
		CustomerID: *customer,
		E164:       *number,
		CallID:     *call,
		CallType:   *callType,
		AllowedIPs: split(*allowed),
	})
	if err != nil {
		return err
	}

	fmt.Printf("%s now reaches trunk %s at %s\n", *number, attached.TrunkID, attached.Bridge.URI)
	return nil
}

func dial(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("dial", flag.ExitOnError)
	from := flags.String("from", "", "one of your numbers, which is what they see")
	to := flags.String("to", "", "who to call")
	customer := flags.String("customer", "demo", "customer the call is billed to")
	bridge := flags.String("bridge", "", "sip uri of the trunk to join, empty makes one")
	var tags routing.TagsFlag
	flags.Var(&tags, "tag", "cost label as key=value, repeat for several")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *from == "" || *to == "" {
		return errors.New("a from and a to are required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	placed, err := service.Call(ctx, phone.CallRequest{
		Owner:  routing.Owner{CustomerID: *customer, Tags: tags.Tags},
		From:   *from,
		To:     *to,
		Bridge: phone.Bridge{URI: *bridge},
	})
	if err != nil {
		return err
	}

	fmt.Printf("calling %s, vendor call %s (%s)\n", *to, placed.VendorCallID, placed.Status)
	return nil
}

func transfer(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("transfer", flag.ExitOnError)
	from := flags.String("from", "", "one of your numbers, which is what the human sees")
	to := flags.String("to", "", "the human to bring onto the call")
	call := flags.String("call", "", "the stream call the caller and agent are already on")
	callType := flags.String("call-type", "", "stream call type, empty means default")
	customer := flags.String("customer", "demo", "customer the call is billed to")
	var tags routing.TagsFlag
	flags.Var(&tags, "tag", "cost label as key=value, repeat for several")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *from == "" || *to == "" {
		return errors.New("a from and a to are required")
	}
	if *call == "" {
		return errors.New("the call to transfer into is required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	placed, err := service.Transfer(ctx, phone.TransferRequest{
		Owner:    routing.Owner{CustomerID: *customer, Tags: tags.Tags},
		From:     *from,
		To:       *to,
		CallID:   *call,
		CallType: *callType,
	})
	if err != nil {
		return err
	}

	fmt.Printf("bringing %s onto %s, vendor call %s (%s)\n",
		*to, *call, placed.VendorCallID, placed.Status)
	return nil
}

func press(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("press", flag.ExitOnError)
	vendor := flags.String("vendor", "telnyx", "vendor carrying the call")
	callID := flags.String("call-id", "", "the vendor call id dialling returned")
	digits := flags.String("digits", "", "what to press, e.g. 1 or 4123")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *callID == "" {
		return errors.New("the vendor call id is required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	if err := service.SendDigits(ctx, *vendor, *callID, *digits); err != nil {
		return err
	}
	fmt.Printf("pressed %s on %s\n", *digits, *callID)
	return nil
}

func list(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("list", flag.ExitOnError)
	customer := flags.String("customer", "demo", "customer whose numbers to show")
	all := flags.Bool("all", false, "include numbers that have been released")
	if err := flags.Parse(arguments); err != nil {
		return err
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	held, err := service.Numbers(ctx, *customer, *all)
	if err != nil {
		return err
	}
	if len(held) == 0 {
		fmt.Println("no numbers")
		return nil
	}

	out := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(out, "NUMBER\tVENDOR\tTRUNK\tPER MONTH\tSTATE")
	for _, number := range held {
		state := "held"
		if number.ReleasedAt != nil {
			state = "released " + number.ReleasedAt.Format("2006-01-02")
		}
		trunk := number.StreamTrunkID
		if trunk == "" {
			trunk = "-"
		}
		fmt.Fprintf(out, "%s\t%s\t%s\t%s\t%s\n",
			number.E164, number.Vendor, trunk, dollars(number.MonthlyCostMicros), state)
	}
	return out.Flush()
}

func release(ctx context.Context, arguments []string) error {
	flags := flag.NewFlagSet("release", flag.ExitOnError)
	number := flags.String("number", "", "number to give back")
	customer := flags.String("customer", "demo", "customer that holds the number")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *number == "" {
		return errors.New("a number is required")
	}

	service, cleanup, err := build(ctx)
	if err != nil {
		return err
	}
	defer cleanup()

	if err := service.Release(ctx, *customer, *number); err != nil {
		return err
	}
	fmt.Printf("released %s, the monthly charge stops\n", *number)
	return nil
}

// build wires the service from the environment. Everything but the vendor registry is
// optional, so searching for a number works with nothing but a vendor key.
func build(ctx context.Context) (*phone.Service, func(), error) {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	registry, err := registry()
	if err != nil {
		return nil, nil, err
	}

	var closers []func()
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			closers[i]()
		}
	}

	var pgStore *store.Store
	if dsn := os.Getenv(postgresEnvVar); dsn != "" {
		pgStore, err = store.Open(dsn)
		if err != nil {
			return nil, nil, err
		}
		closers = append(closers, func() { pgStore.Close() })
		if err := pgStore.Migrate(ctx); err != nil {
			cleanup()
			return nil, nil, err
		}
	}

	var liveClient *live.Client
	if address := os.Getenv(redisEnvVar); address != "" {
		liveClient, err = live.New(live.Options{Address: address})
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		closers = append(closers, liveClient.Close)
	}

	var recorder *routing.Recorder
	if pgStore != nil || liveClient != nil {
		recorder = routing.NewRecorder(routing.Phone, pgStore, liveClient, logger)
		closers = append(closers, recorder.Close)
	}

	// Stream is only needed to attach a number, so a missing key is not fatal here: the
	// operations that need it say so themselves.
	var stream *phone.Stream
	if streaming, err := phone.NewStream(phone.StreamOptions{}); err == nil {
		stream = streaming
	}

	service, err := phone.NewService(phone.ServiceOptions{
		Registry: registry,
		Store:    pgStore,
		Stream:   stream,
		Recorder: recorder,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	return service, cleanup, nil
}

func registry() (*phone.Registry, error) {
	config, err := phone.LoadConfig(os.Getenv(phoneConfigEnvVar))
	if err != nil {
		return nil, err
	}
	return vendors.Registry(config), nil
}

func where(number phone.Available) string {
	if number.Locality != "" && number.Region != "" {
		return number.Locality + ", " + number.Region
	}
	if number.Locality != "" {
		return number.Locality
	}
	return number.Region
}

func dollars(micros int64) string {
	if micros == 0 {
		return "-"
	}
	return fmt.Sprintf("$%.4f", float64(micros)/1_000_000)
}

func split(list string) []string {
	if strings.TrimSpace(list) == "" {
		return nil
	}
	parts := strings.Split(list, ",")
	trimmed := make([]string, 0, len(parts))
	for _, part := range parts {
		if value := strings.TrimSpace(part); value != "" {
			trimmed = append(trimmed, value)
		}
	}
	return trimmed
}
