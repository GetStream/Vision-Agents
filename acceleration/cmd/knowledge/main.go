// Command knowledge fills a knowledge base with documentation an agent can look things up
// in.
//
// It is the writing half of internal/knowledge, which only ever reads: an agent searches a
// namespace mid-sentence, and something has to have put the passages there. Nothing is
// embedded, because nothing that reads them embeds either. The files are cut into passages,
// indexed for full-text search, and found again by the words the caller happens to use.
//
// Usage:
//
//	go run ./cmd/knowledge -namespace docs ../docs README.md
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
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
)

// writeTimeout is generous because ingesting is not on anybody's conversation: the store's
// own default is tuned for a lookup somebody is waiting on.
const writeTimeout = 60 * time.Second

func main() {
	namespace := flag.String("namespace", "", "knowledge base to write into")
	size := flag.Int("chunk", ingest.DefaultChunk, "characters per passage")
	dryRun := flag.Bool("dry-run", false, "print the passages instead of writing them")
	verbose := flag.Bool("verbose", false, "log what is written")
	flag.Parse()

	level := slog.LevelWarn
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	options := options{namespace: *namespace, size: *size, dryRun: *dryRun}
	if err := run(options, flag.Args(), logger); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

type options struct {
	namespace string
	size      int
	dryRun    bool
}

func run(options options, paths []string, logger *slog.Logger) error {
	if strings.TrimSpace(options.namespace) == "" {
		return errors.New("a namespace is required, knowledge is never shared")
	}
	if len(paths) == 0 {
		return errors.New("name the files or directories to ingest")
	}
	if options.size <= 0 {
		return errors.New("a passage needs room for something to read")
	}

	documents, err := ingest.Read(paths, options.size)
	if err != nil {
		return err
	}
	if len(documents) == 0 {
		return fmt.Errorf("there is nothing to read in %s", strings.Join(paths, ", "))
	}

	// A dry run is what tells you whether the documents were cut somewhere sensible
	// before a namespace is filled with passages that answer nothing.
	if options.dryRun {
		for _, document := range documents {
			fmt.Printf("%-60s %5d characters\n", document.Source, len(document.Text))
		}
		fmt.Printf("\n%d passages, nothing written\n", len(documents))
		return nil
	}

	base, err := turbopuffer.New(turbopuffer.Options{Timeout: writeTimeout, Logger: logger})
	if err != nil {
		return err
	}
	defer base.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := base.Upsert(ctx, options.namespace, documents); err != nil {
		return err
	}
	fmt.Printf("wrote %d passages from %d files to %s\n",
		len(documents), ingest.Files(documents), options.namespace)
	return nil
}
