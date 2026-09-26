// Package testenv gives the integration suites the credentials that live in the
// repository's .env file, which is where the Python side and a local router already read
// them from.
//
// Go only ever sees the environment of the process it was started in, so a suite launched
// from an editor's run button skips itself for want of a key that is sitting in a file two
// directories up. Importing this package for its side effect is what closes that gap, and
// it does so the same way for a shell, an IDE and CI:
//
//	import _ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
//
// It also loads the testing environment, which points the suites at their own database so
// that running them never touches the one a local router is using.
package testenv

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// init sets the testing environment over whatever the shell has, loads .env for the
// credentials, and creates the test database if it is missing.
func init() {
	if err := os.Setenv(config.EnvVar, config.Testing); err != nil {
		panic(err)
	}
	// Load writes the settings back into the environment, which is where the suites and
	// the packages holding their own credentials read them.
	settings, _, err := config.Load("")
	if err != nil {
		panic(err)
	}
	loadDotEnv()
	createDatabase(settings.Postgres.DSN)
}

// loadDotEnv loads the nearest .env at or above the working directory. A test runs with
// the working directory set to its own package, so the file is looked for upwards rather
// than beside it.
//
// A missing file is not a failure: every suite already skips itself on the credentials it
// needs, which says the same thing in a better place. A file that cannot be parsed is
// worth a word, because the suite would otherwise skip as though it were not there.
func loadDotEnv() {
	dir, err := os.Getwd()
	if err != nil {
		return
	}

	for {
		path := filepath.Join(dir, ".env")
		if _, err := os.Stat(path); err == nil {
			// Load leaves variables that are already set alone, so a value exported in
			// the shell or by CI wins over a stale line in the file.
			if err := godotenv.Load(path); err != nil {
				fmt.Fprintf(os.Stderr, "testenv: %s: %v\n", path, err)
			}
			return
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return
		}
		dir = parent
	}
}

// createDatabase creates the database the DSN names, connecting through the server's
// postgres database to do it. A server that cannot be reached is left to the suites, which
// skip or fail on it themselves.
func createDatabase(dsn string) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return
	}
	name := strings.TrimPrefix(parsed.Path, "/")
	parsed.Path = "/postgres"
	db := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(parsed.String())))
	defer db.Close()

	ctx := context.Background()
	var exists bool
	if err := db.QueryRowContext(ctx, "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name).Scan(&exists); err != nil || exists {
		return
	}
	if _, err := db.ExecContext(ctx, `CREATE DATABASE "`+strings.ReplaceAll(name, `"`, `""`)+`"`); err != nil {
		fmt.Fprintf(os.Stderr, "testenv: create database %s: %v\n", name, err)
	}
}
