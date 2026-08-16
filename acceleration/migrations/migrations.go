// Package migrations embeds the goose SQL migrations so they ship inside the binary.
package migrations

import "embed"

// FS holds the migration files, ordered by their timestamped names.
//
//go:embed *.sql
var FS embed.FS
