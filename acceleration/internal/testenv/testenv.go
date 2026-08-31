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
package testenv

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/joho/godotenv"
)

// init loads the nearest .env at or above the working directory. A test runs with the
// working directory set to its own package, so the file is looked for upwards rather than
// beside it.
//
// A missing file is not a failure: every suite already skips itself on the credentials it
// needs, which says the same thing in a better place. A file that cannot be parsed is
// worth a word, because the suite would otherwise skip as though it were not there.
func init() {
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
