// Package environment holds the settings that differ between the places the router runs:
// a laptop, the hosted staging deployment and the test suites. Each one is a YAML file
// here, embedded in the binary, that names environment variables and their values.
//
// The files hold defaults, never secrets. Credentials stay in .env locally and in the
// chart in a hosted deployment.
package environment

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"

	"gopkg.in/yaml.v3"
)

// EnvVar names the environment to load. Unset means Development.
const EnvVar = "ROUTER_ENV"

const (
	Development = "development"
	Staging     = "staging"
	Testing     = "testing"
)

//go:embed *.yaml
var files embed.FS

// Settings returns the variables the named environment sets.
func Settings(name string) (map[string]string, error) {
	raw, err := files.ReadFile(name + ".yaml")
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("environment: unknown %s %q, want %s, %s or %s", EnvVar, name, Development, Staging, Testing)
	}
	if err != nil {
		return nil, err
	}
	settings := map[string]string{}
	if err := yaml.Unmarshal(raw, &settings); err != nil {
		return nil, fmt.Errorf("environment: parse %s: %w", name, err)
	}
	return settings, nil
}

// Apply loads the environment EnvVar names and sets each of its variables the process
// does not already have, so .env, the shell and a chart all win over the file. It
// returns the name of the environment it loaded.
func Apply() (string, error) {
	name := os.Getenv(EnvVar)
	if name == "" {
		name = Development
	}
	settings, err := Settings(name)
	if err != nil {
		return "", err
	}
	for key, value := range settings {
		if _, set := os.LookupEnv(key); !set {
			if err := os.Setenv(key, value); err != nil {
				return "", err
			}
		}
	}
	return name, nil
}
