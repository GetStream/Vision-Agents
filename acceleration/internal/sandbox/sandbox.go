// Package sandbox runs code the model wrote somewhere it cannot do any harm.
//
// It is offered to the subagent rather than to the voice model. Code execution is slow
// and a conversation is not: a model that stops to run something mid-sentence has stopped
// talking, which is the one thing a voice agent may not do. The slower model has already
// left the live path, so it is free to take its time.
package sandbox

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ToolName is what the model asks for when it wants to run something.
const ToolName = "run_code"

// MaxFiles is how many files one run may hand back.
const MaxFiles = 4

// MaxFileBytes is the largest file a run may hand back.
const MaxFileBytes = 20 << 20

// MaxTimeout is the longest one run of code may be allowed.
const MaxTimeout = 30 * time.Minute

// Result is what running a piece of code produced.
type Result struct {
	// Output is everything the code printed. A sandbox runs it the way a shell would, so
	// what went to output and what went to errors arrive interleaved, as a person reading
	// a terminal would see them.
	Output string
	// ExitCode is what the process returned. Zero is success.
	ExitCode int
	// Files are the files the code was asked to hand back and did write.
	Files []File
	// Missing are the files it was asked to hand back and did not write, or that were
	// too large to.
	Missing []string
}

// File is something the code wrote that is wanted outside the sandbox.
type File struct {
	// Name is the file's base name, which is what a person is shown.
	Name string
	// MIME is the file's media type.
	MIME string
	Data []byte
}

// Attachment is a file somebody can be shown, once it has been put where they can reach it.
type Attachment struct {
	Name string `json:"name"`
	MIME string `json:"mime_type"`
	URL  string `json:"url"`
	Size int    `json:"size"`
}

// Publisher puts a file where the person the agent is talking to can see it.
type Publisher func(context.Context, File) (Attachment, error)

// Config is how a sandbox is built and how long code may run in it. The zero value is the
// provider's own Python sandbox with its default limits.
type Config struct {
	// Image is the container image to start from, which must have Python. Empty in a built
	// sandbox is a slim Python image; empty otherwise is the provider's own sandbox.
	Image string `json:"image,omitempty"`
	// Setup are shell commands run once on top of the image when it is built.
	Setup []string `json:"setup,omitempty"`
	// TimeoutMs bounds one run of code. Zero is the provider's default.
	TimeoutMs int `json:"timeout_ms,omitempty"`
	// CPU, MemoryGB and DiskGB size the sandbox. Zero leaves the provider's default. The
	// provider's own sandbox cannot be resized, so setting any of them builds an image.
	CPU      int `json:"cpu,omitempty"`
	MemoryGB int `json:"memory_gb,omitempty"`
	DiskGB   int `json:"disk_gb,omitempty"`
	// Network lets code in the sandbox reach the internet. Without it, and without
	// AllowedDomains or NetworkAllowList, every outbound connection is blocked.
	Network bool `json:"network,omitempty"`
	// AllowedDomains are the domains code in the sandbox may reach. Set with Network, it
	// narrows what Network opens.
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	// NetworkAllowList are the CIDR ranges code in the sandbox may reach, narrowing
	// Network the same way. Daytona takes one allow list or the other, never both.
	NetworkAllowList []string `json:"network_allow_list,omitempty"`
}

// Isolated reports whether code in the sandbox may reach nothing outside it.
func (c Config) Isolated() bool {
	return !c.Network && len(c.AllowedDomains) == 0 && len(c.NetworkAllowList) == 0
}

// Built reports whether the sandbox needs an image built for it.
func (c Config) Built() bool {
	return c.Image != "" || len(c.Setup) > 0 || c.CPU > 0 || c.MemoryGB > 0 || c.DiskGB > 0
}

// Timeout is how long one run of code may take, or zero for the provider's default.
func (c Config) Timeout() time.Duration {
	return min(time.Duration(c.TimeoutMs)*time.Millisecond, MaxTimeout)
}

// Sandbox runs code somewhere isolated from everything that matters.
type Sandbox interface {
	// Run executes a piece of Python and returns what it printed, and the files named in
	// outputs that it wrote. An error means the code could not be run at all; code that
	// ran and failed is a Result with a non-zero exit.
	Run(ctx context.Context, code string, outputs []string) (Result, error)
	// Write puts a file into the sandbox at a path relative to where code runs, making the
	// directories it sits in.
	Write(ctx context.Context, path string, data []byte) error
	// Close releases whatever was held open to run code. Safe to call twice.
	Close() error
}

// Granted is a sandbox whose code may call connectors' APIs through grants.
type Granted interface {
	Grants() []Grant
	// Basic is its broker's CredentialBroker.Basic: whether git over HTTPS may send a grant's
	// placeholder.
	Basic() bool
}

// Tool describes running code in box to a model, with the env vars its grants give code.
func Tool(box Sandbox) llm.Tool {
	description := "Run a short Python program and read back what it printed. Use it for anything you would otherwise have to work out in your head, such as arithmetic, dates or parsing, and for making files such as images. Print what you want to know. A skill's own scripts are under skills/<name>/; run one in another language through subprocess, such as subprocess.run([\"go\", \"run\", \"skills/pdf/scripts/merge.go\"])."
	if granted, ok := box.(Granted); ok {
		for _, grant := range granted.Grants() {
			hosts := strings.Join(grant.Hosts, ", ")
			description += fmt.Sprintf(" Code may call %s with $%s: send it over HTTPS as `Authorization: Bearer $%s`; "+
				"the value is a placeholder that works only for %s, only in a request header.", hosts, grant.Env, grant.Env, hosts)
			// The user name is GitHub's for a token
			// (https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation).
			if granted.Basic() {
				description += fmt.Sprintf(" git over HTTPS takes it as the password in the remote's URL, such as "+
					"https://x-access-token:$%s@<host>/<owner>/<repo>.git.", grant.Env)
			} else {
				description += fmt.Sprintf(" git over HTTPS cannot use $%s: git sends it Base64-encoded, which only "+
					"the router's code_sandbox.broker: proxy replaces.", grant.Env)
			}
		}
	}
	return llm.Tool{
		Name:        ToolName,
		Description: description,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"code": map[string]any{
					"type":        "string",
					"description": "The Python to run. Print the answer; nothing else comes back.",
				},
				"files": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"maxItems":    MaxFiles,
					"description": "Absolute paths of files the program writes that the person should get, such as an image it rendered. They are attached to the reply they see.",
				},
			},
			"required": []string{"code"},
		},
	}
}
