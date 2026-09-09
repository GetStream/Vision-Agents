package stream

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// routerFiles are the extensions a router config directory is read from.
var routerFiles = []string{".yaml", ".yml"}

// RouterOptions is the routing options one stored config holds.
//
// It is the shape of a config file as much as an argument: SyncRouters reads one of these
// per file, and DefineRouter takes one directly, so a config written by hand and one
// written in YAML are the same thing said twice.
type RouterOptions struct {
	// Name is what the config is called, which is also how Router{Config: name} finds it.
	// In a file it defaults to the file's own name.
	Name string `json:"name,omitempty"`
	// Tags are cost labels carried onto everything routed under it.
	Tags map[string]string `json:"tags,omitempty"`
	// STT, TTS, LLM and Search are how it transcribes, speaks, answers and looks things
	// up. Each is a default that a per-call option overrides.
	STT    *acceleration.SttOptions    `json:"stt,omitempty"`
	TTS    *acceleration.TtsOptions    `json:"tts,omitempty"`
	LLM    *acceleration.LlmOptions    `json:"llm,omitempty"`
	Search *acceleration.SearchOptions `json:"search,omitempty"`
}

// DefineRouter stores the routing options a Router can then be named after.
//
// A config is what makes "this is the healthcare setup" something said once rather than a
// set of options repeated at every call site. It is written by name, so calling this twice
// edits what is stored rather than storing a second copy of it.
//
//	_, err := stream.DefineRouter(ctx, stream.Backend{}, stream.RouterOptions{
//	    Name: "healthcare",
//	    STT: &acceleration.SttOptions{
//	        Providers:  &[]string{"deepgram", "parakeet"},
//	        DataPolicy: &acceleration.DataPolicy{AllowTraining: &no},
//	    },
//	})
func DefineRouter(
	ctx context.Context,
	backend Backend,
	wanted RouterOptions,
) (*acceleration.RouterConfig, error) {
	if strings.TrimSpace(wanted.Name) == "" {
		return nil, errors.New("stream: a router config needs a name")
	}

	client, err := backend.Client()
	if err != nil {
		return nil, err
	}

	stored, err := namedRouterConfig(ctx, client, wanted.Name)
	if err != nil {
		return nil, err
	}
	return storeRouterConfig(ctx, client, wanted.request(), stored)
}

// ConfigureSTT stores how this router transcribes.
//
// Router{Config: "healthcare"} reads a config; this writes one, so setting a router up and
// routing through it are the same value rather than two names for it. The other three
// modalities are left as they were stored, since saying how something is heard is not a
// statement about how it speaks.
func (r Router) ConfigureSTT(
	ctx context.Context,
	options *acceleration.SttOptions,
) (*acceleration.RouterConfig, error) {
	if r.Config == "" {
		return nil, errors.New(
			`stream: ConfigureSTT writes a named config, so the router needs one: Router{Config: "healthcare"}`)
	}

	client, err := r.client()
	if err != nil {
		return nil, err
	}

	stored, err := namedRouterConfig(ctx, client, r.Config)
	if err != nil {
		return nil, err
	}

	wanted := acceleration.RouterConfigRequest{Name: r.Config, Stt: options}
	// Carried forward rather than restated: a config is one row, and writing the speech
	// half of it should not silently drop the voice half.
	if stored != nil {
		wanted.Tts, wanted.Llm, wanted.Search = stored.Tts, stored.Llm, stored.Search
		wanted.Tags = stored.Tags
	}
	r.label(&wanted.Tags)

	return storeRouterConfig(ctx, client, wanted, stored)
}

// ConfigureTTS stores how this router speaks.
//
// The voice half of ConfigureSTT, on the same terms: the other three modalities are left
// as they were stored. A voice named "custom:receptionist" is one of the customer's own
// and nothing else, where a bare name falls back to the provider's library.
func (r Router) ConfigureTTS(
	ctx context.Context,
	options *acceleration.TtsOptions,
) (*acceleration.RouterConfig, error) {
	if r.Config == "" {
		return nil, errors.New(
			`stream: ConfigureTTS writes a named config, so the router needs one: Router{Config: "healthcare"}`)
	}

	client, err := r.client()
	if err != nil {
		return nil, err
	}

	stored, err := namedRouterConfig(ctx, client, r.Config)
	if err != nil {
		return nil, err
	}

	wanted := acceleration.RouterConfigRequest{Name: r.Config, Tts: options}
	if stored != nil {
		wanted.Stt, wanted.Llm, wanted.Search = stored.Stt, stored.Llm, stored.Search
		wanted.Tags = stored.Tags
	}
	r.label(&wanted.Tags)

	return storeRouterConfig(ctx, client, wanted, stored)
}

// SyncRouters stores every router config a directory of YAML files describes.
//
// The same bargain as an agent directory, for routing: a config that lives in the
// repository is one that can be reviewed, and one written by hand at a call site is not.
// Each file is one config, named by its own name key or by its filename, and each is
// written by name, so running this twice edits rather than duplicates.
//
//	# routers/healthcare.yaml
//	tags:
//	  team: clinical
//	stt:
//	  providers: [deepgram, parakeet]
//	  data_policy:
//	    allow_training: false
//	    retention: none
func SyncRouters(
	ctx context.Context,
	backend Backend,
	directory string,
) ([]*acceleration.RouterConfig, error) {
	paths, err := routerConfigFiles(directory)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("stream: %s holds no .yaml router configs", directory)
	}

	stored := make([]*acceleration.RouterConfig, 0, len(paths))
	for _, path := range paths {
		wanted, err := readRouterConfig(path)
		if err != nil {
			return nil, err
		}
		written, err := DefineRouter(ctx, backend, wanted)
		if err != nil {
			return nil, err
		}
		stored = append(stored, written)
	}
	return stored, nil
}

// request is these options as the router takes them.
func (o RouterOptions) request() acceleration.RouterConfigRequest {
	wanted := acceleration.RouterConfigRequest{
		Name:   strings.TrimSpace(o.Name),
		Stt:    o.STT,
		Tts:    o.TTS,
		Llm:    o.LLM,
		Search: o.Search,
	}
	if len(o.Tags) > 0 {
		tags := o.Tags
		wanted.Tags = &tags
	}
	return wanted
}

// routerConfigFiles is the YAML in a directory, in a fixed order so two runs of a sync do
// the same thing in the same sequence.
func routerConfigFiles(directory string) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, fmt.Errorf("stream: reading %s: %w", directory, err)
	}

	var paths []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !slices.Contains(routerFiles, strings.ToLower(filepath.Ext(entry.Name()))) {
			continue
		}
		paths = append(paths, filepath.Join(directory, entry.Name()))
	}
	slices.Sort(paths)
	return paths, nil
}

// readRouterConfig reads one config file.
//
// The YAML is turned into JSON and decoded through the generated types' own tags, rather
// than given yaml tags of its own: the field names in a file are then the field names on
// the wire by construction, and cannot drift from them. Unknown fields are refused, so a
// misspelt option is reported instead of storing a config that routes to whatever the
// fallback happens to be.
func readRouterConfig(path string) (RouterOptions, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return RouterOptions{}, fmt.Errorf("stream: reading %s: %w", path, err)
	}

	var described any
	if err := yaml.Unmarshal(raw, &described); err != nil {
		return RouterOptions{}, fmt.Errorf("stream: %s is not valid YAML: %w", path, err)
	}
	if described == nil {
		return RouterOptions{}, fmt.Errorf("stream: %s describes no config", path)
	}

	encoded, err := json.Marshal(described)
	if err != nil {
		return RouterOptions{}, fmt.Errorf("stream: %s: %w", path, err)
	}

	var wanted RouterOptions
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wanted); err != nil {
		return RouterOptions{}, fmt.Errorf("stream: %s: %w", path, err)
	}
	if wanted.Name == "" {
		name := filepath.Base(path)
		wanted.Name = strings.TrimSuffix(name, filepath.Ext(name))
	}
	return wanted, nil
}

// namedRouterConfig is the stored config of that name, or nil when there is none.
func namedRouterConfig(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	name string,
) (*acceleration.RouterConfig, error) {
	listed, err := client.ListRouterConfigsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("stream: listing router configs: %w", err)
	}
	if listed.JSON200 == nil {
		return nil, refusal(listed.Status(), listed.JSON400, listed.JSON401)
	}

	for _, config := range *listed.JSON200 {
		if config.Name == name {
			return &config, nil
		}
	}
	return nil, nil
}

// storeRouterConfig writes a config, editing the one of that name rather than adding a
// second.
func storeRouterConfig(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	wanted acceleration.RouterConfigRequest,
	stored *acceleration.RouterConfig,
) (*acceleration.RouterConfig, error) {
	if stored != nil {
		updated, err := client.UpdateRouterConfigWithResponse(ctx, stored.Id, wanted)
		if err != nil {
			return nil, fmt.Errorf("stream: updating router config %s: %w", stored.Id, err)
		}
		if updated.JSON200 == nil {
			return nil, refusal(updated.Status(), updated.JSON400, updated.JSON401, updated.JSON404)
		}
		return updated.JSON200, nil
	}

	created, err := client.CreateRouterConfigWithResponse(ctx, wanted)
	if err != nil {
		return nil, fmt.Errorf("stream: creating router config %s: %w", wanted.Name, err)
	}
	if created.JSON201 == nil {
		return nil, refusal(created.Status(), created.JSON400, created.JSON401)
	}
	return created.JSON201, nil
}
