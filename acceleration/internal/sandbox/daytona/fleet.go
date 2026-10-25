package daytona

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	sdkerrors "github.com/daytona/clients/sdk-go/pkg/errors"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// organizationEnvVar names the Daytona organization whose usage is read. The analytics API
// takes it in every path, and the API key alone does not say it (Daytona's current-key answer
// names no organization), so without it the organization is read off any sandbox the key can
// list.
const organizationEnvVar = "DAYTONA_ORGANIZATION_ID"

// defaultAPIURL is Daytona's own API, as the SDK reaches it without DAYTONA_API_URL.
const defaultAPIURL = "https://app.daytona.io/api"

// readTimeout bounds one request of a fleet's.
const readTimeout = 30 * time.Second

// Fleet is every sandbox of the Daytona organization a key reaches, for the jobs that look
// after them all: the sweep of sandboxes no session needs and the reconciliation of what was
// billed with Daytona's records. Nothing it does is billed.
type Fleet struct {
	client       *sdk.Client
	apiKey       string
	apiURL       string
	organization string
	http         *http.Client
}

// FleetOptions reach Daytona. Empty are DAYTONA_API_KEY, DAYTONA_API_URL or Daytona's own API,
// and DAYTONA_ORGANIZATION_ID.
type FleetOptions struct {
	APIKey, APIURL, Organization string
	HTTPClient                   *http.Client
}

// NewFleet reaches the organization options name.
func NewFleet(o FleetOptions) (*Fleet, error) {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APIKey == "" {
		return nil, stack.Wrap(errors.New("daytona: DAYTONA_API_KEY is required"))
	}
	if o.APIURL == "" {
		o.APIURL = os.Getenv("DAYTONA_API_URL")
	}
	if o.APIURL == "" {
		o.APIURL = defaultAPIURL
	}
	if o.Organization == "" {
		o.Organization = os.Getenv(organizationEnvVar)
	}
	if o.HTTPClient == nil {
		o.HTTPClient = &http.Client{Timeout: readTimeout}
	}
	client, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Fleet{client: client, apiKey: o.APIKey, apiURL: strings.TrimRight(o.APIURL, "/"), organization: o.Organization,
		http: o.HTTPClient}, nil
}

// Close lets the fleet's client go.
func (f *Fleet) Close(ctx context.Context) error { return f.client.Close(ctx) }

// Listed is a sandbox as Daytona keeps it.
type Listed struct {
	Reserved
	State     string
	Labels    map[string]string
	CreatedAt time.Time
}

func listed(box *sdk.Sandbox) Listed {
	found := Listed{Reserved: reserved(box), State: string(box.State), Labels: box.Labels}
	if box.CreatedAt != nil {
		found.CreatedAt, _ = time.Parse(time.RFC3339Nano, *box.CreatedAt)
	}
	return found
}

// Labelled is every sandbox carrying each of labels, the deleted ones left out.
func (f *Fleet) Labelled(ctx context.Context, labels map[string]string) ([]Listed, error) {
	var found []Listed
	for box, err := range f.client.ListSeq(ctx, &sdk.ListSandboxesQuery{Labels: labels}) {
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("daytona: list sandboxes: %w", err))
		}
		found = append(found, listed(box))
		if f.organization == "" {
			f.organization = box.OrganizationId
		}
	}
	return found, nil
}

// Sandbox is one sandbox as Daytona keeps it, and false once Daytona no longer knows it.
func (f *Fleet) Sandbox(ctx context.Context, id string) (Listed, bool, error) {
	box, found, err := f.get(ctx, id)
	if err != nil || !found {
		return Listed{}, false, err
	}
	if f.organization == "" {
		f.organization = box.OrganizationId
	}
	return listed(box), true, nil
}

// Delete deletes a sandbox. One Daytona no longer has counts as deleted.
func (f *Fleet) Delete(ctx context.Context, id string) error {
	box, found, err := f.get(ctx, id)
	if err != nil || !found {
		return err
	}
	if err := box.Delete(ctx); err != nil && !errors.Is(err, sdkerrors.ErrNotFound) {
		return stack.Wrap(fmt.Errorf("daytona: delete sandbox %s: %w", id, err))
	}
	return nil
}

func (f *Fleet) get(ctx context.Context, id string) (*sdk.Sandbox, bool, error) {
	box, err := f.client.Get(ctx, id)
	if errors.Is(err, sdkerrors.ErrNotFound) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, stack.Wrap(fmt.Errorf("daytona: read sandbox %s: %w", id, err))
	}
	return box, true, nil
}

// SandboxUsage is what Daytona's records say one sandbox used over a window: vCPU, memory and
// disk seconds, the last two in GiB. Daytona's records lag its billing by up to 48 hours.
type SandboxUsage struct {
	SandboxID     string  `json:"sandboxId"`
	CPUSeconds    float64 `json:"totalCPUSeconds"`
	RAMGBSeconds  float64 `json:"totalRAMGBSeconds"`
	DiskGBSeconds float64 `json:"totalDiskGBSeconds"`
	FirstStart    string  `json:"firstStart"`
	LastEnd       string  `json:"lastEnd"`
}

// UsagePeriod is a stretch of one sandbox's life at one size: its vCPUs, and its memory and
// disk in GiB. A stopped stretch holds disk alone.
type UsagePeriod struct {
	StartAt string  `json:"startAt"`
	EndAt   string  `json:"endAt"`
	CPU     float64 `json:"cpu"`
	RAMGB   float64 `json:"ramGB"`
	DiskGB  float64 `json:"diskGB"`
}

// Usage is what Daytona's records say each of the organization's sandboxes used from from to
// to.
func (f *Fleet) Usage(ctx context.Context, from, to time.Time) ([]SandboxUsage, error) {
	var usage []SandboxUsage
	err := f.analytics(ctx, "usage/sandbox", from, to, &usage)
	return usage, err
}

// Periods is what Daytona's records say one sandbox used from from to to, a stretch at a time.
func (f *Fleet) Periods(ctx context.Context, id string, from, to time.Time) ([]UsagePeriod, error) {
	var periods []UsagePeriod
	err := f.analytics(ctx, "sandbox/"+url.PathEscape(id)+"/usage", from, to, &periods)
	return periods, err
}

// analytics reads one of the analytics API's reports on the organization into answer.
func (f *Fleet) analytics(ctx context.Context, report string, from, to time.Time, answer any) error {
	base, err := f.analyticsURL(ctx)
	if err != nil {
		return err
	}
	if f.organization == "" {
		if _, err := f.Labelled(ctx, nil); err != nil {
			return err
		}
	}
	if f.organization == "" {
		return stack.Wrap(errors.New("daytona: which organization's usage to read is unknown: set " + organizationEnvVar))
	}
	query := url.Values{"from": {from.UTC().Format(time.RFC3339)}, "to": {to.UTC().Format(time.RFC3339)}}
	address := strings.TrimRight(base, "/") + "/organization/" + url.PathEscape(f.organization) + "/" + report + "?" + query.Encode()
	return f.read(ctx, address, answer)
}

// analyticsURL is where the deployment's analytics API is, as Daytona's config says.
func (f *Fleet) analyticsURL(ctx context.Context) (string, error) {
	var config struct {
		AnalyticsAPIURL string `json:"analyticsApiUrl"`
	}
	if err := f.read(ctx, f.apiURL+"/config", &config); err != nil {
		return "", err
	}
	if config.AnalyticsAPIURL == "" {
		return "", stack.Wrap(errors.New("daytona: this deployment's config names no analytics API"))
	}
	return config.AnalyticsAPIURL, nil
}

func (f *Fleet) read(ctx context.Context, address string, answer any) error {
	ctx, cancel := context.WithTimeout(ctx, readTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return stack.Wrap(err)
	}
	request.Header.Set("Authorization", "Bearer "+f.apiKey)
	response, err := f.http.Do(request)
	if err != nil {
		return stack.Wrap(fmt.Errorf("daytona: %w", err))
	}
	defer response.Body.Close() //nolint:errcheck // read-only
	body, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	if err != nil {
		return stack.Wrap(fmt.Errorf("daytona: %w", err))
	}
	if response.StatusCode != http.StatusOK {
		path := request.URL.Path
		return stack.Wrap(fmt.Errorf("daytona: %s answered %d: %.300s", path, response.StatusCode, body))
	}
	if err := json.Unmarshal(body, answer); err != nil {
		return stack.Wrap(fmt.Errorf("daytona: %s: %w", request.URL.Path, err))
	}
	return nil
}
