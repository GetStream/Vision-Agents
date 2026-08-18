package streamedge

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

const demoBaseURLEnvVar = "EXAMPLE_BASE_URL"

// defaultDemoBaseURL is Stream's hosted video demo, which is the same page the Python
// examples open when they join a call.
const defaultDemoBaseURL = "https://getstream.io/video/demos"

// demoTokenValidity is how long the browser's token lasts. A demo call outliving it is a
// call nobody is still on.
const demoTokenValidity = time.Hour

// DemoURL is a link a person can open to join this call from a browser and talk to the
// agent.
//
// The token is minted here rather than fetched, so this makes no network calls: the
// coordinator registers the browser as a user when it connects, the same way it does for
// the agent. EXAMPLE_BASE_URL points the link at another deployment of the demo.
func (e *Edge) DemoURL(user User) (string, error) {
	if user.ID == "" {
		return "", errors.New("streamedge: a demo user id is required")
	}
	if e.options.APISecret == "" {
		return "", fmt.Errorf("streamedge: %s is required to mint a demo token", apiSecretEnvVar)
	}
	if user.Name == "" {
		user.Name = user.ID
	}

	client, err := getstream.NewClient(e.options.APIKey, e.options.APISecret)
	if err != nil {
		return "", fmt.Errorf("streamedge: demo client: %w", err)
	}
	token, err := client.CreateToken(user.ID, getstream.WithExpiration(demoTokenValidity))
	if err != nil {
		return "", fmt.Errorf("streamedge: mint a demo token: %w", err)
	}

	base := os.Getenv(demoBaseURLEnvVar)
	if base == "" {
		base = defaultDemoBaseURL
	}
	query := url.Values{
		"api_key":    {e.options.APIKey},
		"token":      {token},
		"skip_lobby": {"true"},
		"user_name":  {user.Name},
	}
	return fmt.Sprintf("%s/join/%s?%s",
		strings.TrimSuffix(base, "/"), url.PathEscape(e.options.CallID), query.Encode()), nil
}
