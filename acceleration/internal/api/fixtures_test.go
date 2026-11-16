//go:build integration

package api

import "sync"

// fixture is data loaded once and shared by every test that requires it. Tests read it and
// never change it: opening a session in the fixture's app is fine, renaming the app is not.
type fixture struct {
	testApp
	// userID is an end user of the app.
	userID string
}

// fixtureLoaders are the fixtures there are, by name.
var fixtureLoaders = map[string]func(testData) fixture{
	// standard is an organization, an app with an API key, and a user of the app.
	"standard": func(data testData) fixture {
		return fixture{testApp: data.createApp(), userID: data.suite.utils.uuid()}
	},
}

// loadedFixtures holds every fixture a test has required, for the rest of the run.
var loadedFixtures = struct {
	sync.Mutex
	byName map[string]fixture
}{byName: map[string]fixture{}}

// requireFixture returns the fixture called name, loading it if no test has yet.
func (s *RouterSuite) requireFixture(name string) fixture {
	loadedFixtures.Lock()
	defer loadedFixtures.Unlock()
	if loaded, ok := loadedFixtures.byName[name]; ok {
		return loaded
	}
	load, ok := fixtureLoaders[name]
	s.Require().True(ok, "there is no fixture called %q", name)
	loaded := load(s.data)
	loadedFixtures.byName[name] = loaded
	return loaded
}
