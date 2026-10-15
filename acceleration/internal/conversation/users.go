package conversation

import (
	"context"
	"maps"
	"slices"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

// knownFor is how long a user Chat was seen to have is taken on trust. Every session asks
// about its agent's user, which is the same one each time, and Stream rate limits the query
// per app: hundreds of conversations opening at once are refused on it. Short enough that a
// user deleted from the app is recreated within minutes.
const knownFor = 10 * time.Minute

// maxKnown bounds the users remembered, across every app the deployment writes to.
const maxKnown = 100_000

// knownUsers are the users Chat has, keyed by app and user id, with when that was learned.
var knownUsers = struct {
	sync.Mutex
	at map[string]time.Time
}{at: map[string]time.Time{}}

// CreateMissingUsers creates the users Chat does not have yet and leaves the rest alone.
//
// Chat's upsert replaces a user whole: writing an id with no name or image takes away
// whatever the app had given that person. In an app whose users are real people, that is
// the router quietly rewriting somebody's profile every time it writes near them, so a
// user who is already there is never written.
func CreateMissingUsers(ctx context.Context, client *getstream.Stream, users map[string]getstream.UserRequest) error {
	app := client.BaseUrl() + "\x00" + client.ApiKey()
	unknown := unknownUsers(app, users)
	if len(unknown) == 0 {
		return nil
	}
	ids := slices.Sorted(maps.Keys(unknown))
	limit := len(ids)
	found, err := client.QueryUsers(ctx, &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{
		FilterConditions: map[string]any{"id": map[string]any{"$in": ids}},
		Limit:            &limit,
	}})
	if err != nil {
		return err
	}
	missing := maps.Clone(unknown)
	for _, user := range found.Data.Users {
		delete(missing, user.ID)
	}
	if len(missing) > 0 {
		if _, err := client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: missing}); err != nil {
			return err
		}
	}
	remember(app, ids)
	return nil
}

// unknownUsers is the part of users not known to be in Chat already.
func unknownUsers(app string, users map[string]getstream.UserRequest) map[string]getstream.UserRequest {
	knownUsers.Lock()
	defer knownUsers.Unlock()
	unknown := map[string]getstream.UserRequest{}
	for id, user := range users {
		if at, ok := knownUsers.at[app+"\x00"+id]; !ok || time.Since(at) > knownFor {
			unknown[id] = user
		}
	}
	return unknown
}

func remember(app string, ids []string) {
	knownUsers.Lock()
	defer knownUsers.Unlock()
	if len(knownUsers.at)+len(ids) > maxKnown {
		clear(knownUsers.at)
	}
	now := time.Now()
	for _, id := range ids {
		knownUsers.at[app+"\x00"+id] = now
	}
}
