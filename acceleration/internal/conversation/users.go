package conversation

import (
	"context"
	"maps"
	"slices"

	getstream "github.com/GetStream/getstream-go/v5"
)

// CreateMissingUsers creates the users Chat does not have yet and leaves the rest alone.
//
// Chat's upsert replaces a user whole: writing an id with no name or image takes away
// whatever the app had given that person. In an app whose users are real people, that is
// the router quietly rewriting somebody's profile every time it writes near them, so a
// user who is already there is never written.
func CreateMissingUsers(ctx context.Context, client *getstream.Stream, users map[string]getstream.UserRequest) error {
	if len(users) == 0 {
		return nil
	}
	ids := slices.Sorted(maps.Keys(users))
	limit := len(ids)
	found, err := client.QueryUsers(ctx, &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{
		FilterConditions: map[string]any{"id": map[string]any{"$in": ids}},
		Limit:            &limit,
	}})
	if err != nil {
		return err
	}
	missing := maps.Clone(users)
	for _, user := range found.Data.Users {
		delete(missing, user.ID)
	}
	if len(missing) == 0 {
		return nil
	}
	_, err = client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: missing})
	return err
}
