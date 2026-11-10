package main

import (
	"context"

	getstream "github.com/GetStream/getstream-go/v5"
)

// crewRoleField is the Stream user custom data that holds a person's crew role.
const crewRoleField = "crew_role"

// roles is where each person's crew role is kept, which decides who reads the leaders' inbox.
type roles interface {
	// publish writes each person's name and crew role.
	publish(ctx context.Context, people []Person) error
	// lookup is the user's name and crew role, and whether there is such a user.
	lookup(ctx context.Context, userID string) (name, role string, found bool, err error)
}

// streamRoles keeps the crew roles in the Stream users' custom data.
type streamRoles struct {
	client *getstream.Stream
}

// mint signs a Stream token for userID that lasts tokenValidity.
func (s *streamRoles) mint(userID string) (string, error) {
	return s.client.CreateToken(userID, getstream.WithExpiration(tokenValidity))
}

func newStreamRoles(apiKey, apiSecret string) (*streamRoles, error) {
	client, err := getstream.NewClient(apiKey, apiSecret)
	if err != nil {
		return nil, err
	}
	return &streamRoles{client: client}, nil
}

// publish sets the crew role on users Stream already has, keeping the rest of what they hold,
// and creates the ones it doesn't.
func (s *streamRoles) publish(ctx context.Context, people []Person) error {
	created := map[string]getstream.UserRequest{}
	for _, p := range people {
		_, _, found, err := s.lookup(ctx, p.ID)
		if err != nil {
			return err
		}
		if found {
			if _, err := s.client.UpdateUsersPartial(ctx, &getstream.UpdateUsersPartialRequest{
				Users: []getstream.UpdateUserPartialRequest{{ID: p.ID, Set: map[string]any{crewRoleField: p.CrewRole}}},
			}); err != nil {
				return err
			}
			continue
		}
		name := p.Name
		created[p.ID] = getstream.UserRequest{ID: p.ID, Name: &name, Custom: map[string]any{crewRoleField: p.CrewRole}}
	}
	if len(created) == 0 {
		return nil
	}
	_, err := s.client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: created})
	return err
}

func (s *streamRoles) lookup(ctx context.Context, userID string) (string, string, bool, error) {
	queried, err := s.client.QueryUsers(ctx, &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{
		FilterConditions: map[string]any{"id": map[string]any{"$eq": userID}},
	}})
	if err != nil {
		return "", "", false, err
	}
	if len(queried.Data.Users) == 0 {
		return "", "", false, nil
	}
	user := queried.Data.Users[0]
	name := ""
	if user.Name != nil {
		name = *user.Name
	}
	role, _ := user.Custom[crewRoleField].(string)
	return name, role, true, nil
}
