
Agents need to interact with common tools such as

- Slack/Teams
- CalAI/Calendly/Salesforce/Hubspot/ Google calendar
- Github/Sentry/Linear
- Google drive/docs
- Blender
- Whatsapp, RCS, SMS, iMessage

Typically this is exposed through MCP or CLI

## MCP auth

For MCP auth we need to support 3 types of auth

- No auth/local/public: Blender
- Auth once per app: Add an agent to your slack
- Auth once per user of an app: Post on slack for you

## Omni channel

Messages sent through slack, whatsapp, RCS, SMS and iMessage still need to be stored in the session/channel

So we need to standardize the structure. 

## Permissions

Typically since the user is authenticated, the app can handle permissions...

## Sprint 1

Store all users instead of only storing guest users.
Add a client side redis caching with a LRU based approach to keeping the last 10k users in memory cache. (Split the cache per app. Dont store more than 10k total)


## Sprint 2

Give an agent access to read my tickets on sentry at the company level. (Company level auth)
Also give it access to read my google calendar. (user level auth)

Lets create a next text agent example for this in examples
The setup for the MCP should be done in agent.yaml

The agent call/dashboard page should show a reminder to complete oauth setup for the company level auth. (sentry)
The auth for google calendar should be shown in the chat when the AI needs it, we should handle this as a custom attachment

## Sprint 3

Add a new example app with Blender MCP support. The user can ask for a 3d render and you make it for them
This uses blender + a sandbox started on daytona through the existing sandbox support

## Sprint 4

Add a new omni package, which standardizes messages from these: slack, whatsapp, RCS, SMS and iMessage (through Linq)
To the proper getstream.io message format. Use attachments and custom data to support this nicely

## Tune the syntax before expanding to all and adding docs

