
# Auth

Lets add the following auth modes

- noauth (assumed you run a proxy or otherwise limit traffic to the go backend)
- api_key (api keys + secret using JWT tokens)
- 

## API key

In open source mode add storage for API keys + secrets.
API keys are attached to apps which are attached to organizations.

Validate those API keys + secrets for all users who connect on the API

Follow best practices written down in .factor/features/auth.md

# Hosting for acceleration

# Chart name

accelerate

## CI/CD

For this repo, setup CI/CD to compile the Go binary into an image. similar to how we do this on chat. see workspace/chat and check the CI/CD setup

## K8

In workspace/chat see the infra folder. Add a helm chart so we can run accelerate as a chart.
For now we only want to enable it with 1 pod in the us-east4 region.
This is enabled/disabled at the region, not the shard level

## Best practices

Follow security and normal best practices similar to the chat api

## Deploy with s3

On this branch we experimented with an s3 based deploy flow on k8
https://github.com/GetStream/chat/pull/15071

Don't merge that, but use the s3 + shiply deploy flow (as seen on that branch)
only for the accelerate deploys. 

shiply can be found in workspace/stream-infra (but be sure to update shiply with the changes from master, this is an outdated branch)

## Verification

Tag a version on ci/cd for this acceleration branch
Wait for CI/CD to compile
Iterate if that doesn't work

After that show me step by step how to deploy it to us-east4

