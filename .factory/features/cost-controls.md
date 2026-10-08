
# Cost controls

## API keys

You should be able to set an expiration on an API key
In addition you should be able to set a spend limit at some defined interval (hourly, daily, weekly, monthly)

This ensures that if you lose a development API key you don't immediately get hit with crazy fees.

Always prefix keys with stream_sk_123123

## Credits

The config system should indicate if we're running in either hosted or single-tenant mode
When in hosted mode we should require that the customers purchases credits.

Upgrading credits by $1k can be done once a day at most.
For spend higher than that you have to validate your identity. 
For spend beyond $1k maybe we should require bank payments.

Consider 3d secure. 

## Auto recharge

Should be supported as an opt-in

## Validating of an API call/spend is allowed

- App is active, not blocked
- API key is still active
- API key spend limit isn't reached
- App spend limit isn't reached
- App still has at least $10 of credits

## Policies

Budget policy per app. With a spend cap set to some interval (hourly, daily, weekly, monthly)
Training & Data retention policy.
Prompt injection prevention. Enabled/disabled

