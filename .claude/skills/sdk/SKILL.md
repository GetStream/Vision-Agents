---
name: sdk
description: How to build an SDK for the acceleration backend
---

* we use openAPI, so generate your SDK from the openAPI spec
* some endpoints are server side only. such as configuring agents, or listening to agent dispatch

## SDK best practices

Do deep research on SDK best practices. Use OpenAI sol for this using the tokens available in .env
Based on this deep research create an sdk-mylanguage skill in this repo

## Folder sync structure

* Use the same agent folder sync structure the python SDK uses (see examples folder)

## Where to place the SDK

sdks/mylanguage

## Client side SDKs

* Expose nice stateflow in Kotlin, or your language equivalent so it's easy to customize
* Use the modern UI frameworks (compose or swiftUI)

We want to expose 3 different SDKs client side

* ai-language-core (state layer and APIs only)
* ai-language-ui (ui components)
* ai-language-rtc (add video and voice capabilities which are relatively large)

Include Stream's chat and voice SDKs as dependencies

## Server side SDKs

* for server side we just provide 1 sdk
* show your intended plan for the key operations:
* starting an agent for an inbound call, text message, whatsapp message, slack message etc
* placing an outbound call
* syncing the folder config for an agent

Include Stream's server side SDK as a dependency. 
