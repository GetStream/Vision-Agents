import websockets
import asyncio
import json

# TODO: this was test for websocket connection for events "call." etc.
async def main():
    uri = "wss://video.stream-io-api.com/video/connect?api_key=hd8szvscpxvd&stream-auth-type=jwt&X-Stream-Client=stream-video-react-v1.21.1%7Cclient_bundle%3Dbrowser-esm"
    connection = await websockets.connect(uri, open_timeout=5, user_agent_header="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0")

    #await self._websocket.send(json.dumps(auth_payload))

    auth_payload = {
        #"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NTc2ODYyMzQsInVzZXJfaWQiOiJhZ2VudC1jZmY1YTg3MC1mMDNjLTQxMzAtYjM5My1kMThjNjE3OWU3YTEifQ.gyrAIgz89qLn4EkFNK9c4o-5Oghtnnm_QL-yJHvBE04",
        #"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NTc2ODY1NDgsInVzZXJfaWQiOiJ1c2VyLTc3MGQ4YWJhLWE4ZDQtNGZhYy04NzRlLTYyZGU0Mzg1ODkwMSIsImV4cCI6MTc1NzY5MDE0OH0.D7DIi_Y3THSgzYlO1EHOU5FB5qtzbKF8lO2Zf5oA8iE",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NTc2OTAxNTMsInVzZXJfaWQiOiJ1c2VyLTdlZjZiZWMxLTJkYTYtNDFiZC04MjI2LTA1NDIxNTc4YjA1NSIsImV4cCI6MTc1NzY5Mzc1M30.rMmgx5oV7w2Wn8quOpD5-rYXeoI251e8qWrOWTbhY2Y",
        'user_details': {'id': 'user-7ef6bec1-2da6-41bd-8226-05421578b055'}
    }
    #auth_payload = ""
    print(connection)
    resp = await connection.send(json.dumps(auth_payload))
    print(resp)
    not_sent = True
    while True:
        await asyncio.sleep(1)
        data = await connection.recv()
        if not data:
            continue
        if 'connection.ok' in data:
            client_id = json.loads(data)['connection_id']
        if 'health' in data and not_sent:
            await connection.send(json.dumps({
                'type': 'health.check',
                'client_id': client_id
            }))
            not_sent = False
        print("data", data)


if __name__ == '__main__':
    asyncio.run(main())
