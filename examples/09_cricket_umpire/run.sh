#!/bin/bash

python token_server.py &
sleep 2

python cricket_umpire.py run --call-id cricket-demo-1 &
sleep 2

cd frontend && npm run dev 