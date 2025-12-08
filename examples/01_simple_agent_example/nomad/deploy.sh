#!/bin/bash
set -e

# Get Nomad server IP from Terraform
NOMAD_ADDR=$(cd ../terraform && terraform output -raw nomad_ui_url)
ARTIFACT_REGISTRY=$(cd ../terraform && terraform output -raw artifact_registry)

export NOMAD_ADDR

echo "Nomad address: $NOMAD_ADDR"

# Set secrets as Nomad variables (run once or when secrets change)
if [ "$1" = "secrets" ]; then
  echo "Setting Nomad variables..."
  
  # Read from .env file or environment
  source ../.env 2>/dev/null || true
  
  nomad var put nomad/jobs/vision-agent \
    deepgram_api_key="$DEEPGRAM_API_KEY" \
    elevenlabs_api_key="$ELEVENLABS_API_KEY" \
    anthropic_api_key="$ANTHROPIC_API_KEY" \
    openai_api_key="$OPENAI_API_KEY" \
    stream_api_key="$STREAM_API_KEY" \
    stream_api_secret="$STREAM_API_SECRET"
  
  echo "Secrets stored in Nomad"
  exit 0
fi

# Deploy the job
echo "Deploying vision-agent job..."

# Substitute the artifact registry URL
sed "s|\${artifact_registry}|$ARTIFACT_REGISTRY|g" agent.nomad.hcl > /tmp/agent.nomad.hcl

nomad job run /tmp/agent.nomad.hcl

echo "Done! Check status: nomad job status vision-agent"


