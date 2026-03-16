# Kubernetes Deploy Example

For the full deployment guide (cluster setup, Helm deployment, secrets, monitoring), see the
[Kubernetes Deployment Guide](https://visionagents.ai/guides/kubernetes-deployment).

## Quick start

Copy the .env.example and fill in the required variables:

```
cp .env.example .env
```

Run the example locally to verify everything works:

```
uv run deploy_example.py run
```

## Build the Docker image

There are two Dockerfiles:

- `Dockerfile` - CPU version (python:3.13-slim, ~150MB)
- `Dockerfile.gpu` - GPU version (pytorch:2.9.1-cuda12.8, ~8GB)

### CPU build

```
cd examples/07_k8s_deploy_example
docker buildx build --platform linux/amd64 -t vision-agent-deploy .
```

### GPU build (takes a long time)

```
cd examples/07_k8s_deploy_example
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t vision-agent-deploy:gpu .
```

**Tip:** Building amd64 on Apple Silicon is slow due to emulation. Consider using CI for production builds.

## Push to registry

Tag and push to your container registry:

```
# CPU
docker tag vision-agent-deploy <your-registry>/vision-agent-deploy:latest
docker push <your-registry>/vision-agent-deploy:latest

# GPU
docker tag vision-agent-deploy:gpu <your-registry>/vision-agent-deploy:gpu
docker push <your-registry>/vision-agent-deploy:gpu
```
