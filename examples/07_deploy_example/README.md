## TODO / improvements

- merge monitoring and HTTP efforts into this

# Tips

* US-east. Services like Stream run a global edge network. But many providers default to US-east. So you typically want
  to run in US-east for optimal latency
* CPU build is quick to get up and running. GPU/CUDA takes hours.
* This guide uses Nebius, but you could do this with other K8 enabled clouds quite easily
* GPU setup needs more checks/testing

# Secrets

First copy the .env.example

```
cp .env.example .env
```

Next fill in the required variables and run the example locally to verify everything works

```
uv run deploy_example.py run
```

# Requirements

[Nebius CLI](https://docs.nebius.com/cli/configure)

```
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
source ~/.zshrc # or similar
nebius version
nebius profile create # for auth
```

[HELM](https://helm.sh/docs/intro/install/)

```
brew install helm
```

# 0. Create a new k8s cluster with Nebius CLI

Lookup your parent-id and subnet for Nebius:

```
nebius vpc subnet list
nebius config list | grep parent-id
```

Create the cluster (replace parent-id and subnet):

```
nebius mk8s cluster create \
  --parent-id mk8scluster-<youridhere> \
  --name vision-agents \
  --control-plane-subnet-id <your-subnet-id> \
  --control-plane-version 1.32 \
  --control-plane-endpoints-public-endpoint
```

## Add a Node Group

Lookup your template service account id

```
nebius iam service-account list
```

Choose **one** of the following:

### Option A: CPU Node (cheaper, for testing)

```
nebius mk8s node-group create \
  --parent-id <cluster-id-from-above> \
  --name cpu \
  --template-resources-platform cpu-d3 \
  --template-resources-preset 4vcpu-16gb \
  --template-boot-disk-size-gibibytes 64 \
  --template-service-account-id <your-service-account-id> \
  --fixed-node-count 1
```

### Option B: GPU Node (H200, if you need to run models locally)

GPUs are expensive. You typically don't want to run your voice agents on a server with a GPU.
Even from a load balancing perspective you typically want to spin the GPU related work into it's own cluster.
That being said, in case you want/need to run on a GPU do the following:

**Important:** You must include `--template-gpu-settings-drivers-preset cuda12` to install NVIDIA drivers!

```
nebius mk8s node-group create \
  --parent-id <cluster-id-from-above> \
  --name gpu \
  --template-resources-platform gpu-h200-sxm \
  --template-resources-preset 1gpu-16vcpu-200gb \
  --template-boot-disk-size-gibibytes 300 \
  --template-service-account-id <your-service-account-id> \
  --template-metadata-labels nebius.com/gpu=true \
  --template-gpu-settings-drivers-preset cuda12 \
  --fixed-node-count 1
```

Available GPU presets:

- `1gpu-16vcpu-200gb` - 1x H200, 16 vCPU, 200GB RAM
- `8gpu-128vcpu-1600gb` - 8x H200, 128 vCPU, 1.6TB RAM

Available driver presets (see [Nebius GPU docs](https://docs.nebius.com/kubernetes/gpu/set-up)):

- `cuda12` - CUDA 12.4 (default)
- `cuda12.4` - CUDA 12.4
- `cuda12.8` - CUDA 12.8

### Get kubectl credentials

```
nebius mk8s cluster get-credentials --id <cluster-id> --external --force
kubectl get nodes  # verify connection
```

# 1. Build the Docker image

There are two Dockerfiles:

- `Dockerfile` - CPU version (python:3.13-slim, ~150MB)
- `Dockerfile.gpu` - GPU version (pytorch:2.9.1-cuda12.8, ~8GB)

### CPU build

```
cd examples/05_deploy_example
docker buildx build --platform linux/amd64 -t vision-agent-deploy .
```

### GPU build (takes a long time)

```
cd examples/05_deploy_example
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t vision-agent-deploy:gpu .
```

**Tip:** Building amd64 on Apple Silicon is slow due to emulation. Consider using CI for production builds.

# 2. Push to registry

```
# Lookup your registry id
nebius registry list

# Tag and push (CPU)
docker tag vision-agent-deploy cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:latest
docker push cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:latest

# Or for GPU
docker tag vision-agent-deploy:gpu cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:gpu
docker push cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:gpu
```

# 3. Deploy with Helm

## CPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy" \
  --set image.tag=latest \
  --set image.pullPolicy=Always \
  --set cache.enabled=true \
  --set gpu.enabled=false \
  --set secrets.existingSecret=vision-agent-env
```

## GPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy" \
  --set image.tag=gpu \
  --set image.pullPolicy=Always \
  --set cache.enabled=true \
  --set gpu.enabled=true \
  --set secrets.existingSecret=vision-agent-env
```

# 4. Create secrets

Create a Kubernetes secret from your `.env` file:

```
kubectl create secret generic vision-agent-env --from-env-file=.env
```

To update secrets:

```
kubectl delete secret vision-agent-env
kubectl create secret generic vision-agent-env --from-env-file=.env
kubectl rollout restart deployment/vision-agent
```

# Other tips

## Watch logs

```
kubectl logs -l app.kubernetes.io/name=vision-agent -f --tail=100
```

## Pause cluster (stop paying for compute)

```
# List node groups
nebius mk8s node-group list --parent-id <cluster-id>

# Scale to 0
nebius mk8s node-group update --id <node-group-id> --fixed-node-count 0 --async

# Check status
nebius mk8s node-group get --id <node-group-id>
```

Resume by setting count back to 1.

## Switch between CPU and GPU

Just change `gpu.enabled` and redeploy:

```
# Switch to GPU
helm upgrade vision-agent ./helm --reuse-values --set gpu.enabled=true

# Switch to CPU  
helm upgrade vision-agent ./helm --reuse-values --set gpu.enabled=false
```

Make sure you have the matching node group running.
