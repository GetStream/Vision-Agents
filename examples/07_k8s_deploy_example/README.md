## TODO / improvements

- merge monitoring and HTTP efforts into this

# Tips

* US-east. Services like Stream run a global edge network. But many providers default to US-east. So you typically want
  to run in US-east for optimal latency
* CPU build is quick to get up and running. GPU/CUDA takes hours.
* This guide is cloud-agnostic. You can use any Kubernetes provider (GKE, EKS, AKS, etc.)
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

[kubectl](https://kubernetes.io/docs/tasks/tools/)

[HELM](https://helm.sh/docs/intro/install/)

```
brew install helm
```

# 0. Create a Kubernetes cluster

Create a Kubernetes cluster using your cloud provider of choice:

- **GKE:** `gcloud container clusters create vision-agents ...`
- **EKS:** `eksctl create cluster --name vision-agents ...`
- **AKS:** `az aks create --name vision-agents ...`

Refer to your provider's documentation for cluster creation details.

## Add a Node Group

### Option A: CPU Node (cheaper, for testing)

Create a node group with at least 4 vCPUs and 16GB RAM.

### Option B: GPU Node (for running models locally)

GPUs are expensive. You typically don't want to run your voice agents on a server with a GPU.
Even from a load balancing perspective you typically want to spin the GPU related work into its own cluster.

If you need GPU support, create a node group with an NVIDIA GPU and ensure the appropriate drivers (e.g. CUDA 12) are installed.

### Get kubectl credentials

```
# Use your provider's CLI to configure kubectl access, e.g.:
# gcloud container clusters get-credentials vision-agents
# aws eks update-kubeconfig --name vision-agents
# az aks get-credentials --name vision-agents

kubectl get nodes  # verify connection
```

# 1. Build the Docker image

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

# 2. Push to registry

Tag and push to your container registry:

```
# CPU
docker tag vision-agent-deploy <your-registry>/vision-agent-deploy:latest
docker push <your-registry>/vision-agent-deploy:latest

# GPU
docker tag vision-agent-deploy:gpu <your-registry>/vision-agent-deploy:gpu
docker push <your-registry>/vision-agent-deploy:gpu
```

# 3. Deploy with Helm

## CPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="<your-registry>/vision-agent-deploy" \
  --set image.tag=latest \
  --set image.pullPolicy=Always \
  --set cache.enabled=true \
  --set gpu.enabled=false \
  --set secrets.existingSecret=vision-agent-env
```

## GPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="<your-registry>/vision-agent-deploy" \
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

Scale your node group to 0 nodes using your provider's CLI, then scale back up when needed.

## Switch between CPU and GPU

Just change `gpu.enabled` and redeploy:

```
# Switch to GPU
helm upgrade vision-agent ./helm --reuse-values --set gpu.enabled=true

# Switch to CPU
helm upgrade vision-agent ./helm --reuse-values --set gpu.enabled=false
```

Make sure you have the matching node group running.
