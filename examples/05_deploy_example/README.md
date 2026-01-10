
## TODO / improvements

- shared UV cache for faster booting
- does it make sense to use Nvidia base?
- merge monitoring and HTTP efforts into this

# Secrets

First copy the .env.example

```
cp .env.example .env
```

Next fill in the required variables and run the example locally to verify everything works

```
uv run deploy_example.py
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
  --parent-id <your-project-id> \
  --name vision-agents \
  --control-plane-subnet-id <your-subnet-id> \
  --control-plane-version 1.31 \
  --control-plane-endpoints-public-endpoint
```

## Add a Node Group

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

### Option B: GPU Node (H200, for production)

```
nebius mk8s node-group create \
  --parent-id <cluster-id-from-above> \
  --name gpu \
  --template-resources-platform gpu-h200-sxm \
  --template-resources-preset 1gpu-16vcpu-200gb \
  --template-boot-disk-size-gibibytes 200 \
  --template-service-account-id <your-service-account-id> \
  --template-metadata-labels nebius.com/gpu=true \
  --fixed-node-count 1
```

Available GPU presets:
- `1gpu-16vcpu-200gb` - 1x H200, 16 vCPU, 200GB RAM
- `8gpu-128vcpu-1600gb` - 8x H200, 128 vCPU, 1.6TB RAM

### Get kubectl credentials

```
nebius mk8s cluster get-credentials --id <cluster-id> --external --force
kubectl get nodes  # verify connection
```

# 1. Build the Docker image

```
cd examples/05_deploy_example
docker buildx build --platform linux/amd64 -t vision-agent-deploy .
```

# 2. Push to registry

```
# Lookup your registry id
nebius registry list

# Tag and push
docker tag vision-agent-deploy cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:latest
docker push cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy:latest
```

# 3. Deploy with Helm

## CPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy" \
  --set image.tag=latest \
  --set image.pullPolicy=Always \
  --set cache.enabled=true \
  --set gpu.enabled=false
```

## GPU deployment

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="cr.eu-west1.nebius.cloud/<registry-id>/vision-agent-deploy" \
  --set image.tag=latest \
  --set image.pullPolicy=Always \
  --set cache.enabled=true \
  --set gpu.enabled=true
```

# 4. Create secrets and restart

```
kubectl create secret generic vision-agent-env --from-env-file=.env
kubectl rollout restart deployment/vision-agent
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
