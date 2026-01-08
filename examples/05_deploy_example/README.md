
## TODO / improvements

- shared UV cache for faster booting
- does it make sense to use Nvidia base?
- API/fastapi. (start when a user joins a call? or we make an API call?)
- include an agent monitoring panel?
- 2 nodes + load balancer?
- Terraform instead of nebius CLI?

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
```

Next create the cluster and be sure to replace the subnet and parent id

```
nebius mk8s cluster create \
  --parent-id project-e01zw2jzpr000vckjm7t7n \
  --name vision-agents \
  --control-plane-subnet-id vpcsubnet-e01jfyqqs0hfzpp2c3 \
  --control-plane-version 1.32.11 \
  --control-plane-endpoints-public-endpoint
```

Add a node group with the registry service account (enables automatic image pulling):

```
nebius mk8s node-group create \
  --parent-id <cluster-id-from-above> \
  --name default \
  --template-resources-platform cpu-d3 \
  --template-resources-preset 4vcpu-16gb \
  --template-boot-disk-size-gibibytes 64 \
  --template-service-account-id serviceaccount-e01p3340qm0bm4ns9d \
  --fixed-node-count 1
```

Get credentials for kubectl:

```
nebius mk8s cluster get-credentials --id <cluster-id> --external --force
kubectl get nodes  # verify connection
```

# 1. Build the Docker image (force linux)

Build the docker image. This image doesn't include deps so it will be fast.

```
cd examples/05_deploy_example
docker buildx build --platform linux/amd64 -t vision-agent-deploy .
```

# 2. Lookup your registry id

```
nebius registry list
```

# 3. Tag for your registry

```
docker tag vision-agent-deploy cr.eu-west1.nebius.cloud/e01mct3z7jptkdf0nr/vision-agent-deploy:latest
```

# 4. Push to registry

```
docker push cr.eu-west1.nebius.cloud/e01mct3z7jptkdf0nr/vision-agent-deploy:latest
```

# 5. Start the cluster with Helm (first time or update)

```
helm upgrade --install vision-agent ./helm \
  --set image.repository="cr.eu-west1.nebius.cloud/e01mct3z7jptkdf0nr/vision-agent-deploy" \
  --set image.tag=latest \
  --set cache.enabled=true
```

# 6. Create the required secrets and restart

```
kubectl create secret generic vision-agent-env --from-env-file=.env
kubectl rollout restart deployment/vision-agent
```

# Watch logs

If the container is still starting this will output "ContainerCreating" otherwise will show the logs

```
kubectl logs -l app.kubernetes.io/name=vision-agent -f --tail=100
```
