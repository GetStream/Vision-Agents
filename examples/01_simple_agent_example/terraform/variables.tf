variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "e2-small"
}

variable "allowed_ips" {
  description = "IPs allowed to access Nomad UI (e.g., your office IP)"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Restrict this in production!
}

variable "github_repo" {
  description = "GitHub repo for OIDC auth (e.g., owner/repo)"
  type        = string
}

# API keys are stored in Nomad variables, not Terraform
# See nomad/deploy.sh for secret management

