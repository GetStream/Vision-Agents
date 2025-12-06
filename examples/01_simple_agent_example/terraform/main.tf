terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Artifact Registry for storing the Docker image
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "vision-agents"
  format        = "DOCKER"
}

# Service account for Nomad nodes
resource "google_service_account" "agent" {
  account_id   = "vision-agent"
  display_name = "Vision Agent Service Account"
}

# Allow pulling from Artifact Registry
resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.agent.email}"
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/vision-agents"
}

output "docker_push_command" {
  value = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/vision-agents/simple-agent:latest"
}
