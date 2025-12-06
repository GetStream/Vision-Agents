# Nomad server instance
resource "google_compute_instance" "nomad_server" {
  name         = "nomad-server"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 20
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    # Install Docker
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu

    # Install Nomad
    curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list
    apt-get update && apt-get install -y nomad

    # Configure Nomad server
    cat > /etc/nomad.d/nomad.hcl <<'NOMAD'
    datacenter = "dc1"
    data_dir   = "/opt/nomad/data"

    server {
      enabled          = true
      bootstrap_expect = 1
    }

    client {
      enabled = true
    }

    plugin "docker" {
      config {
        allow_privileged = false
        volumes {
          enabled = true
        }
      }
    }
    NOMAD

    # Start Nomad
    systemctl enable nomad
    systemctl start nomad
  EOF

  service_account {
    email  = google_service_account.agent.email
    scopes = ["cloud-platform"]
  }

  tags = ["nomad-server"]
}

# Firewall for Nomad UI and API
resource "google_compute_firewall" "nomad" {
  name    = "nomad-allow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["4646", "4647", "4648"]
  }

  source_ranges = var.allowed_ips
  target_tags   = ["nomad-server"]
}

output "nomad_server_ip" {
  value = google_compute_instance.nomad_server.network_interface[0].access_config[0].nat_ip
}

output "nomad_ui_url" {
  value = "http://${google_compute_instance.nomad_server.network_interface[0].access_config[0].nat_ip}:4646"
}

