job "vision-agent" {
  datacenters = ["dc1"]
  type        = "service"

  group "agent" {
    count = 1

    network {
      port "http" {
        to = 8080
      }
    }

    task "agent" {
      driver = "docker"

      config {
        image = "${artifact_registry}/simple-agent:latest"
        ports = ["http"]
      }

      template {
        data        = <<-EOF
          DEEPGRAM_API_KEY={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .deepgram_api_key }}{{ end }}
          ELEVENLABS_API_KEY={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .elevenlabs_api_key }}{{ end }}
          ANTHROPIC_API_KEY={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .anthropic_api_key }}{{ end }}
          OPENAI_API_KEY={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .openai_api_key }}{{ end }}
          STREAM_API_KEY={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .stream_api_key }}{{ end }}
          STREAM_API_SECRET={{ with nomadVar "nomad/jobs/vision-agent" }}{{ .stream_api_secret }}{{ end }}
        EOF
        destination = "secrets/env.env"
        env         = true
      }

      resources {
        cpu    = 500
        memory = 512
      }

      restart {
        attempts = 3
        interval = "5m"
        delay    = "15s"
        mode     = "fail"
      }
    }

    restart {
      attempts = 10
      interval = "5m"
      delay    = "25s"
      mode     = "delay"
    }
  }
}

