terraform {
  required_version = ">= 1.5.0"
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

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

# Raw + staging data lake bucket. Objects roll to Nearline after 30 days
# since the pipeline only reads the last few days for backfills.
resource "google_storage_bucket" "data_lake" {
  name                        = "${var.project_id}-raw"
  location                    = var.region
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_storage_bucket" "tmp" {
  name                        = "${var.project_id}-tmp"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  lifecycle_rule {
    condition {
      age = 3
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_bigquery_dataset" "warehouse" {
  dataset_id                 = "warehouse"
  location                   = var.region
  default_table_expiration_ms = null
}

resource "google_bigquery_table" "training_events" {
  dataset_id = google_bigquery_dataset.warehouse.dataset_id
  table_id   = "training_events"

  time_partitioning {
    type  = "DAY"
    field = "event_date"
  }

  schema = file("${path.module}/schema/training_events.json")
}

output "data_lake_bucket" {
  value = google_storage_bucket.data_lake.name
}

output "warehouse_dataset" {
  value = google_bigquery_dataset.warehouse.dataset_id
}
