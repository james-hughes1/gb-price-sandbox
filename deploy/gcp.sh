#!/usr/bin/env bash
# Deploy the GB Power Price Sandbox to Google Cloud Run, set up to cost as little as possible:
# the app scales to zero when idle (slower first load, nothing billed in between).
#
#   Cloud Scheduler (10:15 London) -> Cloud Run Job (pretrain) --writes--> GCS bucket
#   Users -> Cloud Run Service (Streamlit)                      <--reads--  (mounted at /app/data)
#
# Usage:  PROJECT=my-project deploy/gcp.sh <step> [<step> ...]
# Steps, in first-time order:
#   setup     enable APIs; create bucket, service account, image repo; grant permissions
#   seed      upload your local data/ cache to the bucket (skips re-downloading years of data)
#   build     build the image with Cloud Build and push it to Artifact Registry
#   service   deploy (or update) the Streamlit app
#   job       deploy (or update) the daily pretrain job, then run it once
#   schedule  create (or update) the daily trigger for the job
#   all       every step above, in order
# Re-running a step is safe. After the first deploy, `build service job` ships new code (or let
# deploy/cloudbuild.yaml do it on every push).
set -euo pipefail

: "${PROJECT:?set PROJECT to your Google Cloud project id}"
REGION="${REGION:-europe-west2}"
BUCKET="${BUCKET:-${PROJECT}-gbps-data}"
SERVICE="${SERVICE:-gb-price-sandbox}"
JOB="${JOB:-gbps-pretrain}"
REPO="${REPO:-apps}"
SA_NAME="${SA_NAME:-gbps-runner}"
SCHEDULE="${SCHEDULE:-15 10 * * *}"   # Europe/London: just after the 10:00 D-1 forecast cutoff
PUBLIC="${PUBLIC:-true}"               # false: only authenticated callers (e.g. behind IAP)
MAX_INSTANCES="${MAX_INSTANCES:-1}"    # one instance: caps cost even if Train is hammered

SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/gb-price-sandbox"
DATA_FLAGS=(
  --add-volume "name=data,type=cloud-storage,bucket=${BUCKET}"
  --add-volume-mount "volume=data,mount-path=/app/data"
  --set-env-vars "GBPS_DATA_DIR=/app/data"
  --service-account "${SA}"
)

gc() { gcloud --project "${PROJECT}" "$@"; }

setup() {
  gc services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com cloudscheduler.googleapis.com storage.googleapis.com
  gc storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
    || gc storage buckets create "gs://${BUCKET}" --location "${REGION}" --uniform-bucket-level-access
  gc iam service-accounts describe "${SA}" >/dev/null 2>&1 \
    || gc iam service-accounts create "${SA_NAME}" --display-name "GB price sandbox runtime"
  gc storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member "serviceAccount:${SA}" --role roles/storage.objectAdmin >/dev/null
  # Lets Cloud Scheduler (acting as this account) start the job.
  gc projects add-iam-policy-binding "${PROJECT}" \
    --member "serviceAccount:${SA}" --role roles/run.invoker --condition None >/dev/null
  gc artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1 \
    || gc artifacts repositories create "${REPO}" --repository-format docker --location "${REGION}"
  # Lets Cloud Build deploy (used by deploy/cloudbuild.yaml on push).
  local number build_sa
  number="$(gc projects describe "${PROJECT}" --format 'value(projectNumber)')"
  build_sa="${number}-compute@developer.gserviceaccount.com"
  gc projects add-iam-policy-binding "${PROJECT}" \
    --member "serviceAccount:${build_sa}" --role roles/run.admin --condition None >/dev/null
  gc iam service-accounts add-iam-policy-binding "${SA}" \
    --member "serviceAccount:${build_sa}" --role roles/iam.serviceAccountUser >/dev/null
}

seed() {
  gc storage rsync --recursive data "gs://${BUCKET}"
}

build() {
  gc builds submit --tag "${IMAGE}:latest" .
}

service() {
  local auth=--allow-unauthenticated
  [[ "${PUBLIC}" == true ]] || auth=--no-allow-unauthenticated
  # --min-instances 0: scale to zero when idle (cold starts, but no idle cost).
  # --cpu-throttling: CPU billed only while handling a request (the cheaper billing mode).
  # --timeout: Streamlit keeps one long-lived connection per browser tab (training runs in it).
  # --session-affinity: Streamlit keeps session state in memory, so stick users to an instance.
  gc run deploy "${SERVICE}" --image "${IMAGE}:latest" --region "${REGION}" \
    --port 8501 --cpu 2 --memory 4Gi --timeout 3600 --session-affinity \
    --min-instances 0 --max-instances "${MAX_INSTANCES}" --cpu-throttling \
    --execution-environment gen2 \
    "${DATA_FLAGS[@]}" "${auth}"
}

job() {
  # Pretrain takes well under a minute; billed per second while it runs.
  gc run jobs deploy "${JOB}" --image "${IMAGE}:latest" --region "${REGION}" \
    --command python --args=-m,gb_price_sandbox.pretrain \
    --cpu 1 --memory 2Gi --task-timeout 30m --max-retries 1 \
    "${DATA_FLAGS[@]}"
  gc run jobs execute "${JOB}" --region "${REGION}" --wait
}

schedule() {
  local uri="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"
  local verb=create
  gc scheduler jobs describe "${JOB}-daily" --location "${REGION}" >/dev/null 2>&1 && verb=update
  gc scheduler jobs "${verb}" http "${JOB}-daily" --location "${REGION}" \
    --schedule "${SCHEDULE}" --time-zone "Europe/London" --http-method POST --uri "${uri}" \
    --oauth-service-account-email "${SA}"
}

[[ $# -gt 0 ]] || { sed -n '2,20p' "$0"; exit 1; }
for step in "$@"; do
  case "${step}" in
    setup | seed | build | service | job | schedule) "${step}" ;;
    all) setup; seed; build; service; job; schedule ;;
    *) echo "unknown step: ${step}" >&2; exit 1 ;;
  esac
done
