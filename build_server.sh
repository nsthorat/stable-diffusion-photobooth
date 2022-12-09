#!/bin/bash
set -e

GCP_PROJECT_ID="aiphotobooth-369522"
REPO_NAME="aiphotobooth"
IMAGE_NAME="server"
VERSION="latest"
REGION='us-east1'
IMAGE_URL="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${VERSION}"

poetry export --without-hashes --without dev > deps.txt

npx webpack build

# Comment this and uncomment above to build the image locally instead.
gcloud builds submit \
  --project=$GCP_PROJECT_ID \
  --ignore-file=.gcloudignore \
  .

#gcloud run deploy --image $IMAGE_URL --project $GCP_PROJECT_ID
