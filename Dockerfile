# This dockerfile is built on Google Cloud with Cloud Builds.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /code

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# NOTE: Not all project files are copied because we ignore them before submitting the build to
# Google Cloud. See `.dockerignore` and `.gcloudignore` for ignored files.
COPY deps.txt .
RUN pip install --no-cache-dir -r deps.txt

# Remove local pypi and poetry cache.
RUN rm -rf ~/.cache

# Copy the source code.
COPY ./dist ./dist
COPY ./server ./server

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
ENV RUN_ON_CLOUD True
ENV PUBLIC_URL https://aiphotobooth-dour4pltwa-uc.a.run.app

CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 0 server.server:app
