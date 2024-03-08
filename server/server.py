import base64
import math
import random

# from stability_sdk import client
import os
import io
from urllib import parse
import functools
from google.cloud import storage
from datetime import datetime
import json
import modal

from PIL import Image
from PIL.Image import Image as PILImageType
import re

from flask import Flask
from flask import request, send_file, send_from_directory
from typing import Optional, IO
from datasets import load_dataset

from .utils import DebugTimer


app = Flask(__name__)

BUCKET_NAME = "ai-photobooth-images"

# NOTE: If flask stops working and 403s, flush stuff here:
# chrome://net-internals/#sockets


@app.route("/generate_image", methods=["GET", "POST"])
def generate_image():
  data = request.get_json()

  prompt = data["prompt"]
  image_base64 = data["imageBase64"][len("data:image/png;base64,") :]

  buf = io.BytesIO(base64.b64decode(image_base64))
  # Resize to 384x512 (this is a 4:3 image)
  image_pil = Image.open(buf).convert("RGB").resize((512, 384))

  print(f'Generating image for "{prompt}" ...')
  output_img = generate_image(prompt, image_pil, width=image_pil.width, height=image_pil.height)

  output_img_base64 = base64.b64encode(output_img).decode("utf-8")
  print(f"Done generating image: {len(output_img_base64)} bytes.")
  return {"ai_base64": output_img_base64, "generate_index": int(request.args.get("generate_index"))}


@app.route("/generate")
def generate():
  b64_data = parse.unquote(request.args.get("init_img"))
  b64_data = b64_data[len("data:image/png;base64,") :]
  print(b64_data)
  init_img = base64.b64decode(b64_data)
  buf = io.BytesIO(init_img)
  # pil_init_img = Image.open('examples/og.png').convert('RGB').resize((512, 512))
  # print(pil_init_img.mode)
  pil_init_img = Image.open(buf).convert("RGB")
  print(pil_init_img)
  # pil_init_img.show()
  # pil_init_img = None

  img = generate_image(
    request.args.get("prompt"), pil_init_img, width=pil_init_img.width, height=pil_init_img.height
  )

  print("generated img", Image.open(io.BytesIO(img)).mode)
  return send_file(io.BytesIO(img), mimetype="image/png")


@app.route("/save_images", methods=["POST"])
def save_images():
  data = request.get_json()
  assert data is not None
  # Get the current date and time
  now = datetime.now()

  # Format the date and time as a string
  filename = now.strftime("%Y_%m_%d_%H_%M_%S")
  filepath = f"gs://{BUCKET_NAME}/photobooth/{filename}.json"

  with open_file(filepath, "w") as f:
    f.write(json.dumps(data))

  print("Wrote to ", filepath)

  # Save individual images.
  ai_image_paths = []
  real_image_paths = []
  for i, images in enumerate(data):
    ai_image_b64 = images["aiImage"]
    real_image_b64 = images["realImage"]

    ai_image = base64.b64decode(ai_image_b64)
    real_image = base64.b64decode(real_image_b64)
    ai_image_filename = f"gs://{BUCKET_NAME}/collage/{filename}_{i}_ai.png"
    real_image_filename = f"gs://{BUCKET_NAME}/collage/{filename}_{i}_real.png"
    with open_file(ai_image_filename, "wb") as f:
      f.write(ai_image)
    with open_file(real_image_filename, "wb") as f:
      f.write(real_image)
    ai_image_paths.append(ai_image_filename)
    real_image_paths.append(real_image_filename)
    print(f"Wrote images {i} to {ai_image_filename} and {real_image_filename}.")

  # Open the manifest file and write the new visit to it.
  try:
    with open_file("gs://ai-photobooth-images/collage/visits.json") as f:
      visits = json.loads(f.read())
  except Exception:
    visits = {}
  if "ai_image_paths" not in visits:
    visits["ai_image_paths"] = []
  visits["ai_image_paths"].extend(ai_image_paths)
  if "real_image_paths" not in visits:
    visits["real_image_paths"] = []
  visits["real_image_paths"].extend(real_image_paths)
  with open_file("gs://ai-photobooth-images/collage/visits.json", "w") as f:
    f.write(json.dumps(visits))
  return {"file": filepath}


@app.route("/list_visits", methods=["GET"])
def list_visits():
  try:
    with open_file("gs://ai-photobooth-images/collage/visits.json") as f:
      visits = json.loads(f.read())
  except Exception:
    visits = {}

  return visits


@app.route("/image", methods=["GET"])
def image():
  image_type = request.args.get("type")
  visit = request.args.get("visit")

  with open_file(os.path.join(f"gs://{BUCKET_NAME}", visit)) as f:
    visit_json = json.loads(f.read())

  image_b64 = visit_json[0][image_type]
  img_bytes = base64.b64decode(image_b64)
  buf = io.BytesIO(img_bytes)
  return send_file(buf, mimetype="image/png")


def serve_pil_image(pil_img):
  img_io = io.BytesIO()
  pil_img.save(img_io, "PNG", quality=100)
  img_io.seek(0)
  return send_file(img_io, mimetype="image/[mg]")


DATASET = None


@app.route("/generate_collage", methods=["POST"])
def generate_collage():
  data = request.get_json()
  assert data is not None

  global DATASET
  if DATASET is None:
    DATASET = load_dataset("Francesco/people-in-paintings")

  # Choose a random id from 1 - 100
  img_id = math.floor(random.random() * 100)
  target_image = DATASET["test"]["image"][img_id]
  # source_images = [target_image]
  source_images = DATASET["test"]["image"][0:100]

  collage = modal.Function.lookup("collage", "Collage.collage")

  with DebugTimer("Collage"):
    collage_response = collage.remote(
      {
        # "target_image": target_image,
        "target_prompt": data["prompt"],
        "target_num_slices": data["targetSlices"],
        "upscale_target": float(data["upscale"]),
        # "source_images": source_images,
        "source_image_paths": data["imagePaths"],
        "source_num_slices": data["sourceSlices"],
        "resnet_layer": data["resnetLayer"],
        # "target_size": data["targetSize"],
        "diversity": data["diversity"],
      }
    )

  print(collage_response)

  # sleep(10)

  # print(f"Done generating image: {len(output_img_base64)} bytes.")
  return serve_pil_image(collage_response["collage"])


@app.route("/loading.gif")
def loading_gif():
  return send_from_directory("../dist/static", "loading.gif")


@app.route("/logo.png")
def logo():
  return send_from_directory("../dist/static", "logo.png")


@app.route("/bundle.js")
def bundle():
  return send_from_directory("../dist/static/", "bundle.js")


@app.route("/")
@app.route("/collage")
def index():
  return send_from_directory("../dist/", "index.html")


# STABILITY_HOST = "grpc.stability.ai:443"
# print(os.environ["STABILITY_KEY"])
# stability_api = client.StabilityInference(
#   key=os.environ["STABILITY_KEY"],
#   # engine=args.engine,
#   verbose=True,
# )


def generate_image(prompt: str, init_image: PILImageType = None, width=512, height=512):
  sd_modal = modal.Function.lookup("stable-diffusion-xl-turbo", "Model.inference")

  with DebugTimer("Stable Diffusion"):
    collage_response = sd_modal.remote(init_image, prompt, strength=0.6, num_inference_steps=8)

  return collage_response
  answers = stability_api.generate(
    prompt,
    init_image=init_image,
    safety=False,
    start_schedule=0.6,
    end_schedule=0.01,
    width=width,
    height=height,
  )
  images = client.process_artifacts_from_answers(
    "generation", prompt, answers, write=False, verbose=True
  )

  for path, artifact in images:
    if artifact.type == client.generation.ARTIFACT_IMAGE:
      return artifact.binary


GCS_PROTOCOL = "gs://"
GCS_REGEX = re.compile(f"{GCS_PROTOCOL}(.*?)/(.*)")


@functools.cache
def _get_storage_client(thread_id: Optional[int] = None) -> storage.Client:
  # The storage client is not thread safe so we use a thread_id to make sure each thread gets a
  # separate storage client.
  del thread_id
  return storage.Client()


def _parse_gcs_path(filepath: str) -> tuple[str, str]:
  # match a regular expression to extract the bucket and filename
  if matches := GCS_REGEX.match(filepath):
    bucket_name, object_name = matches.groups()
    return bucket_name, object_name
  raise ValueError(f"Failed to parse GCS path: {filepath}")


def _get_gcs_blob(filepath: str) -> storage.Blob:
  bucket_name, object_name = _parse_gcs_path(filepath)
  storage_client = _get_storage_client()
  bucket = storage_client.bucket(bucket_name)
  return bucket.blob(object_name)


def open_file(filepath: str, mode: str = "r") -> IO:
  """Opens a file handle. It works with both GCS and local paths."""
  if filepath.startswith(GCS_PROTOCOL):
    blob = _get_gcs_blob(filepath)
    return blob.open(mode)

  write_mode = "w" in mode
  binary_mode = "b" in mode

  if write_mode:
    base_path = os.path.dirname(filepath)
    os.makedirs(base_path, exist_ok=True)

  encoding = None if binary_mode else "utf-8"
  return open(filepath, mode=mode, encoding=encoding)


def main() -> None:
  # Set environment variables from flags.
  port = int(os.environ.get("PORT", 8080))
  app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
  main()
