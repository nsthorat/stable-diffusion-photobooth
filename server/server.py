import base64

from stability_sdk import client
import os
import io
from urllib import parse
import functools
from google.cloud import storage
from datetime import datetime
import json

from PIL import Image
import re

from flask import Flask
from flask import request, send_file, send_from_directory
import base64
from typing import Optional, IO

app = Flask(__name__)

# NOTE: If flask stops working and 403s, flush stuff here:
# chrome://net-internals/#sockets

@app.route("/generate_image", methods=['GET', 'POST'])
def generate_image():
  data = request.get_json()

  prompt = data['prompt']
  image_base64 = data['imageBase64'][len('data:image/png;base64,'):]

  buf = io.BytesIO(base64.b64decode(image_base64))
    # Resize to 384x512 (this is a 4:3 image)
  image_pil = Image.open(buf).convert('RGB').resize((512, 384))

  print(f'Generating image for "{prompt}" ...')
  output_img = generate_image(prompt, image_pil, width=image_pil.width, height=image_pil.height)

  output_img_base64 = base64.b64encode(output_img).decode('utf-8')
  print(f'Done generating image: {len(output_img_base64)} bytes.')
  return {'ai_base64': output_img_base64, 'generate_index': int(request.args.get('generate_index'))}

@app.route("/generate")
def generate():
  b64_data = parse.unquote(request.args.get('init_img'))
  b64_data = b64_data[len('data:image/png;base64,'):]
  print(b64_data)
  init_img = base64.b64decode(b64_data)
  buf = io.BytesIO(init_img)
  #pil_init_img = Image.open('examples/og.png').convert('RGB').resize((512, 512))
  #print(pil_init_img.mode)
  pil_init_img = Image.open(buf).convert('RGB')
  print(pil_init_img)
  #pil_init_img.show()
  #pil_init_img = None

  img = generate_image(request.args.get('prompt'), pil_init_img, width=pil_init_img.width, height=pil_init_img.height)

  print('generated img', Image.open(io.BytesIO(img)).mode)
  return send_file(io.BytesIO(img), mimetype='image/png')

@app.route("/save_images", methods=['POST'])
def save_images():
  data = request.get_json()
  # Get the current date and time
  now = datetime.now()

  # Format the date and time as a string
  filename = now.strftime("%Y_%m_%d_%H_%M_%S")
  filepath = f'gs://ai-photobooth-images/photobooth/{filename}.json'

  with open_file(filepath, 'w') as f:
    f.write(json.dumps(data))

  print('Wrote to ', filepath)
  return {'file': filepath}

@app.route('/main_image')
def main_image():
  return send_from_directory('static', 'cat_outerspace.png')

@app.route('/loading.gif')
def loading_gif():
  return send_from_directory('../dist/static', 'loading.gif')

@app.route('/')
def index():
  return send_from_directory('../dist/', 'index.html')


@app.route('/bundle.js')
def bundle():
  return send_from_directory('../dist/static/', 'bundle.js')

STABILITY_HOST = "grpc.stability.ai:443"
print(os.environ["STABILITY_KEY"])
stability_api = client.StabilityInference(
  key=os.environ["STABILITY_KEY"],
  #engine=args.engine,
  verbose=True
)

def generate_image(prompt: str, init_image: Image = None, width=512, height=512):
  answers = stability_api.generate(
    prompt,
    init_image=init_image,
    safety=False,
    start_schedule=.6,
    end_schedule=0.01,
    width=width,
    height=height)
  images = client.process_artifacts_from_answers(
    "generation", prompt, answers, write=False, verbose=True
  )

  for path, artifact in images:
    if artifact.type == client.generation.ARTIFACT_IMAGE:
      return artifact.binary

GCS_PROTOCOL = 'gs://'
GCS_REGEX = re.compile(f'{GCS_PROTOCOL}(.*?)/(.*)')

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
  raise ValueError(f'Failed to parse GCS path: {filepath}')

def _get_gcs_blob(filepath: str) -> storage.Blob:
  bucket_name, object_name = _parse_gcs_path(filepath)
  storage_client = _get_storage_client()
  bucket = storage_client.bucket(bucket_name)
  return bucket.blob(object_name)

def open_file(filepath: str, mode: str = 'r') -> IO:
  """Opens a file handle. It works with both GCS and local paths."""
  if filepath.startswith(GCS_PROTOCOL):
    blob = _get_gcs_blob(filepath)
    return blob.open(mode)

  write_mode = 'w' in mode
  binary_mode = 'b' in mode

  if write_mode:
    base_path = os.path.dirname(filepath)
    os.makedirs(base_path, exist_ok=True)

  encoding = None if binary_mode else 'utf-8'
  return open(filepath, mode=mode, encoding=encoding)

def main() -> None:
  # Set environment variables from flags.
  port = int(os.environ.get('PORT', 8080))
  app.run(debug=True, host='0.0.0.0', port=port)

if __name__ == '__main__':
  main()
