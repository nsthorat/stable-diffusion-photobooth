import base64

from stability_sdk import client
import os
import io
from urllib import parse

from PIL import Image

from flask import Flask
from flask import request, send_file, send_from_directory
import base64

app = Flask(__name__)

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

@app.route('/main_image')
def main_image():
  return send_from_directory('static', 'cat_outerspace.png')

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
    "generation", answers, write=False, verbose=True
  )

  for path, artifact in images:
    if artifact.type == client.generation.ARTIFACT_IMAGE:
      return artifact.binary


def main() -> None:
  app.run()
  pass

if __name__ == '__main__':
  main()
