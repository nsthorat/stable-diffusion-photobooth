# ---
# output-directory: "/tmp/stable-diffusion-xl-turbo"
# args: []
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL Turbo Image-to-image
#
# This example is similar to the [Stable Diffusion XL](/docs/examples/stable_diffusion_xl)
# example, but it's a distilled model trained for real-time synthesis and is image-to-image. Learn more about it [here](https://stability.ai/news/stability-ai-sdxl-turbo).
#
# Input prompt:
# `dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k`
#
# Input             |  Output
# :-------------------------:|:-------------------------:
# ![](./stable_diffusion_turbo_input.png)  |  ![](./stable_diffusion_turbo_output.png)

# ## Basic setup

from io import BytesIO
from pathlib import Path

from modal import Image, Stub, build, enter, gpu, method

# ## Define a container image


image = Image.debian_slim().pip_install(
  "Pillow~=10.1.0",
  "diffusers~=0.24.0",
  "transformers~=4.35.2",  # This is needed for `import torch`
  "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
  "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
)

stub = Stub("sd_generate", image=image)

with image.imports():
  import torch
  from diffusers import AutoPipelineForText2Image
  from huggingface_hub import snapshot_download
  from PIL import Image


# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
  @build()
  def download_models(self):
    # Ignore files that we don't need to speed up download time.
    ignore = [
      "*.bin",
      "*.onnx_data",
      "*/diffusion_pytorch_model.safetensors",
    ]

    snapshot_download("runwayml/stable-diffusion-v1-5", ignore_patterns=ignore)

  @enter()
  def enter(self):
    self.pipe = AutoPipelineForText2Image.from_pretrained(
      "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

  @method()
  def inference(self, prompt, width=640, height=480, num_inference_steps=25):
    # init_image = load_image(image)
    # print(f"Running inference with {num_inference_steps} steps and strength {strength}")
    # # "When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1"
    # # See: https://huggingface.co/stabilityai/sdxl-turbo
    # assert num_inference_steps * strength >= 1

    image = self.pipe(
      prompt, num_inference_steps=num_inference_steps, width=width, height=height
    ).images[0]

    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    image_bytes = byte_stream.getvalue()

    return image_bytes


DEFAULT_IMAGE_PATH = Path(__file__).parent / "demo_images/dog.png"


@stub.local_entrypoint()
def main(
  prompt="dog wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
):
  output_image_bytes = Model().inference.remote(prompt)

  dir = Path("/tmp/stable-diffusion-xl-turbo")
  if not dir.exists():
    dir.mkdir(exist_ok=True, parents=True)

  output_path = dir / "output.png"
  print(f"Saving it to {output_path}")
  with open(output_path, "wb") as f:
    f.write(output_image_bytes)


# ## Running the model
#
# We can run the model with different parameters using the following command,
# ```
# modal run stable_diffusion_xl_turbo.py --prompt="harry potter, glasses, wizard"
# ```
