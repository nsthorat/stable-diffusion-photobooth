from __future__ import annotations
from io import BytesIO

from pathlib import Path

from modal import Image, Stub, build, enter, method

model_id = "runwayml/stable-diffusion-v1-5"

stub = Stub("sd_inpaint")


image = (
  Image.debian_slim(python_version="3.10")
  .pip_install(
    "accelerate",
    "diffusers[torch]>=0.15.1",
    "ftfy",
    "torchvision",
    "transformers~=4.38.2",
    "triton",
    "safetensors",
  )
  .pip_install(
    "torch==2.0.1+cu117",
    find_links="https://download.pytorch.org/whl/torch_stable.html",
  )
  .pip_install("xformers", pre=True)
)

with image.imports():
  import torch


@stub.cls(image=image, gpu="A10G", container_idle_timeout=240)
class StableDiffusion:
  @build()
  @enter()
  def initialize(self):
    from diffusers import AutoPipelineForInpainting

    self.pipe = AutoPipelineForInpainting.from_pretrained(
      "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    )
    self.pipe.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    self.pipe.enable_xformers_memory_efficient_attention()

  @method()
  def run_inference(self, init_image, mask_image, prompt: str) -> list[bytes]:
    import torch

    generator = torch.Generator("cuda").manual_seed(92)
    # prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
    image = self.pipe(
      prompt=prompt, image=init_image, mask_image=mask_image, generator=generator
    ).images[0]
    # make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    image_bytes = byte_stream.getvalue()

    return image_bytes
    # with torch.inference_mode():
    #   with torch.autocast("cuda"):
    #     images = self.pipe(
    #       [prompt] * batch_size,
    #       num_inference_steps=steps,
    #       guidance_scale=7.0,
    #     ).images

    # # Convert to PNG bytes
    # image_output = []
    # for image in images:
    #   with io.BytesIO() as buf:
    #     image.save(buf, format="PNG")
    #     image_output.append(buf.getvalue())
    # return image_output


@stub.local_entrypoint()
def entrypoint():
  # print(f"prompt => {prompt}, steps => {steps}, samples => {samples}, batch_size => {batch_size}")
  from diffusers.utils import load_image

  # load base and mask image
  init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
  )
  mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
  )

  dir = Path("/tmp/stable-diffusion")
  if not dir.exists():
    dir.mkdir(exist_ok=True, parents=True)

  sd = StableDiffusion()
  # for i in range(samples):
  # t0 = time.time()
  image = sd.run_inference.remote(
    init_image, mask_image, "Concept art of a digital dog, highly detailed, 8K"
  )
  # total_time = time.time() - t0
  # print(f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image).")
  output_path = dir / "output_test.png"
  print(f"Saving it to {output_path}")
  with open(output_path, "wb") as f:
    f.write(image)
