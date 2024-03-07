"""Computes a collage given images uploaded."""

import functools
import io
import json
import math
import os
import re
from typing import Optional
from modal import Image, Stub, method, gpu, Secret
import numpy as np
from pydantic import BaseModel, ConfigDict
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
import torch
from .utils import chunks, DebugTimer
from .image_embed import EmbedRequest, ImageEmbed
from google.cloud import storage
from typing import IO

IMAGE_BATCH_SIZE = 1024  # 16384
EMBEDDING_DIMS = 128
# EMBEDDING_DIMS = None

MAX_DIVERSITY_K = 10


class CollageRequest(BaseModel):
  """Request to collage a target image, given a set of source images."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  target_image: Optional[PILImageType] = None
  target_prompt: Optional[str] = None
  target_num_slices: int
  upscale_target: float = 1.0
  target_size: Optional[tuple[int, int]] = None

  source_images: Optional[list[PILImageType]] = None
  source_image_paths: Optional[list[str]] = None
  source_num_slices: int

  resnet_layer: str
  choose_top_k: int = 0

  # Increase this to 1.0 to increase how random we select from top-k.
  diversity: float = 0.0


class CollageResponse(BaseModel):
  """Response containing a list of embedding vectors."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  collage: PILImageType


PROJECTION_SEED = 42
TORCH_SEED = 42
MODEL_DIR = "./model"


def _download_model() -> None:
  from torchvision.models import resnet18
  import torch

  torch.manual_seed(TORCH_SEED)
  torch.cuda.manual_seed_all(TORCH_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  resnet18()


image = (
  Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
  .pip_install(
    "pydantic~=2.5",
    "diffusers~=0.24.0",
    "fastapi~=0.108",
    "google-cloud-storage~=2.5.0",
    "pillow~=10.2.0",
    "datasets",
    "torch==2.2.1",
    "torchvision==0.17.1",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "pydantic~=2.5",
    "scikit-learn~=1.3",
  )
  .env({"TORCH_HOME": MODEL_DIR})
  .run_function(_download_model, timeout=60 * 20)
  # .run_function(_download_spacy_model, secret=Secret.from_name("hf-token"), timeout=60 * 10)
)

stub = Stub("collage", image=image)


@stub.cls(
  gpu=gpu.A100(size="80GB"),
  container_idle_timeout=300,
  cpu=8.0,
  secrets=[Secret.from_name("gcloud-secret")],
)
class Collage:
  """Computes a collage."""

  def __enter__(self) -> None:
    # Pre-load the model.
    from torchvision.models import resnet18
    from torchvision.models import feature_extraction
    from torchvision import transforms
    import torch
    from diffusers import AutoPipelineForText2Image
    from datasets import load_dataset

    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    self.model = resnet18()
    self.model.to("cuda")
    self.model.eval()
    self.train_nodes, self.eval_nodes = feature_extraction.get_graph_node_names(self.model)

    self.preprocess = transforms.Compose(
      [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]
    )

    self.generate_image_pipeline = AutoPipelineForText2Image.from_pretrained(
      "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    source_images = load_dataset("Francesco/people-in-paintings")

    self.default_source_images = source_images["test"]["image"][0:100]

  @method()
  def collage(self, request_dict: dict) -> dict:
    """Compute the collage."""
    request = CollageRequest.model_validate(request_dict)

    import torch

    if request.target_prompt:
      with DebugTimer("Generating target image"):
        target_image = self.generate_image_pipeline(
          request.target_prompt, num_inference_steps=25
        ).images[0]
    elif request.target_image:
      target_image = request.target_image
    else:
      raise ValueError("No target image or prompt provided.")
    # target_image = request.target_image or target_image

    target_slices = imgslices(target_image, request.target_num_slices)

    source_images: list[PILImageType]
    if not request.source_image_paths:
      source_images = request.source_images if request.source_images else self.default_source_images
    else:
      source_images = []
      for path in request.source_image_paths:
        with open_file(path, "rb") as f:
          img_bytes = f.read()
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        source_images.append(pil_image)
        print("pil image=", pil_image, pil_image.size, pil_image.mode)

    source_slices = [
      imgslices(source_image, request.source_num_slices) for source_image in source_images
    ]

    image_embed = ImageEmbed.embed  # modal.Function.lookup("image_embed", "ImageEmbed.embed")

    # Compute target embeddings.
    requests = []
    for batch in chunks(target_slices, size=IMAGE_BATCH_SIZE):
      requests.append(
        EmbedRequest(images=batch, layer=request.resnet_layer, random_projection_dim=EMBEDDING_DIMS)
      )

    target_embeddings_list = []
    with DebugTimer(
      f"Computing target embeddings: {len(target_slices)} patches, {len(requests)} batches..."
    ):
      for response in map(self.embed, requests):  # , #image_embed.map(requests):
        target_embeddings_list.append(response)
      target_embeddings = torch.from_numpy(np.concatenate(target_embeddings_list, axis=0)).to(
        "cuda"
      )

    # Flatten source_slices.
    flat_source_slices = [slice for slices in source_slices for slice in slices]
    requests = []
    for batch in chunks(flat_source_slices, size=IMAGE_BATCH_SIZE):
      requests.append(
        EmbedRequest(images=batch, layer=request.resnet_layer, random_projection_dim=EMBEDDING_DIMS)
      )

    source_embeddings_list = []
    with DebugTimer(
      f"Computing source embeddings: {len(source_slices)} images, "
      f"{len(flat_source_slices)} patches, {len(requests)} batches..."
    ):
      # for response in image_embed.map(requests):
      for response in map(self.embed, requests):
        source_embeddings_list.append(response)
      # print("Source embeddings:", source_embeddings_list)

    num_images = len(source_images)
    num_slices = request.source_num_slices * request.source_num_slices

    source_embeddings = torch.from_numpy(
      np.concatenate(source_embeddings_list, axis=0).reshape((num_images, num_slices, -1))
    ).to("cuda")

    with DebugTimer(
      f"Computing patch similarities: {target_embeddings.shape[0]} target x "
      f"{source_embeddings.shape[0]} source patches"
    ):
      top_k = math.ceil(MAX_DIVERSITY_K * request.diversity) + 1
      similarities = topk_similarities(target_embeddings, source_embeddings, k=top_k)
      # print("sims indices", similarities.indices.shape)

      num_source_slices = len(source_slices[0])
      similarity_images: list[PILImageType] = []
      for si, similarity_ids in enumerate(list(similarities.indices.cpu().numpy())):
        # Choose a random number between 0 and top_k.
        if top_k == 0:
          select_idx = 0
        else:
          select_idx = np.random.randint(0, top_k)
        flat_similarity_id = similarity_ids[select_idx]

        # Convert the indices to the original image indices.
        image_id, slice_id = divmod(flat_similarity_id, num_source_slices)

        similarity_images.append(source_slices[image_id][slice_id])

      # print(len(similarity_images))
    target_width, target_height = target_image.size
    target_size = request.target_size or (
      math.floor(target_height * request.upscale_target),
      math.floor(target_width * request.upscale_target),
    )
    with DebugTimer(f"Stitching collage to {target_size}"):
      stitched_output = stitch_slices(similarity_images, target_size)

    torch.cuda.empty_cache()
    return CollageResponse(collage=stitched_output).model_dump()

  @functools.cache
  def extractor(self, layer: str):
    from torchvision.models import feature_extraction

    return_nodes = {
      # node_name: user-specified key for output dict
      layer: "layer1",
      # node: node for node in train_nodes
    }

    return feature_extraction.create_feature_extractor(self.model, return_nodes=return_nodes)

  @functools.cache
  def random_projector(self, num_dims: int, max_dims: int):
    import torch

    return torch.Tensor(np.random.randn(num_dims, max_dims)).to("cuda")

  def embed(self, request_dict: dict) -> dict:
    """Embed documents and return a list of embedding vectors."""
    import torch

    request = EmbedRequest.model_validate(request_dict)
    if request.images:
      images = request.images
    else:
      raise ValueError("No images provided.")

    print("IMAGE", images[0])
    with DebugTimer("Preprocessing"):
      img_slices = [self.preprocess(slice).to("cuda") for slice in images]

      img_batch = torch.stack(img_slices)
    with DebugTimer("Computing activations"):
      with torch.no_grad():
        activations = self.extractor(request.layer)(img_batch)

    embeddings = activations["layer1"].flatten(start_dim=1)
    if request.random_projection_dim and embeddings.shape[1] > request.random_projection_dim:
      with DebugTimer(
        f"Projecting from {embeddings.shape[1]} to {request.random_projection_dim} dimensions"
      ):
        transformer = self.random_projector(embeddings.shape[1], request.random_projection_dim)
        embeddings = embeddings.matmul(transformer)
        # from sklearn.random_projection import GaussianRandomProjection

        # projection = GaussianRandomProjection(
        #   request.random_projection_dim, random_state=request.projection_seed
        # )
        # embeddings = projection.fit_transform(embeddings)
        # log("Projected vectors down to dimension", embeddings.shape[1])

    with DebugTimer("Normalizing"):
      embeddings = torch.nn.functional.normalize(embeddings).to("cpu").numpy()

    return embeddings


def topk_similarities(embeddings: torch.Tensor, query: torch.Tensor, k=5):
  import torch

  # Assume it's already normalized
  # print("query shape:", query.shape)
  flat_query = query.flatten(end_dim=-2)
  similarities = embeddings.matmul(flat_query.transpose(0, 1))
  # similarities = similarities.reshape(query_batch_shape + (-1,))
  # print("sim shape", similarities.shape)
  # print('sim shape', similarities.shape)
  return torch.topk(similarities, k, largest=True, sorted=True)


def imgslices(img, num_slices: int) -> list[PILImageType]:
  slices: list[PILImageType] = []
  slice_width = img.width / num_slices
  slice_height = img.height / num_slices

  # Slice the image into num_slices x num_slices
  for i in range(num_slices):
    for j in range(num_slices):
      x_start = i * slice_width
      y_start = j * slice_height
      slices.append(img.crop((x_start, y_start, x_start + slice_width, y_start + slice_height)))
  return slices


def stitch_slices(slices: list[PILImageType], target_size: tuple[int, int]) -> PILImageType:
  target_num_slices = int(len(slices) ** 0.5)
  target_width, target_height = target_size
  resize_slice_width = math.ceil(target_width / target_num_slices)
  resize_slice_height = math.ceil(target_height / target_num_slices)

  img = PILImage.new(
    "RGB", target_size
  )  # (slice_width * target_num_slices, slice_height * target_num_slices))
  for i, slice in enumerate(slices):
    # x = floor(i / target_num_slices)
    x, y = divmod(i, target_num_slices)

    # Resize the slice so the width or height matches the target size, depending on which is larger.
    # If the
    if slice.width / resize_slice_width < slice.height / resize_slice_height:
      resized_slice = slice.resize(
        (resize_slice_width, math.floor(slice.height * resize_slice_width / slice.width))
      )
      # # Center crop the resized slice.
      # resized_slice = resized_slice.crop(
      #   (
      #     0,
      #     (resized_slice.height - resize_slice_height) // 2,
      #     resize_slice_width,
      #     (resized_slice.height + resize_slice_height) // 2,
      #   )
      # )
    else:
      resized_slice = slice.resize(
        (math.floor(slice.width * resize_slice_height / slice.height), resize_slice_height)
      )
      # Center crop the resized slice.
      # resized_slice = resized_slice.crop(
      #   (
      #     (resized_slice.width - resize_slice_width) // 2,
      #     0,
      #     (resized_slice.width + resize_slice_width) // 2,
      #     resize_slice_height,
      #   )
      # )

    # y = i % target_num_slices
    img.paste(resized_slice, (x * resize_slice_width, y * resize_slice_height))
  return img


GCS_PROTOCOL = "gs://"
GCS_REGEX = re.compile(f"{GCS_PROTOCOL}(.*?)/(.*)")


@functools.cache
def _get_storage_client(thread_id: Optional[int] = None) -> storage.Client:
  # The storage client is not thread safe so we use a thread_id to make sure each thread gets a
  # separate storage client.
  del thread_id
  from google.oauth2 import service_account

  service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
  credentials = service_account.Credentials.from_service_account_info(service_account_info)
  return storage.Client(credentials=credentials)


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


TARGET_NUM_SLICES = 10
SOURCE_NUM_SLICES = 5


@stub.local_entrypoint()
def main() -> None:
  """Test the collage."""
  from datasets import load_dataset

  dataset = load_dataset("Francesco/people-in-paintings")
  target_image = dataset["test"]["image"][15]
  # source_images = [target_image]
  source_images = dataset["test"]["image"][50:100]

  collage = Collage()

  layers = [
    # "conv1",
    # "layer1.0.conv1",
    # "layer2.0.conv1",
    # "layer3.0.conv1",
    # "layer4.0.conv1",
    "layer4.1.add",
    # "avgpool",
    # "fc",
  ]
  for layer in layers:
    collage_response = CollageResponse.model_validate(
      collage.collage.remote(
        CollageRequest(
          target_image=target_image,
          target_num_slices=TARGET_NUM_SLICES,
          upscale_target=5.0,
          source_images=source_images,
          source_num_slices=SOURCE_NUM_SLICES,
          resnet_layer=layer,
          diversity=0.2,
        )
      ),
    )

    collage_response.collage.show(title=f"{TARGET_NUM_SLICES}x{SOURCE_NUM_SLICES} {layer} collage")
