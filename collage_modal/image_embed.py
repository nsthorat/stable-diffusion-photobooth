"""Image embeddings."""

import functools
from typing import Optional

import numpy as np
from modal import Image, Stub, method
from pydantic import BaseModel, ConfigDict
from .utils import DebugTimer
from PIL.Image import Image as PILImageType

# from .batch_utils import batch_by_length_and_call, compress_docs, decompress_docs
# from .transformer_utils import download_model_to_folder, setup_model_device
# from .utils import DebugTimer, log


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
  # hf-transfer package gives maximum download speeds. No progress bar, but 700MB/s.
  .pip_install(
    "torch==2.2.1",
    "torchvision==0.17.1",
    "pydantic~=2.5",
    "fastapi~=0.108",
    "scikit-learn~=1.3",
    "pillow~=10.2.0",
  )
  .env({"TORCH_HOME": MODEL_DIR})
  .run_function(_download_model, timeout=60 * 20)
)

stub = Stub("image_embed", image=image)


class EmbedRequest(BaseModel):
  """Request to embed a list of documents."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  images: list[PILImageType]
  layer: Optional[str] = "flatten"
  # If set, the model will randomly project the embeddings to this dimension.
  random_projection_dim: Optional[int] = None
  # Seed for the random projection.
  projection_seed: int = PROJECTION_SEED


class EmbedResponse(BaseModel):
  """Response containing a list of embedding vectors."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  vectors: np.ndarray


@stub.cls(gpu="A10G", container_idle_timeout=300)
class ImageEmbed:
  """Embedding model from HuggingFace."""

  def __enter__(self) -> None:
    # Pre-load the model.
    from torchvision.models import resnet18
    from torchvision.models import feature_extraction
    from torchvision import transforms
    import torch

    # seed = 0
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #   torch.cuda.manual_seed_all(seed)
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

  @method()
  def embed(self, request_dict: dict) -> dict:
    """Embed documents and return a list of embedding vectors."""
    import torch

    request = EmbedRequest.model_validate(request_dict)
    if request.images:
      images = request.images
    else:
      raise ValueError("No images provided.")

    print("IMAGE", images[0])
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

    response = EmbedResponse(vectors=embeddings)
    return response.model_dump()


@stub.local_entrypoint()
def main() -> None:
  """Run the embed model locally."""
  from datasets import load_dataset

  dataset = load_dataset("Francesco/people-in-paintings")
  target_image = dataset["test"]["image"][15]
  # source_images = dataset["test"]["image"][0:100]
  image_embed = ImageEmbed()
  with DebugTimer("Embedding"):
    print(
      image_embed.embed.remote(
        EmbedRequest(
          images=[target_image], layer="avgpool", random_projection_dim=1024
        ).model_dump()
      )
    )

  # request = EmbedRequest(gzipped_docs=compress_docs(docs), random_projection_dim=10)
  # print(image_embed.embed.remote(request.model_dump()))
