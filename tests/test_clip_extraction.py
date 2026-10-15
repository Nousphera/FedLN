"""Keep the CLIP PIL preprocessing and feature manifest usable with training dependencies."""
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from extract_features import CLIP_REVISION, array_digest, extract_clip


def test_clip_extraction_preserves_pil_pixels_and_projected_features(tmp_path, monkeypatch):
    images = np.random.default_rng(42).integers(0, 256, (2, 31, 47, 3), dtype=np.uint8)
    ids = np.array([9, 3])
    digest = array_digest(images)
    source = tmp_path / "images.npz"
    np.savez(source, images=images, ids=ids, dataset="fixture", source_sha256=digest)

    processor = CLIPImageProcessor()
    expected = processor(images=[Image.fromarray(im) for im in images], return_tensors="pt")["pixel_values"]
    observed = []

    class Encoder:
        def eval(self):
            return self

        def __call__(self, *, pixel_values):
            observed.append(pixel_values)
            return SimpleNamespace(image_embeds=torch.full((len(pixel_values), 512), 2.0))

    def load_processor(name, *, revision, local_files_only):
        assert name == "openai/clip-vit-base-patch32"
        assert revision == CLIP_REVISION
        assert local_files_only is True
        return processor

    monkeypatch.setattr(CLIPImageProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(CLIPVisionModelWithProjection, "from_pretrained", lambda *args, **kwargs: Encoder())
    features, actual_ids, dataset, source_hash, teacher = extract_clip(source, batch_size=1, local_files_only=True)

    torch.testing.assert_close(torch.cat(observed), expected, rtol=0, atol=0)
    np.testing.assert_array_equal(features, np.full((2, 512), 2.0))
    np.testing.assert_array_equal(actual_ids, ids)
    assert (dataset, source_hash) == ("fixture", digest)
    assert teacher["processor_class"] == "CLIPImageProcessor"
    assert teacher["processor_backend"] == "pil"
    assert teacher["l2_normalized"] is False
