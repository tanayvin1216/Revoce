"""
Scene description (the VLM), behind a clean swappable interface.

Default backend: mlx-vlm running Qwen2.5-VL-3B (4-bit) natively on the Mac.
This is PERCEPTION — it produces a short scene description that flows into the
shared ContextState. It does NOT call rank_vocabulary or the UI. Swap the model
or runtime (e.g. OpenVINO on Intel) with a ONE-LINE change in config.py.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod


class SceneDescriber(ABC):
    @abstractmethod
    def describe(self, jpeg_bytes: bytes) -> str:
        ...


class MlxVlmSceneDescriber(SceneDescriber):
    def __init__(self, model_id: str, prompt: str, max_tokens: int) -> None:
        # Lazy heavy imports / model load (happens on the vision worker thread).
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self._model, self._processor = load(model_id)
        self._model_config = load_config(model_id)
        self._prompt = prompt
        self._max_tokens = max_tokens

    def describe(self, jpeg_bytes: bytes) -> str:
        from PIL import Image
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        # NOTE: mlx-vlm's generate() signature has shifted across releases. If a
        # version mismatch shows up, this is the ONE spot to adjust.
        formatted = self._apply_chat_template(
            self._processor, self._model_config, self._prompt, num_images=1
        )
        out = self._generate(
            self._model, self._processor, formatted, [image],
            max_tokens=self._max_tokens, verbose=False,
        )
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        return text.strip().replace("\n", " ")


class MockSceneDescriber(SceneDescriber):
    """Scripted describer for dev without the VLM. Cycles canned scenes so each
    gated VLM 'call' is visibly distinct in the logs."""

    _SCENES = [
        "a kitchen counter with a glass and a water pitcher",
        "someone sitting across the table holding a menu",
        "a living room with a TV showing a sports game",
        "an open front door with a person arriving",
    ]

    def __init__(self) -> None:
        self._i = 0

    def describe(self, jpeg_bytes: bytes) -> str:
        scene = self._SCENES[self._i % len(self._SCENES)]
        self._i += 1
        return scene


def make_scene_describer(cfg) -> SceneDescriber:
    """Factory — the one place the VLM runtime is selected."""
    name = cfg.SCENE_DESCRIBER
    if name == "mlx-vlm":
        try:
            return MlxVlmSceneDescriber(cfg.VLM_MODEL, cfg.VLM_PROMPT, cfg.VLM_MAX_TOKENS)
        except Exception as e:  # lib/model missing -> degrade gracefully
            print(f"[perception] mlx-vlm unavailable ({e}); using MockSceneDescriber")
            return MockSceneDescriber()
    if name == "mock":
        return MockSceneDescriber()
    # if name == "openvino": return OpenVinoSceneDescriber(cfg)   # Intel port, later
    raise ValueError(f"unknown SCENE_DESCRIBER={name!r}")
