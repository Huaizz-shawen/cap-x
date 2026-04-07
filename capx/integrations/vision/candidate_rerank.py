from __future__ import annotations

import base64
import io
import os
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from capx.llm.client import ModelQueryArgs, VLM_MODELS, query_model

RELATION_KEYWORDS = (
    " between ",
    " next to ",
    " left of ",
    " right of ",
    " in front of ",
    " behind ",
    " on top of ",
    " under ",
    " near ",
    " closest to ",
    " farthest from ",
)
DEFAULT_MAX_CANDIDATES = 4
DEFAULT_SCORE_MARGIN = 0.03


def load_candidate_selection_args_from_env() -> ModelQueryArgs | None:
    model = os.getenv("CAPX_OBJECT_RERANK_MODEL") or os.getenv("CAPX_VLM_RERANK_MODEL")
    server_url = os.getenv("CAPX_OBJECT_RERANK_SERVER_URL") or os.getenv("CAPX_VLM_RERANK_SERVER_URL")
    api_key = os.getenv("CAPX_OBJECT_RERANK_API_KEY") or os.getenv("CAPX_VLM_RERANK_API_KEY")
    if not model or not server_url:
        return None
    if model not in VLM_MODELS:
        return None
    return ModelQueryArgs(
        model=model,
        server_url=server_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=64,
        reasoning_effort="low",
        debug=False,
    )


def should_use_vlm_candidate_selection(
    text_prompt: str,
    candidates: list[dict[str, Any]],
    *,
    score_margin_threshold: float = DEFAULT_SCORE_MARGIN,
) -> bool:
    if len(candidates) <= 1:
        return False
    lowered = f" {text_prompt.lower().strip()} "
    if any(keyword in lowered for keyword in RELATION_KEYWORDS):
        return True
    if len(candidates) >= 2:
        top = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        margin = float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0))
        if margin < score_margin_threshold:
            return True
    return False


def parse_candidate_choice(text: str, num_candidates: int) -> int | None:
    matches = re.findall(r"\d+", text)
    for match in matches:
        choice = int(match)
        if 0 <= choice <= num_candidates:
            return choice
    return None


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    return mask_bool.astype(np.uint8)


def build_candidate_selection_image(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    ranked = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:max_candidates]
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tiles: list[Image.Image] = []
    tile_w, tile_h = image.size
    overlay_color = np.array([64, 220, 96], dtype=np.uint8)

    for idx, candidate in enumerate(ranked, start=1):
        overlay = base.copy()
        mask = _mask_to_uint8(candidate["mask"]).astype(bool)
        if mask.shape[:2] == overlay.shape[:2]:
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.45 + overlay_color.astype(np.float32) * 0.55
            ).astype(np.uint8)
        tile = Image.fromarray(overlay, mode="RGB")
        draw = ImageDraw.Draw(tile)
        box = candidate.get("box")
        if box is not None and len(box) == 4:
            draw.rectangle(box, outline=(255, 208, 0), width=3)
        score = float(candidate.get("score", 0.0))
        draw.rectangle((8, 8, min(tile_w - 8, 250), 40), fill=(0, 0, 0))
        draw.text((14, 14), f"Candidate {idx}  score={score:.3f}", fill=(255, 255, 255))
        tiles.append(tile)

    if not tiles:
        return image, []

    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(24, 24, 24))
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas, ranked


def select_mask_candidate_with_vlm(
    args: ModelQueryArgs,
    *,
    image: Image.Image,
    task_goal: str,
    object_query: str,
    candidates: list[dict[str, Any]],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    composite_image, ranked_candidates = build_candidate_selection_image(
        image, candidates, max_candidates=max_candidates
    )
    if not ranked_candidates:
        return None, {"reason": "no_candidates"}

    image_data_url = _image_to_data_url(composite_image)
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Select the best segmentation candidate for a robot manipulation task. "
                "Use the task goal and object query literally. Respect attributes, color, "
                "state, and spatial relations. If none of the shown candidates match, answer 0."
            ),
        },
        {"type": "text", "text": f"Task goal: {task_goal}"},
        {"type": "text", "text": f"Object query: {object_query}"},
        {
            "type": "text",
            "text": (
                f"There are {len(ranked_candidates)} candidate masks shown. "
                "Respond with only the integer candidate index (1-based), or 0 if none match."
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    prompt = [
        {
            "role": "system",
            "content": (
                "You are choosing the single best segmentation candidate for a robot. "
                "Return only a number."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    response = query_model(args, prompt)
    raw_text = response["content"]
    choice = parse_candidate_choice(raw_text, len(ranked_candidates))
    if choice is None or choice == 0:
        return None, {
            "reason": "invalid_or_none",
            "raw_response": raw_text,
            "candidate_count": len(ranked_candidates),
        }
    return ranked_candidates[choice - 1], {
        "reason": "selected",
        "raw_response": raw_text,
        "selected_index": choice,
        "candidate_count": len(ranked_candidates),
    }
