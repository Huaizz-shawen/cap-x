from __future__ import annotations

import numpy as np
from PIL import Image

from capx.integrations.vision import candidate_rerank as rerank
from capx.llm.client import ModelQueryArgs


def test_should_use_vlm_candidate_selection_for_relation_phrase() -> None:
    candidates = [
        {"score": 0.25, "mask": np.zeros((8, 8), dtype=bool), "box": [1, 1, 3, 3]},
        {"score": 0.24, "mask": np.zeros((8, 8), dtype=bool), "box": [4, 4, 6, 6]},
    ]
    assert rerank.should_use_vlm_candidate_selection(
        "the black bowl between the plate and the ramekin",
        candidates,
    )


def test_parse_candidate_choice() -> None:
    assert rerank.parse_candidate_choice("2", 4) == 2
    assert rerank.parse_candidate_choice("Candidate 3 is best.", 4) == 3
    assert rerank.parse_candidate_choice("0", 4) == 0
    assert rerank.parse_candidate_choice("5", 4) is None


def test_select_mask_candidate_with_vlm(monkeypatch) -> None:
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    mask_a = np.zeros((32, 32), dtype=bool)
    mask_a[2:10, 2:10] = True
    mask_b = np.zeros((32, 32), dtype=bool)
    mask_b[12:20, 12:20] = True
    candidates = [
        {"score": 0.25, "mask": mask_a, "box": [2, 2, 10, 10]},
        {"score": 0.24, "mask": mask_b, "box": [12, 12, 20, 20]},
    ]

    monkeypatch.setattr(
        "capx.integrations.vision.candidate_rerank.query_model",
        lambda args, prompt: {"content": "2", "reasoning": None},
    )

    selected, info = rerank.select_mask_candidate_with_vlm(
        ModelQueryArgs(
            model="gemini-3-pro-preview",
            server_url="http://example.test/v1/chat/completions",
        ),
        image=image,
        task_goal="Pick the black bowl between the plate and the ramekin",
        object_query="the black bowl between the plate and the ramekin",
        candidates=candidates,
    )

    assert selected is candidates[1]
    assert info["selected_index"] == 2


def test_build_candidate_selection_image_limits_candidates() -> None:
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    candidates = []
    for idx in range(6):
        mask = np.zeros((16, 16), dtype=bool)
        mask[idx : idx + 2, idx : idx + 2] = True
        candidates.append({"score": 1.0 - idx * 0.1, "mask": mask, "box": [idx, idx, idx + 2, idx + 2]})

    composite, ranked = rerank.build_candidate_selection_image(image, candidates, max_candidates=4)

    assert len(ranked) == 4
    assert composite.size == (32, 32)
