"""The CAP metric: bidirectional NLI between gold and candidate statements.

For a gold statement (premise) and a candidate statement (hypothesis)::

    dir(a -> b) = P(entailment | a, b) + neutral_weight * P(neutral | a, b)
    CAP         = forward_weight * dir(premise -> hypothesis)
                + (1 - forward_weight) * dir(hypothesis -> premise)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import CapConfig

logger = logging.getLogger(__name__)

NLI_CLASSES = ("entailment", "neutral", "contradiction")
_LABEL_PATTERNS = {"entailment": "entail", "neutral": "neutral", "contradiction": "contrad"}


def directional_score(probs: np.ndarray, neutral_weight: float) -> np.ndarray:
    """``probs`` has columns ordered as ``NLI_CLASSES``."""
    return probs[:, 0] + neutral_weight * probs[:, 1]


def cap_score(
    forward_probs: np.ndarray,
    reverse_probs: np.ndarray,
    neutral_weight: float,
    forward_weight: float,
) -> np.ndarray:
    forward = directional_score(forward_probs, neutral_weight)
    reverse = directional_score(reverse_probs, neutral_weight)
    return forward_weight * forward + (1 - forward_weight) * reverse


class NLIModel:
    """Batched NLI cross-encoder returning probabilities in ``NLI_CLASSES`` order."""

    def __init__(self, config: CapConfig, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.config = config
        self.device = device
        dtype = torch.float16 if (config.fp16 and device == "cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.nli_model, revision=config.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model, revision=config.revision, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.class_ids = self._resolve_class_ids(self.model.config.id2label)
        logger.info("Loaded %s (%s) on %s, labels %s", config.nli_model, dtype, device,
                    self.model.config.id2label)

    @staticmethod
    def _resolve_class_ids(id2label: dict[int, str]) -> list[int]:
        ids = []
        for name in NLI_CLASSES:
            matches = [i for i, label in id2label.items() if _LABEL_PATTERNS[name] in label.lower()]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify the {name!r} class in {id2label}")
            ids.append(int(matches[0]))
        return ids

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str], desc: str) -> np.ndarray:
        torch = self._torch
        batch_size = self.config.batch_size
        out = []
        for start in tqdm(range(0, len(premises), batch_size), desc=desc):
            enc = self.tokenizer(
                list(premises[start : start + batch_size]),
                list(hypotheses[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, self.class_ids].cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(NLI_CLASSES)))


def score_cap(df: pd.DataFrame, config: CapConfig, device: str) -> pd.DataFrame:
    """CAP plus the underlying NLI probabilities, indexed like ``df``."""
    model = NLIModel(config, device)
    premises = df["premise"].astype(str).tolist()
    hypotheses = df["hypothesis"].astype(str).tolist()
    forward = model.predict(premises, hypotheses, desc="NLI premise->hypothesis")
    reverse = model.predict(hypotheses, premises, desc="NLI hypothesis->premise")

    out = pd.DataFrame(index=df.index)
    out["cap"] = cap_score(forward, reverse, config.neutral_weight, config.forward_weight)
    for direction, probs in (("forward", forward), ("reverse", reverse)):
        out[f"cap_{direction}"] = directional_score(probs, config.neutral_weight)
        for j, name in enumerate(NLI_CLASSES):
            out[f"p_{name}_{direction}"] = probs[:, j]
        out[f"nli_label_{direction}"] = np.asarray(NLI_CLASSES)[probs.argmax(axis=1)]
    return out
