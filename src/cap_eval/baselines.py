"""Reference-based NLG baselines comparing the candidate answer to the gold answer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import BaselineConfig

logger = logging.getLogger(__name__)

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _ensure_nltk_resources() -> None:
    import nltk

    for name, resource in _NLTK_RESOURCES.items():
        if _nltk_has(nltk, resource):
            continue
        logger.info("Downloading NLTK resource %s", name)
        if not nltk.download(name, quiet=True):
            raise RuntimeError(
                f"Could not download NLTK resource {name!r}. If this is an SSL certificate "
                "error on macOS, run 'Install Certificates.command' from your Python folder, "
                f"or install it manually: python -m nltk.downloader {name}"
            )


def _nltk_has(nltk, resource: str) -> bool:
    for candidate in (resource, f"{resource}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            pass
    return False


def _tokenize(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    return word_tokenize(text.lower())


def score_bleu(df: pd.DataFrame) -> pd.DataFrame:
    """Smoothed sentence BLEU (NLTK, method1) on lowercased word tokens."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    _ensure_nltk_resources()
    smoother = SmoothingFunction().method1

    def bleu(ref: str, hyp: str) -> float:
        hyp_tokens = _tokenize(hyp)
        if not hyp_tokens:
            return 0.0
        return sentence_bleu([_tokenize(ref)], hyp_tokens, smoothing_function=smoother)

    return _pairwise(df, "bleu", bleu)


def score_rouge_l(df: pd.DataFrame) -> pd.DataFrame:
    """ROUGE-L F-measure with Porter stemming."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _pairwise(df, "rouge_l", lambda ref, hyp: scorer.score(ref, hyp)["rougeL"].fmeasure)


def score_meteor(df: pd.DataFrame) -> pd.DataFrame:
    from nltk.translate.meteor_score import meteor_score

    _ensure_nltk_resources()

    def meteor(ref: str, hyp: str) -> float:
        ref_tokens, hyp_tokens = _tokenize(ref), _tokenize(hyp)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return meteor_score([ref_tokens], hyp_tokens)

    return _pairwise(df, "meteor", meteor)


def score_bertscore(df: pd.DataFrame, config: BaselineConfig, device: str) -> pd.DataFrame:
    """BERTScore F1 (no baseline rescaling)."""
    from bert_score import score

    refs, hyps = _refs_and_hyps(df)
    _, _, f1 = score(
        hyps,
        refs,
        model_type=config.bertscore_model,
        lang="en",
        batch_size=config.bertscore_batch_size,
        device=device,
        verbose=False,
    )
    return pd.DataFrame({"bertscore": f1.numpy()}, index=df.index)


def score_comet(df: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """COMET with the question as source, candidate as MT output and gold answer as reference."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(config.comet_model))
    refs, hyps = _refs_and_hyps(df)
    samples = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(df["question"].astype(str), hyps, refs)
    ]
    output = model.predict(samples, batch_size=config.comet_batch_size, gpus=config.comet_gpus)
    return pd.DataFrame({"comet": np.asarray(output.scores)}, index=df.index)


def _refs_and_hyps(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return df["gold_answer"].astype(str).tolist(), df["answer"].astype(str).tolist()


def _pairwise(df: pd.DataFrame, name: str, fn) -> pd.DataFrame:
    refs, hyps = _refs_and_hyps(df)
    return pd.DataFrame({name: [fn(r, h) for r, h in zip(refs, hyps)]}, index=df.index)
