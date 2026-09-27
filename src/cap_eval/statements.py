"""The question + answer -> declarative statement model (mT5).

Training and generation share ``format_inputs`` and ``normalize_text`` so the
model always sees inputs built the same way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import GenerationConfig, StatementModelConfig
from .data import load_statement_training_data, normalize_text

logger = logging.getLogger(__name__)


def format_inputs(questions: pd.Series, answers: pd.Series, template: str) -> list[str]:
    return [
        template.format(question=q, answer=a)
        for q, a in zip(normalize_text(questions), normalize_text(answers))
    ]


def split_statement_data(
    df: pd.DataFrame, config: StatementModelConfig, seed: int
) -> dict[str, pd.DataFrame]:
    """Train / validation / test split; the holdout is divided evenly."""
    from sklearn.model_selection import train_test_split

    stratify = config.stratify_column or None
    train, holdout = train_test_split(
        df,
        test_size=config.holdout_fraction,
        random_state=seed,
        stratify=df[stratify] if stratify else None,
    )
    val, test = train_test_split(
        holdout,
        test_size=0.5,
        random_state=seed,
        stratify=holdout[stratify] if stratify else None,
    )
    return {"train": train, "validation": val, "test": test}


def train_statement_model(
    config: StatementModelConfig, statements_csv: str, output_dir: Path, seed: int
) -> Path:
    """Fine-tune the statement model; returns the directory of the best model."""
    import sacrebleu
    import torch
    from datasets import Dataset, DatasetDict
    from rouge_score import rouge_scorer
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_statement_training_data(statements_csv)
    df["input_text"] = format_inputs(df["question"], df["answer"], config.input_template)
    df["target_text"] = df["statement"]
    splits = split_statement_data(df, config, seed)
    logger.info("Split sizes: %s", {k: len(v) for k, v in splits.items()})

    # Record exactly which source rows went to which split.
    pd.concat(
        [pd.DataFrame({"source_row": frame.index, "split": name}) for name, frame in splits.items()]
    ).sort_values("source_row").to_csv(output_dir / "splits.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model, revision=config.revision)

    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], max_length=config.max_input_length, truncation=True)
        labels = tokenizer(
            text_target=batch["target_text"], max_length=config.max_target_length, truncation=True
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    columns = ["input_text", "target_text"]
    tokenized = DatasetDict(
        {name: Dataset.from_pandas(frame[columns].reset_index(drop=True)) for name, frame in splits.items()}
    ).map(preprocess, batched=True, remove_columns=columns)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [s.strip() for s in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [s.strip() for s in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        rouge_l = np.mean(
            [rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(decoded_labels, decoded_preds)]
        )
        return {"bleu": round(bleu, 2), "rougeL": round(rouge_l * 100, 2)}

    t = config.training
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=seed,
        data_seed=seed,
        full_determinism=t.full_determinism,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        max_grad_norm=t.max_grad_norm,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.generation.num_beams,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=True,
        fp16=False,
        bf16=use_bf16,
        logging_steps=t.logging_steps,
        save_total_limit=t.save_total_limit,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t.early_stopping_patience)],
    )
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    logger.info("Test metrics: %s", test_metrics)
    return final_dir


class StatementGenerator:
    def __init__(self, model_dir: str | Path, config: GenerationConfig, device: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.model.to(device).eval()

    def generate(self, inputs: Sequence[str], desc: str) -> list[str]:
        cfg = self.config
        outputs: list[str] = []
        for start in tqdm(range(0, len(inputs), cfg.batch_size), desc=desc):
            enc = self.tokenizer(
                list(inputs[start : start + cfg.batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_input_length,
            ).to(self.device)
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=cfg.max_new_tokens,
                    num_beams=cfg.num_beams,
                    do_sample=False,
                    early_stopping=True,
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


def generate_premise_hypothesis(
    df: pd.DataFrame, model_dir: str | Path, config: StatementModelConfig, device: str
) -> pd.DataFrame:
    """Add ``premise`` (from the gold answer) and ``hypothesis`` (from the candidate)."""
    generator = StatementGenerator(model_dir, config.generation, device)
    template = config.input_template
    out = df.copy()
    out["premise"] = generator.generate(
        format_inputs(df["question"], df["gold_answer"], template), desc="Premises"
    )
    out["hypothesis"] = generator.generate(
        format_inputs(df["question"], df["answer"], template), desc="Hypotheses"
    )
    return out
