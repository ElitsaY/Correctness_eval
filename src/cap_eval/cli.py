"""Command-line entry point: ``cap-eval <command>``.

    cap-eval train-statements      fine-tune the question+answer -> statement model
    cap-eval generate-statements   write premise/hypothesis statements for QA data
    cap-eval score                 compute CAP and baseline scores, one file per metric
    cap-eval evaluate              benchmark the scores against human labels

Every command writes a ``*.manifest.json`` with the resolved config, input
hashes, package versions and git commit next to its outputs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import ROW_ID, load_correctness_data
from .reproducibility import read_manifest, resolve_device, set_seed, sha256_file, write_manifest

logger = logging.getLogger("cap_eval")

METRICS = ("cap", "bleu", "rouge_l", "meteor", "bertscore", "comet")
DEFAULT_CONFIG = "configs/default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cap-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set cap.batch_size=32 (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-statements", help="fine-tune the statement model")
    train.add_argument("--output-dir", type=Path, help="default: <output_dir>/statement_model")
    train.set_defaults(func=cmd_train_statements)

    generate = commands.add_parser("generate-statements", help="generate premise/hypothesis")
    generate.add_argument("--model-dir", type=Path, required=True)
    generate.add_argument("--input", help="default: data.correctness_csv")
    generate.add_argument("--output", type=Path, help="default: <output_dir>/statements.csv")
    generate.set_defaults(func=cmd_generate_statements)

    score = commands.add_parser("score", help="score QA pairs with CAP and baselines")
    score.add_argument("--metrics", nargs="+", choices=METRICS, help="default: evaluation.metrics")
    score.add_argument("--input", help="default: data.correctness_csv")
    score.add_argument("--scores-dir", type=Path, help="default: <output_dir>/scores")
    score.set_defaults(func=cmd_score)

    evaluate = commands.add_parser("evaluate", help="benchmark scores against human labels")
    evaluate.add_argument("--input", help="default: data.correctness_csv")
    evaluate.add_argument("--scores-dir", type=Path, help="default: <output_dir>/scores")
    evaluate.add_argument("--output-dir", type=Path, help="default: <output_dir>/evaluation")
    evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config, args.overrides)
    unknown = sorted(set(config.evaluation.metrics) - set(METRICS))
    if unknown:
        parser.error(f"evaluation.metrics contains unknown metrics {unknown}; known: {METRICS}")
    args.func(args, config)


def cmd_train_statements(args: argparse.Namespace, config: Config) -> None:
    from .statements import train_statement_model

    set_seed(config.seed, deterministic=config.statement_model.training.full_determinism)
    output_dir = args.output_dir or Path(config.output_dir) / "statement_model"
    final_dir = train_statement_model(
        config.statement_model, config.data.statements_csv, output_dir, config.seed
    )
    write_manifest(
        output_dir / "train.manifest.json",
        command="train-statements",
        config=config.to_dict(),
        inputs={"statements_csv": config.data.statements_csv},
        extra={"model_dir": str(final_dir)},
    )


def cmd_generate_statements(args: argparse.Namespace, config: Config) -> None:
    from .statements import generate_premise_hypothesis

    set_seed(config.seed)
    device = resolve_device(config.device)
    input_csv = args.input or config.data.correctness_csv
    output = args.output or Path(config.output_dir) / "statements.csv"

    df = load_correctness_data(
        input_csv, config.data.label_column, config.labels.severity, require_statements=False
    )
    out = generate_premise_hypothesis(df, args.model_dir, config.statement_model, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["label", "severity"]).to_csv(output, index=False)
    logger.info("Wrote %d rows to %s", len(out), output)
    write_manifest(
        output.with_suffix(".manifest.json"),
        command="generate-statements",
        config=config.to_dict(),
        inputs={"correctness_csv": input_csv},
        extra={"model_dir": str(args.model_dir), "device": device},
    )


def cmd_score(args: argparse.Namespace, config: Config) -> None:
    set_seed(config.seed)
    input_csv = args.input or config.data.correctness_csv
    scores_dir = args.scores_dir or Path(config.output_dir) / "scores"
    metrics = args.metrics or config.evaluation.metrics
    df = load_correctness_data(input_csv, config.data.label_column, config.labels.severity)
    scores_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        logger.info("Scoring %s on %d rows", metric, len(df))
        device = resolve_device(config.device) if metric in {"cap", "bertscore"} else "cpu"
        scores = _run_scorer(metric, df, config, device)
        scores.insert(0, ROW_ID, df[ROW_ID].to_numpy())
        scores.to_csv(scores_dir / f"{metric}.csv", index=False)
        write_manifest(
            scores_dir / f"{metric}.manifest.json",
            command=f"score {metric}",
            config=config.to_dict(),
            inputs={"correctness_csv": input_csv},
            extra={"metric": metric, "device": device, "rows": len(df)},
        )


def _run_scorer(metric: str, df: pd.DataFrame, config: Config, device: str) -> pd.DataFrame:
    from . import baselines

    if metric == "cap":
        from .cap import score_cap

        return score_cap(df, config.cap, device)
    if metric == "bleu":
        return baselines.score_bleu(df)
    if metric == "rouge_l":
        return baselines.score_rouge_l(df)
    if metric == "meteor":
        return baselines.score_meteor(df)
    if metric == "bertscore":
        return baselines.score_bertscore(df, config.baselines, device)
    if metric == "comet":
        return baselines.score_comet(df, config.baselines)
    raise ValueError(f"Unknown metric {metric!r}")


def cmd_evaluate(args: argparse.Namespace, config: Config) -> None:
    from .evaluation import evaluate, write_tables

    input_csv = args.input or config.data.correctness_csv
    scores_dir = args.scores_dir or Path(config.output_dir) / "scores"
    output_dir = args.output_dir or Path(config.output_dir) / "evaluation"

    df = load_correctness_data(input_csv, config.data.label_column, config.labels.severity)
    input_hash = sha256_file(input_csv)
    for metric in config.evaluation.metrics:
        _check_scored_on(scores_dir / f"{metric}.manifest.json", input_hash, metric)
        scores = pd.read_csv(scores_dir / f"{metric}.csv", usecols=[ROW_ID, metric])
        df = df.merge(scores, on=ROW_ID, how="left", validate="one_to_one")
        missing = int(df[metric].isna().sum())
        if missing:
            raise ValueError(f"{metric} scores are missing for {missing} rows; re-run `score`")

    result = evaluate(df, config.labels, config.evaluation, config.seed)
    write_tables(result, output_dir, config.evaluation.ci_level)
    write_manifest(
        output_dir / "evaluate.manifest.json",
        command="evaluate",
        config=config.to_dict(),
        inputs={"correctness_csv": input_csv}
        | {f"scores_{m}": scores_dir / f"{m}.csv" for m in config.evaluation.metrics},
        extra={"rows": len(df)},
    )
    print((output_dir / "summary.md").read_text())


def _check_scored_on(manifest_path: Path, input_hash: str, metric: str) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No scores for {metric!r}: run `cap-eval score --metrics {metric}`")
    scored_hash = read_manifest(manifest_path)["inputs"]["correctness_csv"]["sha256"]
    if scored_hash != input_hash:
        raise ValueError(
            f"{metric} scores were computed on a different input file; re-run `score`"
        )


if __name__ == "__main__":
    main()
