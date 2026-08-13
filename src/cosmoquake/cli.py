"""Command-line interface.

    cosmoquake train                     retrain the classifier
    cosmoquake predict trace.mseed       classify one or more records
    cosmoquake catalog data/ -o out.csv  score a directory into a catalog
    cosmoquake info                      show what the packaged model knows
    cosmoquake serve                     run the web service
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from cosmoquake import __version__
from cosmoquake.detection import (
    DEFAULT_LTA_SECONDS,
    DEFAULT_STA_SECONDS,
    DEFAULT_THRESHOLD_OFF,
    DEFAULT_THRESHOLD_ON,
)
from cosmoquake.model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    ModelError,
    load_model,
    train_model,
)
from cosmoquake.pipeline import Prediction, analyze
from cosmoquake.waveform import CSV_SUFFIXES, MSEED_SUFFIXES, WaveformError

#: Files the catalog command will pick up when handed a directory.
READABLE_SUFFIXES = frozenset(CSV_SUFFIXES + MSEED_SUFFIXES)


def _add_detector_options(parser: argparse.ArgumentParser) -> None:
    """Attach the STA/LTA tuning flags shared by predict and catalog."""
    group = parser.add_argument_group("STA/LTA detector")
    group.add_argument(
        "--sta", type=float, default=DEFAULT_STA_SECONDS, help="short window, seconds"
    )
    group.add_argument(
        "--lta", type=float, default=DEFAULT_LTA_SECONDS, help="long window, seconds"
    )
    group.add_argument(
        "--thr-on", type=float, default=DEFAULT_THRESHOLD_ON, help="trigger-on ratio"
    )
    group.add_argument(
        "--thr-off", type=float, default=DEFAULT_THRESHOLD_OFF, help="trigger-off ratio"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the fully configured argument parser."""
    parser = argparse.ArgumentParser(
        prog="cosmoquake",
        description="Detect and classify seismic events on the Moon and Mars.",
    )
    parser.add_argument("--version", action="version", version=f"cosmoquake {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the classifier")
    train.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--trees", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--class-weight",
        choices=("none", "balanced", "balanced_subsample"),
        default="balanced_subsample",
        help="rebalancing strategy for the forest (default: balanced_subsample)",
    )
    train.add_argument(
        "--no-eval", action="store_true", help="skip cross-validation for a faster run"
    )

    predict = subparsers.add_parser("predict", help="classify seismic records")
    predict.add_argument("paths", type=Path, nargs="+", help=".csv or .mseed records")
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument(
        "--detect-with",
        type=Path,
        default=None,
        help="companion file to run the trigger on (single-input only)",
    )
    predict.add_argument("--json", action="store_true", help="emit JSON instead of text")
    _add_detector_options(predict)

    catalog = subparsers.add_parser("catalog", help="score a directory into a CSV catalog")
    catalog.add_argument("directory", type=Path)
    catalog.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    catalog.add_argument("-o", "--out", type=Path, default=Path("catalog.csv"))
    catalog.add_argument(
        "--pattern", default="*", help="glob applied to filenames, e.g. '*.mseed'"
    )
    _add_detector_options(catalog)

    info = subparsers.add_parser("info", help="describe the trained model")
    info.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    info.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="run the web service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def format_prediction(prediction: Prediction) -> str:
    """Render one prediction as an aligned text block."""
    detection = prediction.detection
    arrival = (
        f"{detection.arrival_time:,.1f} s"
        if detection.triggered
        else "not triggered (whole record analysed)"
    )
    lines = [
        f"{prediction.source}",
        f"  event         {prediction.label}  ({prediction.description})",
        f"  confidence    {prediction.confidence:.1%}",
        f"  arrival       {arrival}",
        f"  record        {prediction.n_samples:,} samples, "
        f"{prediction.duration:,.0f} s @ {prediction.sampling_rate:.2f} Hz",
        f"  downlink      {prediction.raw_bytes:,} B -> "
        f"{prediction.compression_ratio:,.0f}x smaller",
        "  features",
    ]
    lines.extend(
        f"    {name:<18}{value:.6g}" for name, value in prediction.features.to_dict().items()
    )
    lines.append("  probabilities")
    lines.extend(
        f"    {label:<18}{score:.1%}" for label, score in prediction.probabilities.items()
    )
    if prediction.warning:
        lines.append(f"  WARNING       {prediction.warning}")
    return "\n".join(lines)


def iter_records(directory: Path, pattern: str) -> Iterable[Path]:
    """Yield readable seismic files under ``directory``, sorted by name.

    Only one file per event is kept: when a record exists as both ``.csv``
    and ``.mseed``, the CSV wins, since it carries explicit timestamps.
    """
    candidates = [
        path
        for path in sorted(directory.rglob(pattern))
        if path.is_file() and path.suffix.lower() in READABLE_SUFFIXES
    ]
    seen: set[Path] = set()
    for path in sorted(candidates, key=lambda p: (p.suffix.lower() not in CSV_SUFFIXES,)):
        stem = path.with_suffix("")
        if stem in seen:
            continue
        seen.add(stem)
        yield path


def _cmd_train(args: argparse.Namespace) -> int:
    print(f"Training on {args.data} ...")
    model = train_model(
        dataset_path=args.data,
        n_estimators=args.trees,
        random_state=args.seed,
        evaluate=not args.no_eval,
        class_weight=None if args.class_weight == "none" else args.class_weight,
    )
    path = model.save(args.out)

    metadata = model.metadata
    print(f"  samples       {metadata['n_samples']}")
    print(f"  classes       {', '.join(f'{k}={v}' for k, v in metadata['class_counts'].items())}")
    scores = metadata.get("cross_validation")
    if scores:
        print(
            f"  {scores['n_splits']}-fold CV   accuracy {scores['accuracy']:.1%} "
            f"(majority baseline {scores['majority_baseline']:.1%}), "
            f"balanced accuracy {scores['balanced_accuracy']:.1%}"
        )
        for label, values in scores["per_class"].items():
            auc = values.get("ranking_auc")
            auc_text = f", AUC {auc:.2f}" if auc is not None else ""
            print(
                f"    {label:<14}recall {values['recall']:.2f} "
                f"(n={values['support']}){auc_text}"
            )
    elif metadata.get("note"):
        print(f"  CV            {metadata['note']}")
    print(f"Saved model to {path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.detect_with and len(args.paths) > 1:
        print("--detect-with only applies to a single input file", file=sys.stderr)
        return 2

    model = load_model(args.model)
    results: list[Prediction] = []
    failures = 0

    for path in args.paths:
        try:
            results.append(
                analyze(
                    path,
                    model=model,
                    detection_path=args.detect_with,
                    sta_seconds=args.sta,
                    lta_seconds=args.lta,
                    threshold_on=args.thr_on,
                    threshold_off=args.thr_off,
                )
            )
        except (WaveformError, OSError) as error:
            failures += 1
            print(f"{path}: {error}", file=sys.stderr)

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print("\n".join(format_prediction(result) for result in results))

    return 1 if failures and not results else 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    import pandas as pd

    model = load_model(args.model)
    rows: list[dict[str, object]] = []
    failures = 0

    for path in iter_records(args.directory, args.pattern):
        try:
            result = analyze(
                path,
                model=model,
                sta_seconds=args.sta,
                lta_seconds=args.lta,
                threshold_on=args.thr_on,
                threshold_off=args.thr_off,
            )
        except (WaveformError, OSError) as error:
            failures += 1
            print(f"{path}: {error}", file=sys.stderr)
            continue

        # Column names follow the challenge's required output catalog format.
        rows.append(
            {
                "filename": path.name,
                "time_rel(sec)": round(result.detection.arrival_time, 3),
                "mq_type": result.label,
                "confidence": round(result.confidence, 4),
                "triggered": result.detection.triggered,
                **result.features.to_dict(),
            }
        )
        print(f"{path.name}: {result.label} ({result.confidence:.0%})")

    if not rows:
        print(f"no readable records under {args.directory}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    skipped = f" ({failures} skipped)" if failures else ""
    print(f"\nWrote {len(rows)} events to {args.out}{skipped}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    payload = {
        "model_path": str(args.model),
        "classes": list(model.classes),
        "features": list(model.feature_names),
        **model.metadata,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Model         {args.model}")
    print(f"Trained       {model.metadata.get('trained_at', 'unknown')}")
    print(f"Dataset       {model.metadata.get('dataset', 'unknown')} "
          f"({model.metadata.get('n_samples', '?')} samples)")
    print(f"Classes       {', '.join(model.classes)}")
    scores = model.metadata.get("cross_validation")
    if scores:
        print(f"{scores['n_splits']}-fold CV     accuracy {scores['accuracy']:.1%} "
              f"(majority baseline {scores['majority_baseline']:.1%})")
        print(f"              balanced accuracy {scores['balanced_accuracy']:.1%}, "
              f"macro F1 {scores['macro_f1']:.3f}")
        for label, values in scores["per_class"].items():
            auc = values.get("ranking_auc")
            auc_text = f"  AUC {auc:.2f}" if auc is not None else ""
            print(f"  {label:<14}precision {values['precision']:.2f}  "
                  f"recall {values['recall']:.2f}  n={values['support']}{auc_text}")
    print("Feature importance")
    for name, value in model.feature_importances().items():
        bar = "#" * round(value * 40)
        print(f"  {name:<20}{value:.3f} {bar}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "cosmoquake.web.app:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    handlers = {
        "train": _cmd_train,
        "predict": _cmd_predict,
        "catalog": _cmd_catalog,
        "info": _cmd_info,
        "serve": _cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except (ModelError, WaveformError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
