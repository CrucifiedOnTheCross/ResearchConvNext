from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham_pipeline.methods import all_methods, get_method

DEFAULT_METHODS = [
    "ce",
    "weighted_ce",
    "focal",
    "logit_adjustment",
    "balanced_softmax",
    "supcon",
    "center",
    "proxy_anchor",
    "arcface",
]


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_train_command(
    python_bin: str,
    config: str,
    method: str,
    seed: int,
    epochs: int | None,
    root: str,
    extra_set: list[str],
) -> list[str]:
    cmd = [
        python_bin,
        "scripts/train.py",
        "--config",
        config,
        "--set",
        f"training.method={method}",
        "--set",
        f"seed={seed}",
        "--set",
        f"output.root={root}",
    ]

    if epochs is not None:
        cmd.extend(["--set", f"training.epochs={epochs}"])

    for item in extra_set:
        cmd.extend(["--set", item])

    return cmd


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run method × seed experiment grid."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Force one base config for every method; by default configs/<method>.yaml is used.",
    )

    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated method list, for example: "
            "ce,weighted_ce,focal,supcon,center,proxy_anchor"
        ),
    )

    parser.add_argument(
        "--seeds",
        default="42,52,62",
        help="Comma-separated seed list.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs. If omitted, config value is used.",
    )

    parser.add_argument(
        "--root",
        default="runs",
        help="Output root for experiment runs.",
    )

    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to call train.py and collect_results.py.",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help=(
            "Extra --set overrides passed to every train.py run. "
            "Example: --set model.compile=false"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Do not call collect_results.py after experiments.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue grid if one experiment fails.",
    )

    args = parser.parse_args()

    methods = parse_list(args.methods)
    unknown = sorted(set(methods) - set(all_methods()))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose {all_methods()}")
    seeds = [int(x) for x in parse_list(args.seeds)]

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    total = len(methods) * len(seeds)
    current = 0
    failures = []

    print("Experiment grid")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total}")
    print(f"Output root: {root}")

    for method in methods:
        for seed in seeds:
            current += 1

            print(
                f"\n=== Run {current}/{total}: method={method}, seed={seed} ===",
                flush=True,
            )

            cmd = build_train_command(
                python_bin=args.python_bin,
                config=args.config or str(Path("configs") / f"{method}.yaml"),
                method=method,
                seed=seed,
                epochs=args.epochs,
                root=args.root,
                extra_set=args.set,
            )

            try:
                run_command(cmd, dry_run=args.dry_run)
            except subprocess.CalledProcessError as error:
                failures.append(
                    {
                        "method": method,
                        "seed": seed,
                        "returncode": error.returncode,
                        "cmd": cmd,
                    }
                )

                print(
                    f"FAILED: method={method}, seed={seed}, "
                    f"returncode={error.returncode}",
                    file=sys.stderr,
                    flush=True,
                )

                if not args.continue_on_error:
                    raise

    if not args.no_collect:
        collect_cmd = [
            args.python_bin,
            "scripts/collect_results.py",
            "--root",
            args.root,
        ]

        run_command(collect_cmd, dry_run=args.dry_run)

    if failures:
        print("\nFailed runs:")
        for item in failures:
            print(
                f"- method={item['method']}, seed={item['seed']}, "
                f"returncode={item['returncode']}"
            )

        raise SystemExit(1)

    print("\nAll experiments finished successfully.")


if __name__ == "__main__":
    main()
