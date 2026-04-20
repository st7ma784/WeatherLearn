"""
Auto-create a W&B report dashboard for PTL diagnostics.

Usage:
  python setup_wandb_dashboard.py --entity <entity> --project <project>

Optional:
  --title "PTL Diagnostics Dashboard"
  --query "config.method = \"grid\""
  --dry-run
"""

import argparse
import textwrap


def _build_parser():
    parser = argparse.ArgumentParser(description="Create a W&B report dashboard for PTL diagnostics.")
    parser.add_argument("--entity", required=True, type=str, help="W&B entity/user or team")
    parser.add_argument("--project", required=True, type=str, help="W&B project name")
    parser.add_argument("--title", default="PTL Diagnostics Dashboard", type=str, help="Report title")
    parser.add_argument(
        "--description",
        default="Auto-generated dashboard for PTL ionosphere forecasting diagnostics.",
        type=str,
        help="Report description",
    )
    parser.add_argument(
        "--query",
        default="",
        type=str,
        help="Optional W&B run query to filter runs in all panels",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating a report")
    return parser


def _safe_import_reports():
    try:
        from wandb.apis import reports as wr
        return wr, None
    except Exception as exc:
        return None, exc


def _line_panel(wr, title, ys, x="Step"):
    return wr.LinePlot(title=title, x=x, y=ys)


def _build_blocks(wr, runset):
    blocks = [
        wr.H1("PTL Diagnostics Dashboard"),
        wr.P(
            "This report is auto-generated for PTL diagnostics. "
            "Use it as a standard layout for run review and regression detection."
        ),
        wr.H2("A) Training Health"),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                _line_panel(wr, "Loss / MSE", ["train_loss", "train_mse", "val_loss", "val_mse"]),
                _line_panel(wr, "Gradient Norm", ["diag/grad_norm"]),
                _line_panel(wr, "Runtime Throughput", ["diag/step_time_s", "diag/samples_per_sec"]),
            ],
        ),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                _line_panel(
                    wr,
                    "Prediction vs Target Std by Channel",
                    [
                        "diag/train_pred_std_c0",
                        "diag/train_pred_std_c1",
                        "diag/train_pred_std_c2",
                        "diag/train_pred_std_c3",
                        "diag/train_pred_std_c4",
                        "diag/train_target_std_c0",
                        "diag/train_target_std_c1",
                        "diag/train_target_std_c2",
                        "diag/train_target_std_c3",
                        "diag/train_target_std_c4",
                    ],
                )
            ],
        ),
        wr.H2("B) Skill vs Baselines"),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                _line_panel(wr, "Skill vs Persistence", ["diag/val_skill_persistence"]),
                _line_panel(wr, "Skill vs Climatology", ["diag/val_skill_climatology"]),
                _line_panel(wr, "Event vs Quiet Error", ["diag/val_event_mse", "diag/val_quiet_mse"]),
            ],
        ),
        wr.H2("C) Spatial / Behavior Diagnostics"),
        wr.P(
            textwrap.dedent(
                """
                Media diagnostics are logged by PTL model hooks under these keys:
                - diag/val_rmse_map
                - diag/val_example_panel
                - diag/val_residual_hist
                - diag/val_lead_time_table

                Add Media/Table panels in the W&B UI for these keys if your SDK version
                does not support programmatic creation of media panels.
                """
            ).strip()
        ),
        wr.H2("D) Rollout Diagnostics"),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                _line_panel(wr, "Validation Loss", ["val_loss"]),
                _line_panel(wr, "Validation MSE", ["val_mse"]),
            ],
        ),
        wr.H2("E) EMA / Config Controls"),
        wr.P(
            "Track and compare runs by config fields: use_ema, use_ema_eval, "
            "ema_decay, ema_warmup_steps, cache_stats, diagnostics_interval."
        ),
    ]
    return blocks


def create_report(entity, project, title, description, query="", dry_run=False):
    wr, import_error = _safe_import_reports()
    if wr is None:
        raise RuntimeError(
            "Unable to import wandb reports API. Install/upgrade wandb and retry. "
            f"Original error: {import_error}"
        )

    runset = wr.Runset(entity=entity, project=project, name="PTL Runs", query=query)
    blocks = _build_blocks(wr, runset)

    if dry_run:
        print("[DRY RUN] Would create report with the following metadata:")
        print(f"  entity={entity}")
        print(f"  project={project}")
        print(f"  title={title}")
        print(f"  query={query!r}")
        print(f"  blocks={len(blocks)}")
        return None

    report = wr.Report(
        project=project,
        entity=entity,
        title=title,
        description=description,
        blocks=blocks,
    )
    report.save()
    return report


def main():
    args = _build_parser().parse_args()
    report = create_report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        description=args.description,
        query=args.query,
        dry_run=args.dry_run,
    )
    if report is not None:
        print("Created W&B report dashboard successfully.")
        if hasattr(report, "url"):
            print(f"Report URL: {report.url}")


if __name__ == "__main__":
    main()
