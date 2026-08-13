# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.

from xrex.eval.eval_utils import EvaluationTaskNew, ForwardEvalNew
from xrex.eval.metrics_recsys import RecsysRCEMetrics, merge_report_extras
from xrex.utils.utils import flatten_dict


def report_forward_eval_results(
    results: dict[str, ForwardEvalNew | EvaluationTaskNew], step: int | None = None
) -> dict[str, float]:
    del step
    all_metrics = {}
    for eval_name, eval in results.items():
        all_metrics[eval_name] = {}
        assert eval.metrics is not None
        for metric_name, metric in eval.metrics:
            all_metrics[eval_name][metric_name] = metric.get()
            if isinstance(metric, RecsysRCEMetrics):
                all_metrics[eval_name][f"{metric_name}.pos_frac"] = metric.get_ratio()
    return merge_report_extras(results, flatten_dict(all_metrics))
