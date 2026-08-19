# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import jax
from xai_configlib import Config, configclass

from xrex.eval.metrics import ForwardMetrics
from xrex.train.misc import TrainingState
from xrex.utils.utils import alarm_after, flatten_dict

rank_logger = logging.getLogger("rank")


@configclass
class EvalModule(Config):
    eval_name: str
    eval_conf: "EvaluationTaskNew"
    eval_dataset: Config
    eval_bs_per_device: float | int
    eval_seq_len: int

    def initialize(self):
        if not all(
            hasattr(self, attr)
            for attr in [
                "eval_name",
                "eval_conf",
                "eval_dataset",
                "eval_bs_per_device",
                "eval_seq_len",
            ]
        ):
            raise ValueError("All fields must be specified")


@dataclass
class EvaluationTaskNew(Config):
    max_steps: int = -1
    eval_step_timeout: int = 300
    metrics: list[tuple[str, ForwardMetrics]] | None = None

    def initialize(self):
        if self.metrics is not None:
            for _, metric in self.metrics:
                metric.initialize()

    def report(self, name: str, step: int | None = None) -> str:
        results = sorted(self.results)
        max_len = max([len(name) for name in results])
        step = ""
        if step is not None:
            step = f" at {step}"
        output = "Eval {name}{step}:\n"
        for name, eval_result in results:
            padded_name = name + " " * (max_len - len(name))
            output += f"{padded_name}:  {eval_result.get()}\n"
        return output

    @staticmethod
    def params_from_state(state):
        return state.params


@configclass
class ForwardEvalNew(EvaluationTaskNew):
    def run(self, data_iter, forward_fn, mesh, name=None, **kwargs):
        if self.metrics is not None:
            self.initialize()
        rank_logger.info(f"start eval {name}...")
        i = 0
        while True:
            if self.max_steps > 0 and i >= self.max_steps:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                rank_logger.info(f"Stop iter at step {i} for eval {name} ...")
                break
            except Exception:
                rank_logger.error(f"Error at step {i} for eval {name}:")
                rank_logger.error(traceback.format_exc())
                break
            with alarm_after(self.eval_step_timeout):
                logits = forward_fn(batch)
            if isinstance(logits, dict):
                assert "output" in logits.keys()
                logits = logits["output"]
            for _metric_name, metric in self.metrics:
                metric.run_batch(
                    logits,
                    batch.get("inputs"),
                    batch.get("targets"),
                    batch.get("mask", None),
                )
            rank_logger.info(f"eval step {i} for eval {name}")
            i += 1
        data_iter.close()
        rank_logger.info(f"Eval finished for {name}")
        return self


def run_forward_evals_new(
    evals: list[tuple[str, ForwardEvalNew]],
    data_iter: Iterator,
    *,
    forward_fn: Callable,
    state: TrainingState,
    mesh: jax.sharding.Mesh,
    sampler=None,
) -> dict[str, ForwardEvalNew]:
    return {
        name: conf.run(
            data_iter,
            partial(forward_fn, params=conf.params_from_state(state)),
            mesh,
            name=name,
            sampler=sampler,
        )
        for name, conf in evals
    }


def report_forward_eval_results(
    results: dict[str, ForwardEvalNew | EvaluationTaskNew],
    metric_folder: str | None = None,
    elapsed_samples: int | None = None,
    is_data_shard_output_rank: bool = True,
) -> dict[str, float]:
    all_metrics = {}
    for eval_name, eval in results.items():
        all_metrics[eval_name] = {}
        assert eval.metrics is not None
        for metric_name, metric in eval.metrics:
            value = metric.get()
            if value is not None:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        name = metric_name if sub_key is None else f"{metric_name}/{sub_key}"
                        all_metrics[eval_name][name] = sub_value
                else:
                    all_metrics[eval_name][metric_name] = value
            metric.after_run(
                eval_name=eval_name,
                elapsed_samples=elapsed_samples,
                metric_folder=metric_folder,
                is_data_shard_output_rank=is_data_shard_output_rank,
            )
    return flatten_dict(all_metrics)
