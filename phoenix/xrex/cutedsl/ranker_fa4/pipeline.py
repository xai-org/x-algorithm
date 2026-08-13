# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Derived from flash-attention (https://github.com/Dao-AILab/flash-attention);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg


def _override_create(parent_cls, child_cls):
    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


class PipelineStateSimple:
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


def _call_with_elect_one(parent_method, self, state, elect_one, syncwarp, loc, ip):
    if const_expr(elect_one):
        if const_expr(syncwarp):
            cute.arch.sync_warp()
        with cute.arch.elect_one():
            parent_method(self, state, loc=loc, ip=ip)
    else:
        parent_method(self, state, loc=loc, ip=ip)


class _PipelineIndexPhaseMixin:
    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Boolean | None = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    create = _override_create(NamedBarrierOg, None)

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineCpAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineCpAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineCpAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Boolean | None = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)
