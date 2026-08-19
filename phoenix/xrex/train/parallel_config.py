# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from xai_configlib import Config, configclass

from xrex.utils.gpu import NUM_PHYSICAL_DEVICES_PER_NODE


@configclass
class ParallelConfig(Config):
    dp: int = 1
    ep: int = 1

    num_devices_per_process: int = 1
    num_devices_per_node: int = NUM_PHYSICAL_DEVICES_PER_NODE

    @property
    def axis_names(self) -> tuple[str, str, str, str, str, str]:
        return ("stage", "expert", "replica", "data", "seq", "model")

    def mesh_shape(self):
        return (1, self.ep, self.dp, 1, 1, 1)

    def mesh_axis_names(self):
        return self.axis_names

    def num_devices(self):
        return self.ep * self.dp

    def get_local_batch_size_and_rank(self, global_batch_size):
        num_data_devices = self.num_devices()

        num_data_processes = num_data_devices // self.num_devices_per_process
        read_bsz_per_process = global_batch_size // num_data_processes

        assert num_data_processes >= 1
        assert num_data_devices % self.num_devices_per_process == 0
        assert read_bsz_per_process >= 1
        assert global_batch_size % num_data_processes == 0, (
            global_batch_size,
            num_data_processes,
        )

        return read_bsz_per_process, num_data_processes, 1
