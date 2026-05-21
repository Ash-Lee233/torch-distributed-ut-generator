# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.pipelining.PipelineStage 接口功能正确性
API 名称：torch.distributed.pipelining.PipelineStage
API 签名：
  class PipelineStage(_PipelineStageBase):
      def __init__(
          self,
          submodule: nn.Module,
          stage_index: int,
          num_stages: int,
          device: torch.device,
          input_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
          output_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
          group: dist.ProcessGroup | None = None,
          dw_builder: Callable[[], Callable[..., None]] | None = None,
      ) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | input_args / output_args 全 None vs 同时提供                 | 已覆盖：test_init_no_args / test_init_with_input_output_args |
| 枚举选项         | stage_index 取 0（first） / num_stages-1（last）             | 已覆盖：test_first_stage / test_last_stage     |
| 参数类型         | submodule:nn.Module；input_args/output_args:Tensor 或 tuple   | 已覆盖：test_init_tuple_input_args             |
| 传参与不传参     | 仅必填 vs 显式 group=None                                    | 已覆盖：test_init_no_args / test_init_explicit_group_none |
| 等价类/边界值    | 2-stage 拓扑覆盖 first/last                                  | 已覆盖：test_first_stage / test_last_stage     |
| 正常传参场景     | is_first / is_last / num_stages 属性正确                     | 已覆盖：test_stage_properties                  |
| 异常传参场景     | 仅传 output_args 不传 input_args -> AssertionError            | 已覆盖：test_output_without_input_raises       |
| 混合设备类型     | submodule 与 device 都必须在同一 NPU；不支持 CPU 模块         | 未覆盖：PipelineStage 强制 device 一致         |

未覆盖项及原因：
- 混合设备类型：PipelineStage 在初始化时按指定 device 收发通信，不接受 NPU/CPU 混合 submodule。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed.pipelining import PipelineStage


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29505')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _build_submodule(rank, device):
    """Each rank holds a single Linear stage."""
    return nn.Linear(8, 8).to(device)


def _test_init_no_args(rank, world_size, device_name):
    """PipelineStage init without input/output args (runtime shape inference)."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
        )
        assert stage.stage_index == rank
        assert stage.num_stages == world_size
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_with_input_output_args(rank, world_size, device_name):
    """PipelineStage init with both input_args and output_args."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        inp = torch.zeros(4, 8, device=device)
        out = torch.zeros(4, 8, device=device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
            input_args=inp,
            output_args=out,
        )
        assert stage.inputs_meta is not None
        assert len(stage.inputs_meta) == 1
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_tuple_input_args(rank, world_size, device_name):
    """input_args/output_args provided as tuples."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        inp = (torch.zeros(4, 8, device=device),)
        out = (torch.zeros(4, 8, device=device),)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
            input_args=inp,
            output_args=out,
        )
        assert isinstance(stage.inputs_meta, tuple)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_first_stage(rank, world_size, device_name):
    """stage_index=0 should be the first stage."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
        )
        if rank == 0:
            assert stage.is_first is True
            assert stage.is_last is False
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_last_stage(rank, world_size, device_name):
    """stage_index=num_stages-1 should be the last stage."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
        )
        if rank == world_size - 1:
            assert stage.is_last is True
            assert stage.is_first is False
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_stage_properties(rank, world_size, device_name):
    """num_stages and stage_index match constructor inputs."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
        )
        assert stage.stage_index == rank
        assert stage.num_stages == world_size
        # The submodule should be retained on the stage
        assert stage.submod is sub or hasattr(stage, 'submod')
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_explicit_group_none(rank, world_size, device_name):
    """Explicitly passing group=None uses the default process group."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        stage = PipelineStage(
            submodule=sub,
            stage_index=rank,
            num_stages=world_size,
            device=device,
            group=None,
        )
        assert stage is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_output_without_input_raises(rank, world_size, device_name):
    """Passing output_args without input_args raises AssertionError."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        sub = _build_submodule(rank, device)
        out = torch.zeros(4, 8, device=device)
        raised = False
        try:
            PipelineStage(
                submodule=sub,
                stage_index=rank,
                num_stages=world_size,
                device=device,
                input_args=None,
                output_args=out,
            )
        except AssertionError:
            raised = True
        assert raised, "Expected AssertionError when output_args is set without input_args"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestPipeliningPipelineStage(TestCase):
    """Tests for torch.distributed.pipelining.PipelineStage."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_init_no_args(self):
        mp.spawn(_test_init_no_args,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_with_input_output_args(self):
        mp.spawn(_test_init_with_input_output_args,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_tuple_input_args(self):
        mp.spawn(_test_init_tuple_input_args,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_first_stage(self):
        mp.spawn(_test_first_stage,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_last_stage(self):
        mp.spawn(_test_last_stage,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_stage_properties(self):
        mp.spawn(_test_stage_properties,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_explicit_group_none(self):
        mp.spawn(_test_init_explicit_group_none,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_output_without_input_raises(self):
        mp.spawn(_test_output_without_input_raises,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
