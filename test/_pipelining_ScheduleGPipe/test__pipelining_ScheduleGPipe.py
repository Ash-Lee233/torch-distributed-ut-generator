# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.pipelining.ScheduleGPipe 接口功能正确性
API 名称：torch.distributed.pipelining.ScheduleGPipe
API 签名（继承自 PipelineScheduleSingle）：
  class ScheduleGPipe(PipelineScheduleSingle):
      def __init__(
          self,
          stage: _PipelineStageBase,
          n_microbatches: int,
          loss_fn: Callable | None = None,
          args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
          kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
          output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
          scale_grads: bool = True,
      )
      def step(self, *args, target=None, losses=None, return_outputs=True, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | loss_fn None vs 提供                                         | 已覆盖：test_init_default / test_init_with_loss_fn |
| 枚举选项         | scale_grads True/False                                       | 已覆盖：test_scale_grads                       |
| 参数类型         | stage:_PipelineStageBase；n_microbatches:int；loss_fn:Callable| 已覆盖：test_init_default                      |
| 传参与不传参     | 仅 stage+n_microbatches vs 全字段                            | 已覆盖：test_init_default / test_init_full     |
| 等价类/边界值    | n_microbatches = 2 / 4 (>= num_stages 约束)                   | 已覆盖：test_step / test_step_n_microbatches_4 |
| 正常传参场景     | 调度执行完成无异常，最后一阶段返回 Tensor                    | 已覆盖：test_step                              |
| 异常传参场景     | n_microbatches=0 不合法                                       | 已覆盖：test_invalid_n_microbatches            |
| 混合设备类型     | stage 与输入张量必须同 NPU；CPU 输入不被支持                   | 未覆盖：调度强制使用 stage.device 收发         |

未覆盖项及原因：
- 混合设备类型：ScheduleGPipe 通过 stage.device 收发，张量必须与 stage.device 一致。

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

from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29506')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _make_stage(rank, world_size, device):
    """Each rank holds a single Linear stage."""
    sub = nn.Linear(8, 8).to(device)
    return PipelineStage(
        submodule=sub,
        stage_index=rank,
        num_stages=world_size,
        device=device,
    )


def _loss_fn(out, tgt):
    return ((out - tgt) ** 2).mean()


def _test_init_default(rank, world_size, device_name):
    """ScheduleGPipe default construction."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        sched = ScheduleGPipe(stage, n_microbatches=2)
        assert sched._n_microbatches == 2
        assert sched._stage is stage
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_with_loss_fn(rank, world_size, device_name):
    """ScheduleGPipe with loss_fn provided."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        sched = ScheduleGPipe(stage, n_microbatches=2, loss_fn=_loss_fn)
        assert sched._loss_fn is _loss_fn
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_full(rank, world_size, device_name):
    """ScheduleGPipe full keyword construction."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        sched = ScheduleGPipe(
            stage,
            n_microbatches=2,
            loss_fn=_loss_fn,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
            output_merge_spec=None,
            scale_grads=True,
        )
        assert sched._n_microbatches == 2
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_scale_grads(rank, world_size, device_name):
    """scale_grads parameter accepted as True / False."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        s1 = ScheduleGPipe(stage, n_microbatches=2, scale_grads=True)
        s2 = ScheduleGPipe(stage, n_microbatches=2, scale_grads=False)
        assert s1 is not None and s2 is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_step(rank, world_size, device_name):
    """Run a full GPipe step across two stages."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        sched = ScheduleGPipe(stage, n_microbatches=2, loss_fn=_loss_fn)

        # batch = 4, divisible into 2 microbatches
        x = torch.randn(4, 8, device=device)
        target = torch.randn(4, 8, device=device)

        if stage.is_first:
            out = sched.step(x, target=target)
        elif stage.is_last:
            out = sched.step(target=target)
        else:
            out = sched.step()

        if stage.is_last:
            assert out is not None
            assert out.shape == torch.Size([4, 8])
            assert out.device.type == device_name
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_step_n_microbatches_4(rank, world_size, device_name):
    """ScheduleGPipe with n_microbatches=4 executes 4 microbatches across 2 stages."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        sched = ScheduleGPipe(stage, n_microbatches=4, loss_fn=_loss_fn)

        x = torch.randn(8, 8, device=device)
        target = torch.randn(8, 8, device=device)

        if stage.is_first:
            sched.step(x, target=target)
        elif stage.is_last:
            sched.step(target=target)
        else:
            sched.step()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_invalid_n_microbatches(rank, world_size, device_name):
    """n_microbatches=0 should be rejected by the scheduler."""
    _init_dist_hccl(rank, world_size)
    try:
        device = torch.device(f'{device_name}:{rank}')
        stage = _make_stage(rank, world_size, device)
        raised = False
        try:
            ScheduleGPipe(stage, n_microbatches=0)
        except (ValueError, AssertionError, ZeroDivisionError):
            raised = True
        assert raised, "Expected an error for n_microbatches=0"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestPipeliningScheduleGPipe(TestCase):
    """Tests for torch.distributed.pipelining.ScheduleGPipe."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_init_default(self):
        mp.spawn(_test_init_default,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_with_loss_fn(self):
        mp.spawn(_test_init_with_loss_fn,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_full(self):
        mp.spawn(_test_init_full,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_scale_grads(self):
        mp.spawn(_test_scale_grads,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step(self):
        mp.spawn(_test_step,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_n_microbatches_4(self):
        mp.spawn(_test_step_n_microbatches_4,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_n_microbatches(self):
        mp.spawn(_test_invalid_n_microbatches,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
