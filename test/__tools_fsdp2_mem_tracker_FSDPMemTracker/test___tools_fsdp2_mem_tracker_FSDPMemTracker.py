# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker 接口功能正确性
API 名称：torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker
API 签名：
  class FSDPMemTracker(MemTracker):
      def __init__(
          self,
          mod: torch.nn.Module,
          optm: torch.optim.Optimizer | None = None,
      ) -> None
      def track_inputs(self, inputs: tuple[Any, ...]) -> None
      def __enter__(self) -> "FSDPMemTracker"
      def __exit__(self, *args: Any) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | optm 缺省为 None vs 传入实际 optimizer                       | 已覆盖：test_init_without_optimizer / test_init_with_optimizer |
| 枚举选项         | 上下文管理器：进入与退出                                     | 已覆盖：test_context_manager_basic             |
| 参数类型         | mod 必须为 FSDPModule；optm 为 torch.optim.Optimizer 或 None | 已覆盖：test_init_wrong_module_type            |
| 传参与不传参     | 仅传 mod vs 传 mod+optm                                      | 已覆盖：test_init_without_optimizer / test_init_with_optimizer |
| 等价类/边界值    | 一层 fully_shard 模块                                        | 已覆盖：test_init_without_optimizer            |
| 正常传参场景     | 在 spawn 子进程下完成 with 块 + forward 跟踪                 | 已覆盖：test_track_forward                     |
| 异常传参场景     | 传入非 FSDPModule 时抛 AssertionError                        | 已覆盖：test_init_wrong_module_type            |
| 混合设备类型     | FSDPMemTracker 仅作用于 FSDP 模块（NPU 上），不支持 CPU mod  | 未覆盖：API 要求 mod 为 FSDPModule，FSDP 不支持 CPU 后端 |

未覆盖项及原因：
- 混合设备类型：FSDPModule 必须依附实际分布式后端，CPU 后端无法完成 fully_shard 流程。

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

from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29504')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_init_without_optimizer(rank, world_size, device_name):
    """Construct FSDPMemTracker with FSDP-wrapped module only."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.fsdp import fully_shard

        model = _TinyModel().to(f'{device_name}:{rank}')
        sharded = fully_shard(model)
        tracker = FSDPMemTracker(sharded)
        assert tracker is not None
        # memory_tracking should be present from MemTracker base class
        assert hasattr(tracker, 'memory_tracking')
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_with_optimizer(rank, world_size, device_name):
    """Construct FSDPMemTracker with optimizer."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.fsdp import fully_shard

        model = _TinyModel().to(f'{device_name}:{rank}')
        sharded = fully_shard(model)
        opt = torch.optim.SGD(sharded.parameters(), lr=1e-3)
        tracker = FSDPMemTracker(sharded, opt)
        assert tracker is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_init_wrong_module_type(rank, world_size, device_name):
    """Non-FSDP module should raise AssertionError."""
    _init_dist_hccl(rank, world_size)
    try:
        raised = False
        model = _TinyModel().to(f'{device_name}:{rank}')
        try:
            FSDPMemTracker(model)
        except AssertionError:
            raised = True
        assert raised, "Expected AssertionError for non-FSDP module"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_context_manager_basic(rank, world_size, device_name):
    """FSDPMemTracker works as a context manager."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.fsdp import fully_shard

        model = _TinyModel().to(f'{device_name}:{rank}')
        sharded = fully_shard(model)
        tracker = FSDPMemTracker(sharded)

        with tracker:
            pass

        assert tracker is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_track_forward(rank, world_size, device_name):
    """Track a forward pass via FSDPMemTracker."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.fsdp import fully_shard

        model = _TinyModel().to(f'{device_name}:{rank}')
        sharded = fully_shard(model)
        opt = torch.optim.SGD(sharded.parameters(), lr=1e-3)

        tracker = FSDPMemTracker(sharded, opt)
        inp = torch.randn(4, 8, device=f'{device_name}:{rank}')
        tracker.track_inputs((inp,))
        with tracker:
            out = sharded(inp)
            # Trigger backward to exercise mem tracker hooks
            out.sum().backward()

        # memory_tracking should now contain entries for tracked modules
        assert tracker.memory_tracking is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestFSDP2MemTracker(TestCase):
    """Tests for torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_init_without_optimizer(self):
        mp.spawn(_test_init_without_optimizer,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_with_optimizer(self):
        mp.spawn(_test_init_with_optimizer,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_init_wrong_module_type(self):
        mp.spawn(_test_init_wrong_module_type,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_context_manager_basic(self):
        mp.spawn(_test_context_manager_basic,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_track_forward(self):
        mp.spawn(_test_track_forward,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
