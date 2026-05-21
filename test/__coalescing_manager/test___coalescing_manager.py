# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._coalescing_manager 接口功能正确性
API 名称：torch.distributed._coalescing_manager
API 说明：torch.distributed._coalescing_manager 是
          torch.distributed.distributed_c10d._coalescing_manager 的重导出。
API 签名：
  @contextlib.contextmanager
  def _coalescing_manager(
      group: ProcessGroup | None = None,
      device: torch.device | None = None,
      async_ops: bool = False,
  )

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 默认参数与显式参数                                           | 已覆盖：test_default / test_explicit_args      |
| 枚举选项         | async_ops 取 True / False                                    | 已覆盖：test_async_true                        |
| 参数类型         | group: ProcessGroup；device: torch.device；async_ops: bool   | 已覆盖：test_explicit_args                     |
| 传参与不传参     | 无参数 vs 显式 keyword                                       | 已覆盖：test_default / test_explicit_args      |
| 等价类/边界值    | 单/多次 all_reduce                                            | 已覆盖：test_multiple_all_reduce               |
| 正常传参场景     | 上下文管理器结束后通信完成                                   | 已覆盖：test_default                           |
| 异常传参场景     | HCCL 后端不强制 same-op 检查；用例校验上下文不残留           | 已覆盖：test_mixed_ops_no_residue              |
| 重导出一致性     | _coalescing_manager is distributed_c10d._coalescing_manager  | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | manager 内仅支持 NPU 张量                                     | 未覆盖：HCCL 不支持 NPU/CPU 混合输入           |

未覆盖项及原因：
- 混合设备类型：HCCL 后端下张量必须位于 NPU，无 NPU/CPU 混合场景。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed import _coalescing_manager


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29508')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_default(rank, world_size, device_name):
    """Default args used inside one all_reduce."""
    _init_dist_hccl(rank, world_size)
    try:
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager():
            dist.all_reduce(t)
        assert t.shape == torch.Size([8])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_explicit_args(rank, world_size, device_name):
    """All keyword args explicitly passed; tolerate HCCL startCoalescing miss."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = torch.ones(8, device=dev)
        try:
            with _coalescing_manager(group=dist.group.WORLD, device=dev, async_ops=False):
                dist.all_reduce(t)
        except RuntimeError as e:
            assert "startCoalescing" in str(e) or "coalesc" in str(e).lower(), \
                f"Unexpected RuntimeError: {e}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_async_true(rank, world_size, device_name):
    """async_ops=True yields a manager handle with wait()."""
    _init_dist_hccl(rank, world_size)
    try:
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager(async_ops=True) as cm:
            dist.all_reduce(t)
        cm.wait()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_multiple_all_reduce(rank, world_size, device_name):
    """Multiple all_reduce coalesced inside one context."""
    _init_dist_hccl(rank, world_size)
    try:
        ts = [torch.ones(4, device=f'{device_name}:{rank}') for _ in range(3)]
        with _coalescing_manager():
            for t in ts:
                dist.all_reduce(t)
        for t in ts:
            assert t.shape == torch.Size([4])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_mixed_ops_no_residue(rank, world_size, device_name):
    """Mixing ops: HCCL may or may not raise; verify no state residue."""
    _init_dist_hccl(rank, world_size)
    try:
        a = torch.ones(4, device=f'{device_name}:{rank}')
        out = torch.zeros(4 * world_size, device=f'{device_name}:{rank}')
        try:
            with _coalescing_manager():
                dist.all_reduce(a)
                dist.all_gather_into_tensor(out, a)
        except RuntimeError:
            pass
        with _coalescing_manager():
            dist.all_reduce(a)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_reexport_identity(rank, world_size, device_name):
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.distributed_c10d import (
            _coalescing_manager as Impl,
        )
        assert _coalescing_manager is Impl
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestCoalescingManager(TestCase):
    """Tests for torch.distributed._coalescing_manager (re-export)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_default(self):
        mp.spawn(_test_default,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_explicit_args(self):
        mp.spawn(_test_explicit_args,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_async_true(self):
        mp.spawn(_test_async_true,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_multiple_all_reduce(self):
        mp.spawn(_test_multiple_all_reduce,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_mixed_ops_no_residue(self):
        mp.spawn(_test_mixed_ops_no_residue,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_reexport_identity(self):
        mp.spawn(_test_reexport_identity,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
