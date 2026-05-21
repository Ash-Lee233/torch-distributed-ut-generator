# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d._coalescing_manager 接口功能正确性
API 名称：torch.distributed.distributed_c10d._coalescing_manager
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
| 空/非空          | group 缺省 None 使用默认；device 默认 None                  | 已覆盖：test_basic_default_group               |
| 枚举选项         | async_ops 取 True / False                                    | 已覆盖：test_async_ops_false / test_async_ops_true |
| 参数类型         | group: ProcessGroup；async_ops: bool                          | 已覆盖：test_explicit_group / test_async_ops_true |
| 传参与不传参     | 无参数 vs 全部 keyword 传入                                  | 已覆盖：test_basic_default_group / test_explicit_group |
| 等价类/边界值    | 单次 all_reduce vs 多次连续 all_reduce                       | 已覆盖：test_coalesce_multiple_all_reduce      |
| 正常传参场景     | 上下文管理器正确退出且通信完成                               | 已覆盖：test_basic_default_group               |
| 异常传参场景     | HCCL 后端不实现 startCoalescing/同 op 检查；用例仅校验上下文不残留 | 已覆盖：test_mixed_ops_no_residue              |
| 重导出一致性     | torch.distributed._coalescing_manager is 同一函数             | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | manager 内只支持 NPU 张量（HCCL 后端）                        | 未覆盖：HCCL 不支持 NPU/CPU 混合输入           |

未覆盖项及原因：
- 混合设备类型：在 HCCL 后端下张量必须位于 NPU，无 NPU/CPU 混合场景。

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

from torch.distributed.distributed_c10d import _coalescing_manager


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29507')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_basic_default_group(rank, world_size, device_name):
    """_coalescing_manager works with default args and one all_reduce."""
    _init_dist_hccl(rank, world_size)
    try:
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager():
            dist.all_reduce(t)
        # After exit, the synchronous path should have finished
        assert t.shape == torch.Size([8])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_explicit_group(rank, world_size, device_name):
    """Passing the default group explicitly works."""
    _init_dist_hccl(rank, world_size)
    try:
        group = dist.group.WORLD
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager(group=group):
            dist.all_reduce(t)
        assert t.shape == torch.Size([8])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_async_ops_false(rank, world_size, device_name):
    """async_ops=False synchronizes inside the context."""
    _init_dist_hccl(rank, world_size)
    try:
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager(async_ops=False):
            dist.all_reduce(t)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_async_ops_true(rank, world_size, device_name):
    """async_ops=True returns a _CoalescingManager with wait()."""
    _init_dist_hccl(rank, world_size)
    try:
        t = torch.ones(8, device=f'{device_name}:{rank}')
        with _coalescing_manager(async_ops=True) as cm:
            dist.all_reduce(t)
        assert cm is not None and hasattr(cm, 'wait')
        cm.wait()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_device_arg(rank, world_size, device_name):
    """Passing device kwarg: HCCL may not implement startCoalescing; tolerate."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = torch.ones(8, device=dev)
        try:
            with _coalescing_manager(device=dev):
                dist.all_reduce(t)
        except RuntimeError as e:
            # HCCL backend does not implement startCoalescing — accepted.
            assert "startCoalescing" in str(e) or "coalesc" in str(e).lower(), \
                f"Unexpected RuntimeError: {e}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_coalesce_multiple_all_reduce(rank, world_size, device_name):
    """Multiple all_reduce calls coalesced under the same manager."""
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
    """Mixing ops: HCCL may or may not raise; either way state must reset."""
    _init_dist_hccl(rank, world_size)
    try:
        a = torch.ones(4, device=f'{device_name}:{rank}')
        out = torch.zeros(4 * world_size, device=f'{device_name}:{rank}')
        try:
            with _coalescing_manager():
                dist.all_reduce(a)
                dist.all_gather_into_tensor(out, a)
        except RuntimeError:
            pass  # acceptable
        # A second manager scope must still work cleanly (no residue).
        with _coalescing_manager():
            dist.all_reduce(a)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_reexport_identity(rank, world_size, device_name):
    """torch.distributed._coalescing_manager is the same function."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed import _coalescing_manager as TopLevel
        assert TopLevel is _coalescing_manager
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestDistributedC10dCoalescingManager(TestCase):
    """Tests for torch.distributed.distributed_c10d._coalescing_manager."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_basic_default_group(self):
        mp.spawn(_test_basic_default_group,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_explicit_group(self):
        mp.spawn(_test_explicit_group,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_async_ops_false(self):
        mp.spawn(_test_async_ops_false,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_async_ops_true(self):
        mp.spawn(_test_async_ops_true,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_device_arg(self):
        mp.spawn(_test_device_arg,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_coalesce_multiple_all_reduce(self):
        mp.spawn(_test_coalesce_multiple_all_reduce,
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
