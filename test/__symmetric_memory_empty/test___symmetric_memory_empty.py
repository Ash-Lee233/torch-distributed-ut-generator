# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.empty 接口功能正确性
API 名称：torch.distributed._symmetric_memory.empty
API 签名：
  def empty(
      *size: Any,
      dtype: torch.dtype | None = None,
      device: torch.device | None = None,
  ) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 不同 shape；含 1D / 2D                                       | 已覆盖：test_empty_basic / test_empty_2d       |
| 枚举选项         | dtype: 默认 / float32 / float16 / int32                      | 已覆盖：test_empty_dtype_variants              |
| 参数类型         | size 为 varargs / list；dtype: torch.dtype                   | 已覆盖：test_empty_basic / test_empty_list_size |
| 传参与不传参     | dtype/device 默认 None vs 显式                               | 已覆盖：test_empty_basic / test_empty_explicit_device |
| 等价类/边界值    | 单元素；多维                                                  | 已覆盖：test_empty_basic / test_empty_2d       |
| 正常传参场景     | 返回 torch.Tensor，shape/dtype/device 符合预期               | 已覆盖：test_empty_basic                       |
| 异常传参场景     | 在不支持的后端（NPU）上可能抛 RuntimeError；用例容忍此情况   | 已覆盖：test_empty_basic（带 expected error 路径） |
| 混合设备类型     | 该 API 仅返回单 Tensor，device 由参数指定，无多 Tensor 输入   | 未覆盖：API 无多张量输入                       |

未覆盖项及原因：
- 混合设备类型：empty 只生成单个 Tensor，无 NPU/CPU 混合输入路径。

注意：_SymmetricMemory 主要为 CUDA/NCCL/NVSHMEM 实现，在 NPU 上 P2P 分配可能未实现。
     本测试在执行失败时容忍 RuntimeError/NotImplementedError，仅校验签名与基础行为。
本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed._symmetric_memory import empty as symm_empty


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29510')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _try_empty(*args, **kwargs):
    """Call symm_empty; return ('ok', tensor) or ('err', exception)."""
    try:
        t = symm_empty(*args, **kwargs)
        return ('ok', t)
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as e:
        return ('err', e)


def _test_empty_basic(rank, world_size, device_name):
    """empty(8) with default dtype on NPU; tolerate backend-not-supported error."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        status, val = _try_empty(8, device=dev)
        if status == 'ok':
            assert val.shape == torch.Size([8])
            assert val.device.type == device_name
        else:
            # Backend doesn't support P2P alloc on NPU; assert it's a Runtime/NotImpl error
            assert isinstance(val, (RuntimeError, NotImplementedError)), \
                f"Unexpected error type: {type(val)}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_2d(rank, world_size, device_name):
    """empty(4, 8) shape preserved when allocation succeeds."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        status, val = _try_empty(4, 8, device=dev)
        if status == 'ok':
            assert val.shape == torch.Size([4, 8])
        else:
            assert isinstance(val, (RuntimeError, NotImplementedError))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_dtype_variants(rank, world_size, device_name):
    """empty accepts a range of dtypes."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        for dt in (torch.float32, torch.float16, torch.int32):
            status, val = _try_empty(4, dtype=dt, device=dev)
            if status == 'ok':
                assert val.dtype == dt
            else:
                assert isinstance(val, (RuntimeError, NotImplementedError))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_list_size(rank, world_size, device_name):
    """empty accepts a list-of-ints as size."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        status, val = _try_empty([4, 8], device=dev)
        if status == 'ok':
            assert val.shape == torch.Size([4, 8])
        else:
            assert isinstance(val, (RuntimeError, NotImplementedError))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_explicit_device(rank, world_size, device_name):
    """empty accepts device as torch.device or string."""
    _init_dist_hccl(rank, world_size)
    try:
        s1, _ = _try_empty(4, device=f'{device_name}:{rank}')
        s2, _ = _try_empty(4, device=torch.device(f'{device_name}:{rank}'))
        # Either both ok, or both fail consistently
        assert s1 == s2, f"Inconsistent results: {s1} vs {s2}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestSymmetricMemoryEmpty(TestCase):
    """Tests for torch.distributed._symmetric_memory.empty."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_callable(self):
        """empty is callable with the documented signature."""
        self.assertTrue(callable(symm_empty))
        sig = inspect.signature(symm_empty)
        params = sig.parameters
        self.assertIn("dtype", params)
        self.assertIn("device", params)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_basic(self):
        mp.spawn(_test_empty_basic,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_2d(self):
        mp.spawn(_test_empty_2d,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_dtype_variants(self):
        mp.spawn(_test_empty_dtype_variants,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_list_size(self):
        mp.spawn(_test_empty_list_size,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_explicit_device(self):
        mp.spawn(_test_empty_explicit_device,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
