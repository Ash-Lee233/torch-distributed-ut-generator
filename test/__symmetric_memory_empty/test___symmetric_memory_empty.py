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
| 异常传参场景     | 用例不再容忍错误；NPU/HCCL 下若 API 未实现，用例将 ERROR     | 已覆盖：失败路径直接暴露                       |
| 混合设备类型     | 该 API 仅返回单 Tensor，device 由参数指定，无多 Tensor 输入   | 未覆盖：API 无多张量输入                       |

未覆盖项及原因：
- 混合设备类型：empty 只生成单个 Tensor，无 NPU/CPU 混合输入路径。

注意：_SymmetricMemory 主要为 CUDA/NCCL/NVSHMEM 实现。本测试不再容忍后端未实现错误：
     若在 NPU/HCCL 上 P2P 分配抛出 RuntimeError/NotImplementedError，
     用例将直接 ERROR/FAIL，由用户判断是否为框架缺口。

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


def _test_empty_basic(rank, world_size, device_name):
    """empty(8) returns a 1D tensor with the expected shape/device."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty(8, device=dev)
        assert t.shape == torch.Size([8]), f"Unexpected shape: {t.shape}"
        assert t.device.type == device_name, f"Unexpected device: {t.device}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_2d(rank, world_size, device_name):
    """empty(4, 8) preserves a 2D shape."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty(4, 8, device=dev)
        assert t.shape == torch.Size([4, 8]), f"Unexpected shape: {t.shape}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_dtype_variants(rank, world_size, device_name):
    """empty accepts float32 / float16 / int32 dtypes."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        for dt in (torch.float32, torch.float16, torch.int32):
            t = symm_empty(4, dtype=dt, device=dev)
            assert t.dtype == dt, f"Expected dtype {dt}, got {t.dtype}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_list_size(rank, world_size, device_name):
    """empty accepts a list-of-ints as size."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty([4, 8], device=dev)
        assert t.shape == torch.Size([4, 8]), f"Unexpected shape: {t.shape}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_empty_explicit_device(rank, world_size, device_name):
    """empty accepts device as torch.device object or string, both succeed."""
    _init_dist_hccl(rank, world_size)
    try:
        t1 = symm_empty(4, device=f'{device_name}:{rank}')
        t2 = symm_empty(4, device=torch.device(f'{device_name}:{rank}'))
        assert t1.shape == torch.Size([4])
        assert t2.shape == torch.Size([4])
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
