# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.rendezvous 接口功能正确性
API 名称：torch.distributed._symmetric_memory.rendezvous
API 签名：
  def rendezvous(
      tensor: torch.Tensor,
      group: c10d.GroupName | ProcessGroup,
  ) -> _SymmetricMemory

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | tensor 非空；group 非空                                       | 已覆盖：test_rendezvous_with_group_obj         |
| 枚举选项         | group 类型：ProcessGroup vs 字符串                            | 已覆盖：test_rendezvous_with_group_obj / test_rendezvous_with_group_name |
| 参数类型         | tensor:Tensor；group:ProcessGroup|str                         | 已覆盖：test_invalid_group_type                |
| 传参与不传参     | 两个参数均为必填                                              | 已覆盖：test_missing_args_raises               |
| 等价类/边界值    | 1D tensor vs 2D tensor                                        | 已覆盖：test_rendezvous_with_group_obj / test_rendezvous_2d |
| 正常传参场景     | 返回 _SymmetricMemory                                         | 已覆盖：test_rendezvous_with_group_obj         |
| 异常传参场景     | group 类型错误 -> TypeError；不再容忍后端 RuntimeError        | 已覆盖：用例直接 ERROR/FAIL                    |
| 混合设备类型     | 仅接受单 tensor 输入，无 NPU/CPU 混合场景                      | 未覆盖：API 仅取单 tensor                      |

未覆盖项及原因：
- 混合设备类型：rendezvous 仅接受单 tensor 输入。

注意：_SymmetricMemory 主要为 CUDA/NCCL/NVSHMEM 实现。本测试不再容忍后端未实现错误：
     若在 NPU/HCCL 上 rendezvous 抛出 RuntimeError/NotImplementedError，
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

from torch.distributed._symmetric_memory import (
    empty as symm_empty,
    rendezvous as symm_rendezvous,
)


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29512')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_rendezvous_with_group_obj(rank, world_size, device_name):
    """rendezvous accepts a ProcessGroup object."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty(8, device=dev)
        sm = symm_rendezvous(t, dist.group.WORLD)
        assert sm is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_rendezvous_with_group_name(rank, world_size, device_name):
    """rendezvous accepts a group name string."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty(8, device=dev)
        group_name = dist.group.WORLD.group_name
        sm = symm_rendezvous(t, group_name)
        assert sm is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_rendezvous_2d(rank, world_size, device_name):
    """rendezvous works with 2D tensors."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = symm_empty(2, 4, device=dev)
        sm = symm_rendezvous(t, dist.group.WORLD)
        assert sm is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_invalid_group_type(rank, world_size, device_name):
    """Passing an int as group should raise TypeError."""
    _init_dist_hccl(rank, world_size)
    try:
        dev = torch.device(f'{device_name}:{rank}')
        t = torch.empty(4, device=dev)
        raised = False
        try:
            symm_rendezvous(t, 1234)  # type: ignore[arg-type]
        except TypeError:
            raised = True
        assert raised, "Expected TypeError for unsupported group type"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestSymmetricMemoryRendezvous(TestCase):
    """Tests for torch.distributed._symmetric_memory.rendezvous."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_callable(self):
        """rendezvous is callable with documented signature."""
        self.assertTrue(callable(symm_rendezvous))
        sig = inspect.signature(symm_rendezvous)
        params = sig.parameters
        self.assertIn("tensor", params)
        self.assertIn("group", params)

    def test_missing_args_raises(self):
        """Missing required args -> TypeError."""
        raised = False
        try:
            symm_rendezvous()  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_with_group_obj(self):
        mp.spawn(_test_rendezvous_with_group_obj,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_with_group_name(self):
        mp.spawn(_test_rendezvous_with_group_name,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rendezvous_2d(self):
        mp.spawn(_test_rendezvous_2d,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_group_type(self):
        mp.spawn(_test_invalid_group_type,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
