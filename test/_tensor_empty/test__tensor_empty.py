# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.empty 接口功能正确性
API 名称：torch.distributed.tensor.empty
API 签名：empty(
              *size,
              dtype: Optional[torch.dtype] = None,
              layout: torch.layout = torch.strided,
              requires_grad: bool = False,
              device_mesh: Optional[DeviceMesh] = None,
              placements: Optional[Sequence[Placement]] = None,
          ) -> DTensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 不同 shape；含 1D / 2D / 3D                                  | 已覆盖：test_empty_1d / test_empty_replicate / test_empty_3d |
| 枚举选项         | placements: Replicate() / Shard(0) / Shard(1)                | 已覆盖：test_empty_replicate / test_empty_shard0 / test_empty_shard1 |
| 参数类型         | dtype: None / float16 / int32；size 为 varargs / list / tuple| 已覆盖：test_empty_dtype_float16 / test_empty_dtype_int32 / test_empty_list_size / test_empty_tuple_size |
| 传参与不传参     | requires_grad 默认 vs 显式 True                              | 已覆盖：test_empty_requires_grad               |
| 等价类/边界值    | 单元素 shape；大小可被 world_size 整除                       | 已覆盖：test_empty_single_element / test_empty_shard0 |
| 正常传参场景     | 返回 DTensor，shape/dtype/device 符合预期                    | 已覆盖：test_empty_replicate                   |
| 异常传参场景     | 合法 DeviceMesh + 合法 placements 下无稳定异常路径            | 未覆盖：原因如上                              |
| 混合设备类型     | DTensor 由 DeviceMesh 决定设备，不接受单独的 CPU/NPU 张量参数 | 未覆盖：API 通过 mesh 统一设备语义             |

未覆盖项及原因：
- 异常传参场景：合法参数组合下无稳定异常路径；非法参数由 DeviceMesh / placements 内部约束保证。
- 混合设备类型：empty 不接受张量入参，设备完全由 DeviceMesh 决定。

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

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, empty
from torch.distributed.tensor.placement_types import Replicate, Shard


def _init_dist(rank, world_size, fn, *args):
    """Init HCCL process group then dispatch to test fn."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29503')
    torch_npu.npu.set_device(rank)
    dist.init_process_group('hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _test_empty_replicate(rank, world_size):
    """empty + Replicate returns DTensor with full local shape."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(4, 8, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor), f"Expected DTensor, got {type(t)}"
    assert t.shape == torch.Size([4, 8]), f"Expected [4,8], got {t.shape}"
    assert t.device.type == device_type, f"Unexpected device {t.device}"
    local = t.to_local()
    assert local.shape == torch.Size([4, 8])


def _test_empty_shard0(rank, world_size):
    """empty + Shard(0) splits along dim 0."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[0] == 4 // world_size
    assert local.shape[1] == 8


def _test_empty_shard1(rank, world_size):
    """empty + Shard(1) splits along dim 1."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(4, 8, device_mesh=mesh, placements=[Shard(1)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[0] == 4
    assert local.shape[1] == 8 // world_size


def _test_empty_dtype_float16(rank, world_size):
    """empty with dtype=float16 returns float16 DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(4, 4, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16


def _test_empty_dtype_int32(rank, world_size):
    """empty with dtype=int32 returns int32 DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(6, dtype=torch.int32, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.int32


def _test_empty_requires_grad(rank, world_size):
    """empty with requires_grad=True returns differentiable DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(4, 4, dtype=torch.float32, requires_grad=True,
              device_mesh=mesh, placements=[Replicate()])
    assert t.requires_grad is True


def _test_empty_list_size(rank, world_size):
    """empty accepts a list shape."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty([4, 8, 2], device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 8, 2])


def _test_empty_tuple_size(rank, world_size):
    """empty accepts a tuple shape."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty((4, 4), device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 4])


def _test_empty_1d(rank, world_size):
    """empty creates a 1D DTensor correctly."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(16, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([16])
    assert isinstance(t, DTensor)


def _test_empty_3d(rank, world_size):
    """empty creates a 3D DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(2, 4, 8, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([2, 4, 8])


def _test_empty_single_element(rank, world_size):
    """empty with a single-element shape returns a 1-elem DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = empty(1, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([1])


class TestDTensorEmpty(TestCase):
    """Tests for torch.distributed.tensor.empty."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_empty_replicate(self):
        mp.spawn(_init_dist, args=(2, _test_empty_replicate), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_shard0(self):
        mp.spawn(_init_dist, args=(2, _test_empty_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_shard1(self):
        mp.spawn(_init_dist, args=(2, _test_empty_shard1), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_dtype_float16(self):
        mp.spawn(_init_dist, args=(2, _test_empty_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_dtype_int32(self):
        mp.spawn(_init_dist, args=(2, _test_empty_dtype_int32), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_requires_grad(self):
        mp.spawn(_init_dist, args=(2, _test_empty_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_list_size(self):
        mp.spawn(_init_dist, args=(2, _test_empty_list_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_tuple_size(self):
        mp.spawn(_init_dist, args=(2, _test_empty_tuple_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_1d(self):
        mp.spawn(_init_dist, args=(2, _test_empty_1d), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_3d(self):
        mp.spawn(_init_dist, args=(2, _test_empty_3d), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_empty_single_element(self):
        mp.spawn(_init_dist, args=(2, _test_empty_single_element), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
