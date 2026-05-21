# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp.MixedPrecisionPolicy 接口功能正确性
API 名称：torch.distributed.fsdp.MixedPrecisionPolicy
API 签名：
  @dataclass(frozen=True)
  class MixedPrecisionPolicy:
      param_dtype: torch.dtype | None = None
      reduce_dtype: torch.dtype | None = None
      output_dtype: torch.dtype | None = None
      cast_forward_inputs: bool = True

说明：torch.distributed.fsdp.MixedPrecisionPolicy 是
     torch.distributed._composable.fsdp.MixedPrecisionPolicy 的重导出。
     本文件聚焦：验证 fsdp 顶层入口名可用、字段语义与底层实现一致、
     可与 fully_shard 协同。

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 字段为 None / 合法 torch.dtype                              | 已覆盖：test_default_construction              |
| 枚举选项         | cast_forward_inputs True/False；dtype 多种                  | 已覆盖：test_with_full_kwargs                  |
| 参数类型         | torch.dtype / None / bool                                   | 已覆盖：test_dtype_field_types                 |
| 传参与不传参     | 全部省略 vs 关键字显式传                                    | 已覆盖：test_default_construction / test_with_full_kwargs |
| 等价类/边界值    | reduce_dtype 与 param_dtype 不同；output_dtype 单独指定     | 已覆盖：test_distinct_reduce_dtype             |
| 正常传参场景     | 与 fully_shard 联合使用                                     | 已覆盖：test_apply_with_fully_shard            |
| 异常传参场景     | frozen dataclass 字段不可修改                               | 已覆盖：test_frozen_instance                   |
| 重导出一致性     | 顶层与 _composable.fsdp 入口指向同一类                       | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | 纯 dataclass 配置，不接受 Tensor 输入                       | 未覆盖：API 仅为配置容器，无设备语义           |

未覆盖项及原因：
- 混合设备类型：MixedPrecisionPolicy 是配置 dataclass，无张量与设备语义。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import dataclasses

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed.fsdp import MixedPrecisionPolicy


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29502')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_apply_policy(rank, world_size, device_name):
    """Use MixedPrecisionPolicy from fsdp namespace with fully_shard."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed.fsdp import fully_shard

        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 8),
        ).to(f'{device_name}:{rank}')

        sharded = fully_shard(model, mp_policy=policy)
        assert sharded is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestFSDPMixedPrecisionPolicy(TestCase):
    """Test cases for torch.distributed.fsdp.MixedPrecisionPolicy."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_default_construction(self):
        """All fields default to None / True."""
        policy = MixedPrecisionPolicy()
        self.assertIsNone(policy.param_dtype)
        self.assertIsNone(policy.reduce_dtype)
        self.assertIsNone(policy.output_dtype)
        self.assertTrue(policy.cast_forward_inputs)

    def test_with_full_kwargs(self):
        """All fields explicitly set via keyword args."""
        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertEqual(policy.output_dtype, torch.bfloat16)
        self.assertFalse(policy.cast_forward_inputs)

    def test_dtype_field_types(self):
        """Field values are stored as-is."""
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self.assertIsInstance(policy.param_dtype, torch.dtype)
        self.assertIsInstance(policy.cast_forward_inputs, bool)

    def test_distinct_reduce_dtype(self):
        """reduce_dtype may legitimately differ from param_dtype."""
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        self.assertNotEqual(policy.param_dtype, policy.reduce_dtype)

    def test_frozen_instance(self):
        """Assignment to fields should raise FrozenInstanceError."""
        policy = MixedPrecisionPolicy()
        raised = False
        try:
            policy.reduce_dtype = torch.float32  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            raised = True
        self.assertTrue(raised)

    def test_reexport_identity(self):
        """fsdp.MixedPrecisionPolicy is the same class as _composable.fsdp version."""
        from torch.distributed._composable.fsdp import (
            MixedPrecisionPolicy as ComposableMP,
        )
        self.assertIs(MixedPrecisionPolicy, ComposableMP)

    def test_equality(self):
        """Identical-field policies compare equal."""
        a = MixedPrecisionPolicy(param_dtype=torch.float16, cast_forward_inputs=False)
        b = MixedPrecisionPolicy(param_dtype=torch.float16, cast_forward_inputs=False)
        self.assertEqual(a, b)

    @skipIfUnsupportMultiNPU(2)
    def test_apply_with_fully_shard(self):
        """Policy integrates with fully_shard on multi-NPU."""
        world_size = 2
        mp.spawn(
            _test_apply_policy,
            args=(world_size, self.device_name),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
