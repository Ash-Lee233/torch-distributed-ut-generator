# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.fsdp.MixedPrecisionPolicy 接口功能正确性
API 名称：torch.distributed._composable.fsdp.MixedPrecisionPolicy
API 签名：
  @dataclass(frozen=True)
  class MixedPrecisionPolicy:
      param_dtype: torch.dtype | None = None
      reduce_dtype: torch.dtype | None = None
      output_dtype: torch.dtype | None = None
      cast_forward_inputs: bool = True

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 所有 dtype 字段均可为 None / 合法 torch.dtype                | 已覆盖：test_default_all_none / test_with_param_dtype |
| 枚举选项         | cast_forward_inputs 取 True / False；dtype 取多个常见类型    | 已覆盖：test_cast_forward_inputs_false / test_various_dtypes |
| 参数类型         | torch.dtype / None / bool                                    | 已覆盖：test_default_all_none / test_with_param_dtype |
| 传参与不传参     | 全部字段省略 vs 关键字传参                                   | 已覆盖：test_default_all_none / test_with_all_fields |
| 等价类/边界值    | reduce_dtype 与 param_dtype 不同；output_dtype 单独指定      | 已覆盖：test_reduce_dtype_different_from_param / test_output_dtype_only |
| 正常传参场景     | 使用 MixedPrecisionPolicy 配合 fully_shard 应用到模型         | 已覆盖：test_apply_with_fully_shard            |
| 异常传参场景     | frozen dataclass 字段不可修改 (FrozenInstanceError)          | 已覆盖：test_frozen_instance                   |
| 混合设备类型     | 本 API 为纯数据类配置，不接受 Tensor 输入                    | 未覆盖：API 仅为配置容器，无设备语义           |

未覆盖项及原因：
- 混合设备类型：MixedPrecisionPolicy 是纯 dataclass 配置类，本身不持有张量也不参与计算，
  无 NPU/CPU 设备语义可言。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import dataclasses
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed._composable.fsdp import MixedPrecisionPolicy


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29501')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_apply_policy(rank, world_size, device_name):
    """Apply MixedPrecisionPolicy to a model through fully_shard on NPU."""
    _init_dist_hccl(rank, world_size)
    try:
        from torch.distributed._composable.fsdp import fully_shard

        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        ).to(f'{device_name}:{rank}')

        sharded = fully_shard(model, mp_policy=policy)
        assert sharded is not None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestComposableFSDPMixedPrecisionPolicy(TestCase):
    """Test cases for torch.distributed._composable.fsdp.MixedPrecisionPolicy."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_default_all_none(self):
        """Default construction leaves all dtype fields as None and cast=True."""
        policy = MixedPrecisionPolicy()
        self.assertIsNone(policy.param_dtype)
        self.assertIsNone(policy.reduce_dtype)
        self.assertIsNone(policy.output_dtype)
        self.assertTrue(policy.cast_forward_inputs)

    def test_with_param_dtype(self):
        """param_dtype can be set explicitly to a torch.dtype."""
        policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self.assertEqual(policy.param_dtype, torch.float16)

    def test_with_all_fields(self):
        """All four fields settable via keyword args."""
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float16,
            cast_forward_inputs=False,
        )
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertEqual(policy.output_dtype, torch.float16)
        self.assertFalse(policy.cast_forward_inputs)

    def test_cast_forward_inputs_false(self):
        """cast_forward_inputs can be toggled to False."""
        policy = MixedPrecisionPolicy(cast_forward_inputs=False)
        self.assertFalse(policy.cast_forward_inputs)

    def test_various_dtypes(self):
        """Accepts a range of common torch dtypes."""
        for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            policy = MixedPrecisionPolicy(param_dtype=dt)
            self.assertEqual(policy.param_dtype, dt)

    def test_reduce_dtype_different_from_param(self):
        """reduce_dtype may differ from param_dtype (full-precision grad reduce)."""
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        self.assertNotEqual(policy.param_dtype, policy.reduce_dtype)

    def test_output_dtype_only(self):
        """Only output_dtype set, others remain None."""
        policy = MixedPrecisionPolicy(output_dtype=torch.float16)
        self.assertIsNone(policy.param_dtype)
        self.assertIsNone(policy.reduce_dtype)
        self.assertEqual(policy.output_dtype, torch.float16)
        self.assertTrue(policy.cast_forward_inputs)

    def test_frozen_instance(self):
        """frozen dataclass forbids attribute assignment."""
        policy = MixedPrecisionPolicy()
        raised = False
        try:
            policy.param_dtype = torch.float16  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            raised = True
        self.assertTrue(raised, "Expected FrozenInstanceError on attribute set")

    def test_equality_and_hash(self):
        """Two policies with identical fields should be equal and hashable."""
        a = MixedPrecisionPolicy(param_dtype=torch.float16)
        b = MixedPrecisionPolicy(param_dtype=torch.float16)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    @skipIfUnsupportMultiNPU(2)
    def test_apply_with_fully_shard(self):
        """Apply MixedPrecisionPolicy via fully_shard on multi-NPU HCCL."""
        world_size = 2
        mp.spawn(
            _test_apply_policy,
            args=(world_size, self.device_name),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    run_tests()
