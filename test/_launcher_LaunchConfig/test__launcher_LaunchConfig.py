# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.LaunchConfig 接口功能正确性
API 名称：torch.distributed.launcher.LaunchConfig
API 说明：torch.distributed.launcher.LaunchConfig 是
          torch.distributed.launcher.api.LaunchConfig 的重导出。
         本文件聚焦：验证导出可用、字段语义与底层一致。

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 必填 vs 全字段                                               | 已覆盖：test_minimal_construction              |
| 枚举选项         | start_method spawn/fork                                      | 已覆盖：test_start_method                      |
| 参数类型         | int / str / dict                                             | 已覆盖：test_field_types                       |
| 传参与不传参     | 仅必填 vs 关键字传入                                         | 已覆盖：test_minimal_construction / test_named_args |
| 等价类/边界值    | rdzv_timeout 写入 rdzv_configs                               | 已覆盖：test_rdzv_timeout_promotion            |
| 正常传参场景     | 字段值与输入一致                                             | 已覆盖：test_named_args                        |
| 异常传参场景     | 缺少必填字段 -> TypeError                                     | 已覆盖：test_missing_required_field_raises     |
| 重导出一致性     | launcher.LaunchConfig is launcher.api.LaunchConfig            | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | 纯配置 dataclass，不接受 Tensor 输入                          | 未覆盖：API 无设备语义                         |

未覆盖项及原因：
- 混合设备类型：LaunchConfig 是配置 dataclass，无张量与设备语义。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import dataclasses

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

from torch.distributed.launcher import LaunchConfig


class TestLauncherLaunchConfig(TestCase):
    """Tests for torch.distributed.launcher.LaunchConfig (re-export)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_minimal_construction(self):
        """Construction with the three required positional args."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=2, nproc_per_node=4)
        self.assertEqual(cfg.min_nodes, 1)
        self.assertEqual(cfg.max_nodes, 2)
        self.assertEqual(cfg.nproc_per_node, 4)

    def test_named_args(self):
        """Selected kwargs propagate to the dataclass."""
        cfg = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="abc",
            rdzv_backend="c10d",
            role="trainer",
        )
        self.assertEqual(cfg.run_id, "abc")
        self.assertEqual(cfg.rdzv_backend, "c10d")
        self.assertEqual(cfg.role, "trainer")

    def test_start_method(self):
        """start_method accepted as spawn/fork."""
        for sm in ("spawn", "fork"):
            cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1,
                               start_method=sm)
            self.assertEqual(cfg.start_method, sm)

    def test_field_types(self):
        """Stored field types match annotations."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertIsInstance(cfg.min_nodes, int)
        self.assertIsInstance(cfg.rdzv_configs, dict)
        self.assertIsInstance(cfg.role, str)

    def test_rdzv_timeout_promotion(self):
        """rdzv_timeout writes into rdzv_configs."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1,
                           rdzv_timeout=120)
        self.assertEqual(cfg.rdzv_configs.get("timeout"), 120)

    def test_missing_required_field_raises(self):
        """Missing required field raises TypeError."""
        raised = False
        try:
            LaunchConfig(max_nodes=1, nproc_per_node=1)  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_is_dataclass(self):
        """Class is a dataclass."""
        self.assertTrue(dataclasses.is_dataclass(LaunchConfig))

    def test_reexport_identity(self):
        """Re-export points to the same dataclass."""
        from torch.distributed.launcher.api import LaunchConfig as ApiLaunchConfig
        self.assertIs(LaunchConfig, ApiLaunchConfig)


if __name__ == "__main__":
    run_tests()
