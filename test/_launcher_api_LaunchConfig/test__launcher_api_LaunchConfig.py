# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.api.LaunchConfig 接口功能正确性
API 名称：torch.distributed.launcher.api.LaunchConfig
API 签名（关键字段）：
  @dataclass
  class LaunchConfig:
      min_nodes: int
      max_nodes: int
      nproc_per_node: int
      logs_specs: LogsSpecs | None = None
      run_id: str = ""
      role: str = "default_role"
      rdzv_endpoint: str = ""
      rdzv_backend: str = "etcd"
      rdzv_configs: dict[str, Any] = field(default_factory=dict)
      rdzv_timeout: int = -1
      max_restarts: int = 3
      monitor_interval: float = 0.1
      start_method: str = "spawn"
      ...

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 必填位置参数 + 可选项目缺省                                  | 已覆盖：test_minimal_construction              |
| 枚举选项         | start_method spawn/fork；rdzv_backend etcd/c10d              | 已覆盖：test_start_method_options / test_rdzv_backend_options |
| 参数类型         | min_nodes/max_nodes/nproc_per_node:int；run_id:str；configs:dict | 已覆盖：test_field_types                       |
| 传参与不传参     | 仅必填 vs 全字段                                             | 已覆盖：test_minimal_construction / test_full_kwargs |
| 等价类/边界值    | rdzv_timeout 设置时同步写入 rdzv_configs；显式 timeout 已存在 | 已覆盖：test_post_init_timeout_promotion / test_existing_timeout_preserved |
| 正常传参场景     | 字段值符合输入                                               | 已覆盖：test_minimal_construction              |
| 异常传参场景     | 缺少必填字段 (min_nodes 等) -> TypeError                      | 已覆盖：test_missing_required_field_raises     |
| 重导出一致性     | launcher.LaunchConfig 与 launcher.api.LaunchConfig 同源       | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | 配置 dataclass，不接受 Tensor 输入                            | 未覆盖：API 仅为配置容器，无设备语义           |

未覆盖项及原因：
- 混合设备类型：LaunchConfig 是纯配置 dataclass，无张量与设备语义。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import dataclasses

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

from torch.distributed.launcher.api import LaunchConfig


class TestLauncherApiLaunchConfig(TestCase):
    """Tests for torch.distributed.launcher.api.LaunchConfig."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_minimal_construction(self):
        """LaunchConfig with only required fields uses documented defaults."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(cfg.min_nodes, 1)
        self.assertEqual(cfg.max_nodes, 1)
        self.assertEqual(cfg.nproc_per_node, 1)
        self.assertEqual(cfg.role, "default_role")
        self.assertEqual(cfg.start_method, "spawn")
        self.assertEqual(cfg.max_restarts, 3)
        # __post_init__ creates a DefaultLogsSpecs when logs_specs is None.
        self.assertIsNotNone(cfg.logs_specs)

    def test_full_kwargs(self):
        """All commonly used kwargs are stored as-is."""
        cfg = LaunchConfig(
            min_nodes=2,
            max_nodes=4,
            nproc_per_node=8,
            run_id="my_run",
            role="worker",
            rdzv_endpoint="127.0.0.1:29400",
            rdzv_backend="c10d",
            max_restarts=5,
            monitor_interval=0.5,
            start_method="spawn",
        )
        self.assertEqual(cfg.run_id, "my_run")
        self.assertEqual(cfg.role, "worker")
        self.assertEqual(cfg.rdzv_endpoint, "127.0.0.1:29400")
        self.assertEqual(cfg.rdzv_backend, "c10d")
        self.assertEqual(cfg.max_restarts, 5)
        self.assertAlmostEqual(cfg.monitor_interval, 0.5)

    def test_field_types(self):
        """Field types match the dataclass annotations."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=2, nproc_per_node=1)
        self.assertIsInstance(cfg.min_nodes, int)
        self.assertIsInstance(cfg.max_nodes, int)
        self.assertIsInstance(cfg.nproc_per_node, int)
        self.assertIsInstance(cfg.run_id, str)
        self.assertIsInstance(cfg.rdzv_configs, dict)
        self.assertIsInstance(cfg.start_method, str)

    def test_start_method_options(self):
        """start_method accepts spawn/fork strings."""
        for sm in ("spawn", "fork", "forkserver"):
            cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1,
                               start_method=sm)
            self.assertEqual(cfg.start_method, sm)

    def test_rdzv_backend_options(self):
        """rdzv_backend accepts common strings."""
        for backend in ("etcd", "c10d", "static"):
            cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1,
                               rdzv_backend=backend)
            self.assertEqual(cfg.rdzv_backend, backend)

    def test_post_init_timeout_promotion(self):
        """Explicit rdzv_timeout is promoted into rdzv_configs['timeout']."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1,
                           rdzv_timeout=600)
        self.assertEqual(cfg.rdzv_configs.get("timeout"), 600)

    def test_existing_timeout_preserved(self):
        """A timeout already in rdzv_configs is not overwritten by default."""
        cfg = LaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1,
            rdzv_configs={"timeout": 123},
        )
        self.assertEqual(cfg.rdzv_configs["timeout"], 123)

    def test_default_timeout_in_configs(self):
        """If neither rdzv_timeout nor configs['timeout'] set, default 900 is inserted."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(cfg.rdzv_configs.get("timeout"), 900)

    def test_monitor_interval_default(self):
        """monitor_interval defaults to 0.1."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertAlmostEqual(cfg.monitor_interval, 0.1)

    def test_run_id_default_empty(self):
        """run_id defaults to empty string."""
        cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)
        self.assertEqual(cfg.run_id, "")

    def test_missing_required_field_raises(self):
        """Missing required field (min_nodes) raises TypeError."""
        raised = False
        try:
            LaunchConfig(max_nodes=1, nproc_per_node=1)  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_is_dataclass(self):
        """LaunchConfig is decorated with @dataclass."""
        self.assertTrue(dataclasses.is_dataclass(LaunchConfig))

    def test_reexport_identity(self):
        """torch.distributed.launcher.LaunchConfig is the same class."""
        from torch.distributed.launcher import LaunchConfig as TopLevel
        self.assertIs(TopLevel, LaunchConfig)


if __name__ == "__main__":
    run_tests()
