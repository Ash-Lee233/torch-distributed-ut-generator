# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.api.elastic_launch 接口功能正确性
API 名称：torch.distributed.launcher.api.elastic_launch
API 签名：
  class elastic_launch:
      def __init__(
          self,
          config: LaunchConfig,
          entrypoint: Callable | str | None,
      ) -> None
      def __call__(self, *args) -> dict[int, Any]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | entrypoint 为 None vs callable vs str                        | 已覆盖：test_init_none_entrypoint / test_init_callable_entrypoint / test_init_str_entrypoint |
| 枚举选项         | entrypoint 三种类型                                          | 已覆盖：test_init_none_entrypoint 等           |
| 参数类型         | config 为 LaunchConfig；entrypoint 为 Callable|str|None       | 已覆盖：test_init_callable_entrypoint          |
| 传参与不传参     | 两个参数均为必填                                             | 已覆盖：test_missing_config_raises             |
| 等价类/边界值    | 仅构造，无超出范围的输入                                     | 已覆盖：test_init_callable_entrypoint          |
| 正常传参场景     | 构造后存有 _config / _entrypoint 字段                        | 已覆盖：test_attributes_stored                 |
| 异常传参场景     | 缺少 config 参数 -> TypeError                                | 已覆盖：test_missing_config_raises             |
| 重导出一致性     | launcher.elastic_launch is launcher.api.elastic_launch       | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | 该类调度多进程 agent，不直接接受 Tensor                        | 未覆盖：API 无 Tensor 输入                     |

未覆盖项及原因：
- 混合设备类型：elastic_launch 调度多进程 agent，不接受张量输入。
- __call__ 实际启动 LocalElasticAgent，需要真实 rendezvous 后端与多进程环境，
  且会带来强副作用；在单进程 UT 中仅做构造与属性校验。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

from torch.distributed.launcher.api import LaunchConfig, elastic_launch


def _entrypoint(x):
    return x


class TestLauncherApiElasticLaunch(TestCase):
    """Tests for torch.distributed.launcher.api.elastic_launch."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

    def test_init_callable_entrypoint(self):
        """Callable entrypoint is stored."""
        launcher = elastic_launch(self.cfg, _entrypoint)
        self.assertIs(launcher._entrypoint, _entrypoint)

    def test_init_str_entrypoint(self):
        """String entrypoint is stored."""
        launcher = elastic_launch(self.cfg, "script.py")
        self.assertEqual(launcher._entrypoint, "script.py")

    def test_init_none_entrypoint(self):
        """None entrypoint is allowed."""
        launcher = elastic_launch(self.cfg, None)
        self.assertIsNone(launcher._entrypoint)

    def test_attributes_stored(self):
        """_config and _entrypoint are properly stored."""
        launcher = elastic_launch(self.cfg, _entrypoint)
        self.assertIs(launcher._config, self.cfg)
        self.assertIs(launcher._entrypoint, _entrypoint)

    def test_call_signature(self):
        """__call__ is a method that accepts *args."""
        launcher = elastic_launch(self.cfg, _entrypoint)
        self.assertTrue(callable(launcher))
        sig = inspect.signature(launcher.__call__)
        # *args param expected
        kinds = {p.kind for p in sig.parameters.values()}
        self.assertIn(inspect.Parameter.VAR_POSITIONAL, kinds)

    def test_missing_config_raises(self):
        """Missing required config arg raises TypeError."""
        raised = False
        try:
            elastic_launch()  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_missing_entrypoint_raises(self):
        """Missing entrypoint arg raises TypeError."""
        raised = False
        try:
            elastic_launch(self.cfg)  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_is_class(self):
        """elastic_launch is a class (despite the lowercase name)."""
        self.assertTrue(inspect.isclass(elastic_launch))

    def test_reexport_identity(self):
        """launcher.elastic_launch is the same class as launcher.api.elastic_launch."""
        from torch.distributed.launcher import elastic_launch as TopLevel
        self.assertIs(TopLevel, elastic_launch)


if __name__ == "__main__":
    run_tests()
