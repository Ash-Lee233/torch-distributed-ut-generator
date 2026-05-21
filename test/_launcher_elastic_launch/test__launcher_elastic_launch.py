# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.launcher.elastic_launch 接口功能正确性
API 名称：torch.distributed.launcher.elastic_launch
API 说明：torch.distributed.launcher.elastic_launch 是
          torch.distributed.launcher.api.elastic_launch 的重导出。

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | entrypoint 为 None vs callable vs str                        | 已覆盖：test_init_callable / test_init_str / test_init_none |
| 枚举选项         | entrypoint 三种类型                                          | 已覆盖：test_init_callable / test_init_str / test_init_none |
| 参数类型         | config:LaunchConfig；entrypoint:Callable|str|None             | 已覆盖：test_init_callable                     |
| 传参与不传参     | 两个参数均为必填                                             | 已覆盖：test_missing_args                      |
| 等价类/边界值    | 仅构造检查                                                   | 已覆盖：test_attributes_stored                 |
| 正常传参场景     | _config / _entrypoint 字段正确                               | 已覆盖：test_attributes_stored                 |
| 异常传参场景     | 缺少参数 TypeError                                           | 已覆盖：test_missing_args                      |
| 重导出一致性     | launcher.elastic_launch is launcher.api.elastic_launch       | 已覆盖：test_reexport_identity                 |
| 混合设备类型     | 不接受 Tensor 输入                                            | 未覆盖：API 无 Tensor 输入                     |

未覆盖项及原因：
- 混合设备类型：elastic_launch 调度多进程 agent，不接受张量输入。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

from torch.distributed.launcher import LaunchConfig, elastic_launch


def _entrypoint(x):
    return x


class TestLauncherElasticLaunch(TestCase):
    """Tests for torch.distributed.launcher.elastic_launch (re-export)."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=1)

    def test_init_callable(self):
        launcher = elastic_launch(self.cfg, _entrypoint)
        self.assertIs(launcher._entrypoint, _entrypoint)

    def test_init_str(self):
        launcher = elastic_launch(self.cfg, "main.py")
        self.assertEqual(launcher._entrypoint, "main.py")

    def test_init_none(self):
        launcher = elastic_launch(self.cfg, None)
        self.assertIsNone(launcher._entrypoint)

    def test_attributes_stored(self):
        launcher = elastic_launch(self.cfg, _entrypoint)
        self.assertIs(launcher._config, self.cfg)
        self.assertIs(launcher._entrypoint, _entrypoint)

    def test_missing_args(self):
        raised_no_cfg = False
        try:
            elastic_launch()  # type: ignore[call-arg]
        except TypeError:
            raised_no_cfg = True
        self.assertTrue(raised_no_cfg)

        raised_no_ep = False
        try:
            elastic_launch(self.cfg)  # type: ignore[call-arg]
        except TypeError:
            raised_no_ep = True
        self.assertTrue(raised_no_ep)

    def test_is_class(self):
        self.assertTrue(inspect.isclass(elastic_launch))

    def test_reexport_identity(self):
        from torch.distributed.launcher.api import elastic_launch as ApiElastic
        self.assertIs(elastic_launch, ApiElastic)


if __name__ == "__main__":
    run_tests()
