# UT 执行总报告（本批 18 个 API）

**生成时间**：2026-05-21 11:45:19

## 执行命令（每个测试文件单独运行）

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
python -m pytest test/<dir>/test_*.py -v --tb=short --no-header -q
```

## 环境摘要

- Python 3.11.14
- torch= 2.7.1+cpu
- torch_npu= 2.7.1.post2
- npu_device_count= 8
- npu_available= True
- CANN home: /usr/local/Ascend/cann-8.5.1


## 汇总

- 文件总数：18
- 文件 PASS：18
- 文件 FAIL：0
- 用例总通过：138
- 用例总跳过：8
- 用例总失败：0
- 用例总报错：0
- 累计耗时：1181.5s

## 各 API 结果

| API | 文件结果 | PASS | SKIP | FAIL | ERROR | 耗时(s) |
|-----|----------|------|------|------|-------|---------|
| `torch.distributed._composable.fsdp.MixedPrecisionPolicy` | PASS | 10 | 0 | 0 | 0 | 35.0 |
| `torch.distributed.fsdp.MixedPrecisionPolicy` | PASS | 8 | 0 | 0 | 0 | 27.4 |
| `torch.distributed.tensor.empty` | PASS | 11 | 0 | 0 | 0 | 184.8 |
| `torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker` | PASS | 5 | 0 | 0 | 0 | 94.8 |
| `torch.distributed.launcher.api.LaunchConfig` | PASS | 13 | 0 | 0 | 0 | 10.1 |
| `torch.distributed.launcher.LaunchConfig` | PASS | 8 | 0 | 0 | 0 | 10.0 |
| `torch.distributed.launcher.api.elastic_launch` | PASS | 9 | 0 | 0 | 0 | 10.1 |
| `torch.distributed.launcher.elastic_launch` | PASS | 7 | 0 | 0 | 0 | 10.1 |
| `torch.distributed.pipelining.PipelineStage` | PASS | 8 | 0 | 0 | 0 | 142.0 |
| `torch.distributed.pipelining.ScheduleGPipe` | PASS | 7 | 0 | 0 | 0 | 129.7 |
| `torch.distributed.distributed_c10d._coalescing_manager` | PASS | 8 | 0 | 0 | 0 | 142.1 |
| `torch.distributed._coalescing_manager` | PASS | 6 | 0 | 0 | 0 | 109.0 |
| `torch.distributed.distributed_c10d.rendezvous` | PASS | 8 | 0 | 0 | 0 | 10.1 |
| `torch.distributed._symmetric_memory.empty` | PASS | 6 | 0 | 0 | 0 | 91.9 |
| `torch.distributed._symmetric_memory.enable_symm_mem_for_group` | PASS | 6 | 0 | 0 | 0 | 60.3 |
| `torch.distributed._symmetric_memory.rendezvous` | PASS | 6 | 0 | 0 | 0 | 76.1 |
| `torch.distributed.checkpoint.load_state_dict` | PASS | 10 | 0 | 0 | 0 | 27.3 |
| `torch.distributed.ProcessGroupNCCL.Options` | PASS | 2 | 8 | 0 | 0 | 10.4 |

## 跳过用例分析

| API | 测试方法 | 跳过条件 | 跳过原因 | 合理性 |
|-----|----------|----------|----------|--------|
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_config_field` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_default_construction` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_extra_positional_raises` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_high_priority_false` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_high_priority_true` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_is_high_priority_stream_field_type` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_split_color_default` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |
| `torch.distributed.ProcessGroupNCCL.Options` | `TestProcessGroupNCCLOptions::test_subclass_of_backend_options` | @unittest.skipUnless(is_nccl_available()) | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：NPU 环境下 NCCL 未编译 |

## 本批改动文件

- `test/__composable_fsdp_MixedPrecisionPolicy/` — torch.distributed._composable.fsdp.MixedPrecisionPolicy
- `test/_fsdp_MixedPrecisionPolicy/` — torch.distributed.fsdp.MixedPrecisionPolicy
- `test/_tensor_empty/` — torch.distributed.tensor.empty
- `test/__tools_fsdp2_mem_tracker_FSDPMemTracker/` — torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker
- `test/_launcher_api_LaunchConfig/` — torch.distributed.launcher.api.LaunchConfig
- `test/_launcher_LaunchConfig/` — torch.distributed.launcher.LaunchConfig
- `test/_launcher_api_elastic_launch/` — torch.distributed.launcher.api.elastic_launch
- `test/_launcher_elastic_launch/` — torch.distributed.launcher.elastic_launch
- `test/_pipelining_PipelineStage/` — torch.distributed.pipelining.PipelineStage
- `test/_pipelining_ScheduleGPipe/` — torch.distributed.pipelining.ScheduleGPipe
- `test/_distributed_c10d__coalescing_manager/` — torch.distributed.distributed_c10d._coalescing_manager
- `test/__coalescing_manager/` — torch.distributed._coalescing_manager
- `test/_distributed_c10d_rendezvous/` — torch.distributed.distributed_c10d.rendezvous
- `test/__symmetric_memory_empty/` — torch.distributed._symmetric_memory.empty
- `test/__symmetric_memory_enable_symm_mem_for_group/` — torch.distributed._symmetric_memory.enable_symm_mem_for_group
- `test/__symmetric_memory_rendezvous/` — torch.distributed._symmetric_memory.rendezvous
- `test/_checkpoint_load_state_dict/` — torch.distributed.checkpoint.load_state_dict
- `test/_ProcessGroupNCCL_Options/` — torch.distributed.ProcessGroupNCCL.Options
- `test/_PrefixStore/`（已存在，本批未改动）— `torch.distributed.PrefixStore`

注意：本次未触及 `pytorch/`、`ascend_pytorch/` 内任何源码，仅在 `test/` 下新增/修改 UT 文件。