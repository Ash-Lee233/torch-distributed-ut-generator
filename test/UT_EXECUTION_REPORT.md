# UT 执行总报告（本批 18 个 API）

**生成时间**：2026-05-21

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
- 文件 PASS：14
- 文件 FAIL（暴露 NPU 功能缺口）：4
- 用例总通过：130
- 用例总跳过：8
- 用例总失败：10
- 累计耗时：约 1200s

## NPU 功能缺口（需要框架/底层介入分析）

本批中以下用例在 NPU/HCCL 上 **ERROR/FAIL**。失败已不再被 `try/except` 吞掉，
错误堆栈与根因如下：

| API | 失败用例 | 根因（来自堆栈） |
|-----|---------|----------------|
| `torch.distributed._symmetric_memory.empty` | `test_empty_basic`, `test_empty_2d`, `test_empty_dtype_variants`, `test_empty_list_size`, `test_empty_explicit_device` | `RuntimeError: get_group_info: no group info associated with the group name 0` — HCCL backend 未在 c10d 注册 group_info，P2P 对称内存分配不可用 |
| `torch.distributed._symmetric_memory.rendezvous` | `test_rendezvous_with_group_obj`, `test_rendezvous_with_group_name`, `test_rendezvous_2d` | 同上（`test_invalid_group_type` 通过——TypeError 在抵达后端前抛出） |
| `torch.distributed._coalescing_manager` (`device=` 参数) | `test_explicit_args` | `RuntimeError: Backend hccl does not implement startCoalescing` — HCCL backend 未实现 `_start_coalescing` |
| `torch.distributed.distributed_c10d._coalescing_manager` (`device=` 参数) | `test_device_arg` | 同上 |

注：`torch.distributed._symmetric_memory.enable_symm_mem_for_group` 全部 PASS，
该 API 已 `@deprecated`，调用返回 None 不进入后端，无运行时缺口。

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
| `torch.distributed.distributed_c10d._coalescing_manager` | **FAIL** | 7 | 0 | 1 | 0 | 142.1 |
| `torch.distributed._coalescing_manager` | **FAIL** | 5 | 0 | 1 | 0 | 109.0 |
| `torch.distributed.distributed_c10d.rendezvous` | PASS | 8 | 0 | 0 | 0 | 10.1 |
| `torch.distributed._symmetric_memory.empty` | **FAIL** | 1 | 0 | 5 | 0 | ~90 |
| `torch.distributed._symmetric_memory.enable_symm_mem_for_group` | PASS | 6 | 0 | 0 | 0 | 60.3 |
| `torch.distributed._symmetric_memory.rendezvous` | **FAIL** | 3 | 0 | 3 | 0 | ~75 |
| `torch.distributed.checkpoint.load_state_dict` | PASS | 10 | 0 | 0 | 0 | 27.3 |
| `torch.distributed.ProcessGroupNCCL.Options` | PASS | 2 | 8 | 0 | 0 | 10.4 |

## 跳过用例分析

| API | 测试方法 | 跳过条件 | 跳过原因 | 合理性 |
|-----|----------|----------|----------|--------|
| `torch.distributed.ProcessGroupNCCL.Options` | 8 个 `test_*` | `@unittest.skipUnless(is_nccl_available())` | torch_npu 构建未编译 NCCL，`ProcessGroupNCCL` 不可用 | 合理：编译期排除，非运行时缺口 |

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
