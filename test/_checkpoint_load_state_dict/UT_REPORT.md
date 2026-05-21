# UT 执行报告：`torch.distributed.checkpoint.load_state_dict`

**生成时间**：2026-05-21 11:45:19

## 执行命令

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


## 统计

- 通过: 10
- 跳过: 0
- 失败: 0
- 报错: 0
- 耗时: 27.3s
- pytest 概要: `======================== 10 passed, 1 warning in 27.30s ========================`

## 测试结果

| 测试方法 | 结果 |
|----------|------|

## 本次改动文件

- `test/_checkpoint_load_state_dict/test_checkpoint_load_state_dict.py` 等同目录文件