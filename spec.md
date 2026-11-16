# Proactive AssetOpsBench — 通用基准基础设施规范 (`spec.md` v3.0)

- **规范版本**：`v3.0`（`SCHEMA_VERSION = "3.0"`，100% 向下兼容 `v2.0` 场景与 GT）
- **定位与目标**：面向 **IBM AssetOpsBench** 全场景适配的**完整主动运维基准评测基础设施（Universal AssetOpsBench Benchmark Infrastructure, `bmk infra`）**，原生覆盖原版 4 大专家域（**IoT、TSFM、FMSR/FMEA、WorkOrder**）及**主动治理（Proactive Governance）**与**声明式自定义控制工具（`custom_tools`）**，使任务 2/3/4/5 团队无需修改任何 `framework/` 代码即可直接转换、校验、回放、评分、重放压测并生成 Cache OFF/ON 对照实验报告。
- **工程红线**：**100% Python 3.10+ 标准库、零第三方 `pip` 依赖、零真值/阈值剧透泄露、全会话字节级前缀不变（`messages_n[:-2] == messages_{n-1}`）、默认 `--concurrency 1` 运行内置双示例任务峰值内存 `< 50 MB` RSS**。

---

## 1. 职责边界与 7 大标准库 CLI 全景

### 1.1 框架模块与 7 个 CLI 职责
1. **多格式零泄露转换器 (`framework/convert.py`)**：支持宽表（`wide`）与窄表（`long`）传感器 CSV、外部目录文件（`fmea_manuals.json`、`workorders.json`）、自动等步长 `tick_grid` 生成、原版 AssetOpsBench 溯源元数据（`original_assetops_meta`）及强制标签黑名单剥离（`FORBIDDEN_LABEL_COLUMNS`），输出物理隔离的 `scenarios/public/*.scenario.json` 与 `scenarios/gt/*.gt.json`。
2. **场景与 GT 静态校验器 (`framework/validate.py`)**：独立校验公开场景与隐藏 GT 的 Schema 完整性、跨文件 `obs_id`/工具名引用一致性、时间轴严格递增性、以及观测载荷与 `messages[0]` 的硬性零泄露检查。
3. **有界多步主动回放套件 (`framework/run.py`)**：支持 `--scenarios` 或 `--scenarios-dir` + `--tags` 批量筛选、单 tick 有界多步推理（`--max-steps-per-tick`）、会话级可变设备/工单世界状态（`WorldState`）、跨场景并发调度（`--concurrency`），输出 `run_log.jsonl` 与 `run_summary.json`。
4. **可组合确定性离线评分器 (`framework/score.py`)**：兼容 v2.0 并在 `completion_check` 中支持 6 种可组合评测模式（`tool_call`、`any_of`、`all_of`、`ordered_sequence`、`world_state`、`diagnosis_only`）及静默轮只读查询豁免开关，输出 `score_report.json`。
5. **轨迹与动作摘要导出器 (`framework/export.py`)**：流式导出固定请求序列 `replay_requests.jsonl` 与精简动作流 `exported_actions.jsonl`。
6. **固定轨迹并发重放压测器 (`framework/replay.py`)**：跨 `session_id` 并发、会话内严格按 `seq_idx` 串行等待重放，支持预热（`--warmup-sessions`）与多轮重复（`--repeats`），输出 `replay_benchmark_report.json`。
7. **Cache OFF/ON 实验对照分析器 (`framework/compare.py`)**：对比 OFF 与 ON 两组实验的 TTFT/延迟降幅、RPS 加速比、实测 `cached_tokens` 命中率与理论前缀复用率、实时轨迹一致性及主动性 F1/验证不退化指标，输出 `comparison_report.json` 及可选 Markdown 报告。

### 1.2 明确不做
- 不引入任何外部库（`pandas`、`numpy`、`requests`、`httpx`、`pydantic` 等）或数据库服务；HTTP 长连接直接基于标准库 `http.client` + `threading.local` 实现。
- 不使用 LLM 裁判：`validate.py`、`score.py`、`compare.py` 均为 100% 确定性算法。

---

## 2. 仓库目录结构

```text
proactive-assetops/
├── README.md                        # 根目录 7 个 CLI 复跑、AssetOpsBench 换数据与 A/B 对比说明
├── spec.md                          # 本设计规范（v3.0 冻结契约）
├── framework/
│   ├── __init__.py                  # SCHEMA_VERSION = "3.0"、支持版本集、时间解析与多资产/子集匹配辅助
│   ├── prompt.py                    # 20 类全局故障分类表 + extra_fault_taxonomy + 严格交替消息构造
│   ├── client.py                    # 线程局部 Keep-Alive 流式 VLLMClient + 全特性零硬编码 MockClient
│   ├── tools.py                     # 4 大专家域 10 个内置工具 + WorldState + custom_tools + 条件审批策略
│   ├── convert.py                   # CLI 1: 宽/窄表 CSV + 外部目录 + tick_grid -> scenarios/{public,gt}
│   ├── validate.py                  # CLI 2: 场景/GT 结构、引用完整性与零泄露校验 -> exit 0/1
│   ├── run.py                       # CLI 3: 场景目录/标签筛选 + 多步回放 -> run_log.jsonl + run_summary.json
│   ├── score.py                     # CLI 4: 6 模式可组合确定性离线评分 -> score_report.json
│   ├── export.py                    # CLI 5: 流式导出 -> replay_requests.jsonl + exported_actions.jsonl
│   ├── replay.py                    # CLI 6: 跨会话并发固定轨迹重放压测 -> replay_benchmark_report.json
│   └── compare.py                   # CLI 7: Cache OFF vs ON 性能/一致性/质量对比 -> comparison_report.{json,md}
├── raw_data/
│   ├── task_01_chiller6/            # 示例 1: 双冷机监测、标签剥离、计划切负荷消警、单点毛刺与历史已关单工单
│   │   ├── telemetry.csv
│   │   └── raw_task_spec.json
│   └── task_02_pump3/               # 示例 2: 双水泵监测、大写 Timestamp、多候选 FMEA 与跨轮紧急停机审批
│       ├── telemetry.csv
│       └── raw_task_spec.json
├── scenarios/
│   ├── public/                      # convert.py 公开产物（run.py 只读此处）
│   └── gt/                          # 物理隔离的隐藏 GT（仅 validate.py 与 score.py 可读）
└── tests/
    └── test_pipeline.py             # 标准库 unittest 全套 AssetOpsBench 原型与基础设施验收测试
```

---

## 3. AssetOpsBench 数据转换、公开场景、4 大专家工具集与可组合 GT 契约

### 3.1 时间戳、模态与半开时间释放窗口
- **支持 Schema 版本**：`SUPPORTED_SCHEMA_VERSIONS = {"2.0", "3.0"}`（新生成文件统一写 `"3.0"`）。
- **时间戳**：UTC ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`，统一由 `datetime.fromisoformat(ts.replace("Z", "+00:00"))` 解析；`replay_ticks` 必须严格递增。
- **模态集合**：`VALID_MODALITIES = {"telemetry", "alarm", "maintenance_log", "workorder_history"}`。
- **观测释放窗口**：第 `0` tick 释放 `release_time <= replay_ticks[0]`；第 `k >= 1` tick 释放 `replay_ticks[k-1] < release_time <= replay_ticks[k]`；同 tick 内 `step_idx >= 1` 的 `new_observations` 恒为 `[]`；任意时刻可见观测满足 `release_time <= current_replay_time`。

### 3.2 多格式原始转换器 (`framework/convert.py`) 与静态校验器 (`framework/validate.py`)

#### (1) `convert.py` 支持特性
1. **宽表与窄表 CSV（`convert_config.csv_format = "wide" | "long"`，默认 `"wide"`）**：
   - `column_mapping` 支持映射 `timestamp`（缺省自动识别 `timestamp`/`Timestamp`/`datetime`/`DateTime`/`time`）、`asset_id`（缺省回退至 `asset_info.asset_id`）、`source_type`、`source_ref`、`metrics`（原列名/原指标名 -> 标准指标名），窄表额外支持映射 `metric_name`（自动识别 `metric_name`/`metric`/`sensor`/`tag`）与 `metric_value`（自动识别 `metric_value`/`value`/`reading`）。
   - **窄表聚合规则（`csv_format == "long"`）**：按 `(release_time, asset_id, source_type, source_ref)` 保序归并同一时刻同一设备的多个传感器行，合并为单条 `modality="telemetry"` 观测字典。
   - **强制零泄露剥离**：定义大小写不敏感黑名单：
     ```python
     FORBIDDEN_LABEL_COLUMNS = {
         "label", "anomaly", "is_anomaly", "anomaly_label", "fault",
         "fault_type", "fault_mode", "gt", "ground_truth", "target", "state_label",
     }
     ```
     宽表列名或窄表行指标名（映射前或映射后）一旦小写命中 `FORBIDDEN_LABEL_COLUMNS` 或 `convert_config.strip_columns`，无条件剥离（优先级高于 `sensor_columns` 白名单）；写出前对合并 `extra_events` 后的全部 `observations[*].payload` 执行硬断言，任何违禁键立即抛 `ValueError`。
2. **外部专家知识库文件自动归并**：
   - 若 `raw_data/<id>/fmea_manuals.json` 存在，读取其列表追加至 `tool_catalog_data["fmea_manuals"]`；
   - 若 `raw_data/<id>/workorders.json` 存在，读取其列表追加至 `tool_catalog_data["historical_workorders"]`。
3. **自动等步长回放网格（`tick_grid`）**：
   - 当 `public_spec.replay_ticks` 省略或为空时，读取 `public_spec`（转换后移除）或 `convert_config` 中的 `tick_grid`：
     - 指定起止：`{"start": "2026-05-10T08:00:00Z", "end": "2026-05-10T09:00:00Z", "step_seconds": 900}`，按 `step_seconds`（`> 0`）生成闭区间等步长 UTC 时间戳序列；
     - 自动推导：`{"auto_from_observations": true, "step_seconds": 900}`，以全部观测最小/最大 `release_time` 为界生成等步长序列（若末个网格点 `< max_time`，额外追加一步 `last + step_seconds` 确保覆盖全部观测）。
4. **AssetOpsBench 溯源与扩展字段透传**：
   - `public_spec` 可选透传：`tags`（`list[str]`）、`original_assetops_meta`（`dict`，记录原版 `dataset`/`utterance_id`/`uuid`/`category`/`type`，严禁含答案或标签键）、`custom_tools`（`list[dict]`）、`extra_fault_taxonomy`（`list[str]`）。
   - `gt_spec` 可选透传：`allow_readonly_query_in_no_action_ticks`（`bool`，默认 `false`）。

#### (2) CLI 2：`framework/validate.py` 校验契约
命令：`python3 -m framework.validate --scenarios-dir scenarios/public --gt-dir scenarios/gt [--out validation_report.json]`
- 逐对检查 `scenarios/public/<id>.scenario.json` 与 `scenarios/gt/<id>.gt.json`：
  1. 配对完整性、`scenario_id` 一致性、`schema_version in {"2.0", "3.0"}`；
  2. `asset_info` 与 `assets` 字段合法（`normal_ranges` 为 `[low, high]` 且 `low <= high`；无 `"role"`/`"is_faulty"` 等剧透字段；`original_assetops_meta` 无 `FORBIDDEN_LABEL_COLUMNS | {"answer", "expected_output", "gold_tool"}`）；
  3. `replay_ticks` 非空且严格递增；`observations` 的 `obs_id` 唯一、`modality in VALID_MODALITIES`、且 `payload` 零 `FORBIDDEN_LABEL_COLUMNS` 键；
  4. 调用 `build_system_message(scenario)` 校验 `messages[0]` 不含 `fmea_manuals`、`symptom_metric`、`threshold_high`、`threshold_low`、`recommended_tool`、`historical_workorders`；
  5. `permissions`、`custom_tools`、`gt.no_action_ticks`（须为合法时间且属于 `replay_ticks`）与每个 `gt.needs` 完整合法（`asset_id` 存在于场景资产表，`visible_from <= valid_window.start <= valid_window.end`，`supporting_obs_ids` 全部存在于 `observations`，`allowed_actions` 与递归 `completion_check` 引用的模式及工具名均合法）。
- 输出 `{"valid": bool, "checked_scenarios": int, "error_count": int, "errors": [{"scenario_id": str, "message": str}]}`；`valid == True` 退出码为 `0`，否则退出码为 `1`。

### 3.3 公开场景格式 (`scenarios/public/<scenario_id>.scenario.json`)
完全兼容 v2.0，新增可选字段均具备安全默认值：
```json
{
  "schema_version": "3.0",
  "scenario_id": "task_01_chiller6",
  "tags": ["iot", "tsfm", "fmsr", "workorder", "chiller", "multi_asset"],
  "original_assetops_meta": {
    "dataset": "IBM/AssetOpsBench",
    "category": "multi_agent_proactive_fmea",
    "utterance_id": "aob_chiller6_adapted_01"
  },
  "asset_info": {
    "site_id": "plant_01",
    "asset_id": "chiller_6",
    "asset_type": "centrifugal_chiller",
    "sensor_units": {"evap_approach_temp_c": "C", "chw_flow_lps": "L/s"},
    "normal_ranges": {"evap_approach_temp_c": [1.0, 2.5], "chw_flow_lps": [40.0, 55.0]}
  },
  "assets": [
    {
      "site_id": "plant_01",
      "asset_id": "chiller_6",
      "asset_type": "centrifugal_chiller",
      "initial_status": "running",
      "sensor_units": {"evap_approach_temp_c": "C", "chw_flow_lps": "L/s"},
      "normal_ranges": {"evap_approach_temp_c": [1.0, 2.5], "chw_flow_lps": [40.0, 55.0]}
    },
    {
      "site_id": "plant_01",
      "asset_id": "chiller_7",
      "asset_type": "centrifugal_chiller",
      "initial_status": "running",
      "sensor_units": {"evap_approach_temp_c": "C", "chw_flow_lps": "L/s"},
      "normal_ranges": {"evap_approach_temp_c": [1.0, 2.5], "chw_flow_lps": [40.0, 55.0]}
    }
  ],
  "extra_fault_taxonomy": [],
  "custom_tools": [],
  "persistent_goals": ["..."],
  "permissions": {
    "has_permission_constraints": false,
    "allowed_decisions": ["wait", "query", "investigate", "request_approval", "execute"],
    "approval_required_tools": [],
    "forbidden_tools": [],
    "approval_policy": {}
  },
  "replay_ticks": ["2026-05-10T08:00:00Z", "..."],
  "observations": ["..."],
  "tool_catalog_data": {
    "fmea_manuals": ["..."],
    "historical_workorders": [
      {
        "workorder_id": "HIST-WO-001",
        "asset_id": "chiller_6",
        "fault_type": "sensor_calibration_drift",
        "priority": "normal",
        "status": "closed",
        "created_at": "2026-04-15T10:00:00Z",
        "summary": "Quarterly approach temp sensor calibration completed"
      }
    ]
  }
}
```

### 3.4 4 大专家域 10 个内置工具、`WorldState` 与声明式 `custom_tools` (`framework/tools.py`)

#### (1) 会话级可变世界状态 `WorldState`
每个场景会话的 `ToolExecutor` 维护独立的 `self.world_state`：
```json
{
  "assets": {
    "chiller_6": {"status": "running"},
    "chiller_7": {"status": "running"}
  },
  "workorders": [],
  "approvals": {}
}
```
- 初始化：对 `asset_map` 中每个资产，以 `{"status": asset.get("initial_status", "running"), **(asset.get("initial_state") or {})}` 初始化 `world_state["assets"][asset_id]`。
- 状态变更：工单创建/更新、审批通过及声明式 `custom_tools` 执行均确定性原地更新 `world_state`。

#### (2) 10 个内置标准专家工具契约（全部受 `release_time <= current_replay_time` 严格时间门控）

| 专家域 | 工具名 (`tool_call.name`) | 适用 `decision` | 入参 `args` 与返回值契约 |
| :--- | :--- | :--- | :--- |
| **IoT** | `query_iot_metadata` | `query` | `{"asset_id": str \| None = None}`：返回指定设备或全部场景设备的拓扑元数据（`site_id`、`asset_id`、`asset_type`、`sensors`、`sensor_units`、`normal_ranges` 及当前 `world_state["assets"][asset_id]`）：`{"site_id": ..., "count": k, "assets": [...]}`。未知 `asset_id` 抛 `ValueError`（`tool_error`）。 |
| **IoT** | `query_iot_history` | `query` | `{"asset_id": str, "limit": int=5, "modality": str="all", "metrics": list[str] \| None=None, "start_time": str \| None=None, "aggregation": str="raw"}`：用 `bisect_right` 截取 `<= replay_time`（若给 `start_time` 再过滤 `>= start_time`）；若给非空 `metrics` 则仅投影对应数值/传感器键（不篡改底层缓存）。`aggregation == "raw"` 返回最后 `limit` 条：`{"asset_id": ..., "modality": ..., "aggregation": "raw", "count": len(records), "records": records}`；`aggregation in {"mean", "min", "max", "latest"}` 对窗口内遥测数值按指标聚合（当给定 `start_time` 且未显式传 `limit` 时聚合 `[start_time, replay_time]` 全切片，否则取最后 `limit` 条），返回 `{"asset_id": ..., "modality": ..., "aggregation": aggregation, "sample_count": n, "metrics": {...}}`。 |
| **TSFM** | `check_tsfm_anomaly` | `investigate` | `{"asset_id": str, "metric": str, "window": int=8}`：取截至当前已释放的最近 `window`（`>=1`）条数值遥测，返回 `asset_id, metric, is_anomaly, out_of_range, latest, range, window_count, mean, std, z_score, slope_per_min, consecutive_breaches, pattern`（`"nominal"` \| `"transient_spike"` \| `"sustained_anomaly"`，公式与 v2.0 完全一致）。 |
| **TSFM** | `forecast_tsfm_metric` | `investigate` | `{"asset_id": str, "metric": str, "horizon_minutes": float=30.0, "window": int=8}`：取最近 `window`（`>=2`）条已释放数值遥测做纯标准库 OLS 线性回归预测（精确公式见下），支持阈值越界前的**预测性主动预警**。 |
| **FMSR** | `lookup_fmea_manual` | `investigate` | `{"asset_type": str \| None=None, "keyword": str \| None=None, "fault_type": str \| None=None}`：从 `tool_catalog_data.fmea_manuals` 检索：若传 `asset_type` 按其过滤（若三参数均为空则回退过滤主资产 `asset_type`）；若传 `fault_type` 忽略大小写精确匹配；若传 `keyword` 将其按空白分词为小写词元，要求全部词元命中该条目的可检索文本拼接串（覆盖结构化与原版自由文本字段：`fault_type, failure_mode, symptoms, cause, effects, remediation, description, symptom_metric, recommended_tool`）。返回 `{"asset_type": ..., "keyword": ..., "fault_type": ..., "count": len(manuals), "manuals": manuals}`。 |
| **WorkOrder** | `query_workorder_history` | `query` | `{"asset_id": str \| None=None, "status": str="all", "limit": int=10}`：`status in {"all", "open", "in_progress", "closed"}`。筛选 `tool_catalog_data.historical_workorders` 中满足 `created_at <= replay_time` 的工单，并与当前会话 `world_state["workorders"]` 按 `workorder_id` 合并（会话内已更新版本覆盖历史原记录），按 `asset_id` 与 `status` 过滤后按 `created_at` 升序取最后 `limit` 条：`{"asset_id": ..., "status": status, "count": k, "workorders": [...]}`。 |
| **WorkOrder** | `update_workorder` | `execute` | `{"workorder_id": str, "status": str \| None=None, "note": str="", "priority": str \| None=None}`：在会话工单或已释放历史工单（`created_at <= replay_time`）中查找 `workorder_id`（不存在或原状态为 `"closed"` 则报 `tool_error`），更新 `status`（限 `open`/`in_progress`/`closed`）、`priority`、追加非空 `note` 至 `notes` 列表并设 `updated_at = replay_time`，写入 `world_state["workorders"]` 并同步更新 `world_state["assets"][asset_id]`（`last_workorder_id`, `last_workorder_action: "updated"`, `workorder_status`）。用于**已有未结工单去重追加/提级**场景。 |
| **WorkOrder** | `create_maintenance_workorder` | `execute` | `{"asset_id": str, "fault_type": str, "summary": str}`：生成 `workorder_id = f"WO-{turn_idx}-{step_idx}"`（`priority: "normal"`, `status: "open"`, `created_at: replay_time`），追加至 `world_state["workorders"]` 并更新 `world_state["assets"][asset_id]`（`last_workorder_id`, `last_workorder_action: "created"`, `workorder_status: "open"`）。 |
| **WorkOrder** | `create_urgent_shutdown_workorder` | `execute` | `{"asset_id": str, "fault_type": str, "summary": str}`：校验权限/审批生效后生成紧急停机工单（`priority: "urgent_shutdown"`, `status: "open"`），追加至 `world_state["workorders"]` 并更新 `world_state["assets"][asset_id]`（额外置 `"status": "shutdown_scheduled"`, `"urgent_shutdown": true`）。 |
| **Governance** | `submit_approval_request` | `request_approval` | `{"asset_id": str, "target_tool": str, "reason": str}`：支持 `permissions.approval_policy` 条件审批策略（见下）。 |

#### (3) `forecast_tsfm_metric` 精确 OLS 预测公式
设截至 `replay_time` 最近 `w = max(int(window), 2)` 个数值点为 `[(dt_0, v_0), ..., (dt_{m-1}, v_{m-1})]`，`horizon = float(horizon_minutes)`（`>= 0`），`bounds = normal_ranges.get(metric)`：
- 若 `m == 0`：返回 `latest=None, slope_per_min=0.0, predicted_value=None, residual_std=0.0, prediction_band=[None, None], range=bounds, predicted_breach=False, estimated_minutes_to_breach=None, window_count=0, horizon_minutes=horizon`。
- 若 `m >= 1`：令 `t_i = (dt_i - dt_0).total_seconds() / 60.0`，`t_last = t_{m-1}`，`latest = v_{m-1}`：
  - 当 `m >= 2` 且 `denom = sum((t_i - t_mean)^2) > 1e-12` 时：`slope = sum((t_i - t_mean)*(v_i - v_mean)) / denom`，`intercept = v_mean - slope * t_mean`，残差标准差 `residual_std = sqrt(sum((v_i - (intercept + slope * t_i))^2) / max(m - 2, 1))`；否则 `slope = 0.0, intercept = latest, residual_std = 0.0`。
  - 预测值 `pred = intercept + slope * (t_last + horizon)`，95% 预测带 `prediction_band = [round(pred - 1.96 * residual_std, 4), round(pred + 1.96 * residual_std, 4)]`。
  - 若 `bounds = [low, high]`：`already = (latest < low or latest > high)`，`predicted_breach = already or (pred < low or pred > high)`；`estimated_minutes_to_breach` 在 `already` 时为 `0.0`，未越界且 `slope > 1e-9` 时为 `round(max(0.0, (high - latest) / slope), 4)`，未越界且 `slope < -1e-9` 时为 `round(max(0.0, (low - latest) / (-slope)), 4)`，否则为 `None`。数值字段均保留 4 位小数。

#### (4) 条件审批策略（`approval_policy`）与声明式自定义工具（`custom_tools`）
1. **条件审批 `permissions.approval_policy`**：
   - 可按工具名（或通配键 `"*"`）配置 `{"delay_turns": int (默认 1, >=0), "auto_approve": bool (默认 true), "required_world_state": dict (可选前置状态要求)}`。
   - 调用 `submit_approval_request` 时：若 `target_tool in forbidden_tools`、或 `auto_approve is False`、或 `required_world_state` 未被当前 `world_state` 满足（递归子集匹配），返回 `{"approved": false, "reason": ...}`，由 `ToolExecutor` 记为 `executed: false, authorized: false, status: "unauthorized"`；
   - 通过时计算 `eff = turn_idx + delay_turns`，更新 `approved_from_turn[(asset_id, target_tool)] = min(prev_eff, eff)` 与 `world_state["approvals"][f"{asset_id}:{target_tool}"]`，返回 `{"approved": true, "asset_id": asset_id, "target_tool": target_tool, "delay_turns": delay_turns, "effective_from_turn": eff}`。
2. **声明式 `scenario["custom_tools"]`**：
   - 每项结构：`{"name": str, "decision": "query|investigate|request_approval|execute", "description": str, "required_args": list[str], "terminal": bool (默认 true, 前置准备动作可设 false), "state_effects": dict (可选), "static_output": dict (可选)}`。
   - 执行时统一经过 `allowed_decisions`、`forbidden_tools`、`approval_required_tools` 与 `required_args` 校验；若定义 `state_effects`，将其合并写入 `world_state["assets"][target_asset]`（若某效果值为字符串 `"$args.<key>"`，自动替换为本次调用 `args["<key>"]` 的值），返回 `{"tool": name, "asset_id": target_asset, "applied_effects": applied, **(static_output or {})}`。

### 3.5 可组合隐藏 GT 契约 (`scenarios/gt/<scenario_id>.gt.json`)
100% 兼容 v2.0（`completion_check.mode` 省略时默认为 `"tool_call"`），完整支持 **6 种验证模式**：
1. `"tool_call"`（默认）：在 `>= visible_from` 存在一次授权且成功的 `required_tool` 调用，且 `required_args_subset` 递归子集匹配 `tool_call.args`。
2. `"any_of"`：`{"mode": "any_of", "checks": [<check_1>, <check_2>, ...]}`，任一子项满足即验证通过（例如允许创建新工单或更新已有工单）。
3. `"all_of"`：`{"mode": "all_of", "checks": [<check_1>, <check_2>, ...]}`，非空且全部子项均满足。
4. `"ordered_sequence"`：`{"mode": "ordered_sequence", "steps": [{"tool_name": str, "args_subset": dict}, ...]}`，在 `>= visible_from` 的有效授权成功步骤序列中存在严格先后递增子序列依次匹配各步（例如先 `start_standby_pump` 再 `create_urgent_shutdown_workorder`）。
5. `"world_state"`：`{"mode": "world_state", "expected_state_subset": {...}}`，场景结束时的最终 `world_state` 满足 `expected_state_subset` 递归子集匹配（字典要求键值子集匹配，列表要求每个期望元素至少匹配实际列表中的一个元素），且在 `>= visible_from` 至少发生过一次授权状态变更动作。
6. `"diagnosis_only"`：`{"mode": "diagnosis_only"}`（纯监测/趋势预测/根因诊断场景，严禁乱下发执行动作）：当且仅当（a）在 `>= visible_from` 有效上报匹配该 `need` 的 `reported_needs` 且当步 `evidence_obs_ids` 至少引用 1 条已释放 `supporting_obs_ids`；（b）全回放过程中针对 `need.asset_id` 的 `decision == "execute"` 尝试次数为 `0`（含成功或未授权）。

此外，`gt.json` 顶层支持可选 `"allow_readonly_query_in_no_action_ticks": bool`（默认 `false`；设为 `true` 时，在 `no_action_ticks` 或 `< valid_window.start` 执行只读授权 `decision == "query"` 不计入 `in_no_action_tick` 或 `"early"` 抢跑）。

---

## 4. 零剧透动态 Prompt、Keep-Alive 流式客户端、扩展 `MockClient` 与 A/B 对比器

### 4.1 动态零剧透 Prompt 与全会话前缀不变性 (`framework/prompt.py`)
- **`GLOBAL_FAULT_TAXONOMY`（扩展至 20 个跨设备标准故障码，仅含名称字符串）**：
  覆盖冷水机组、水泵、AHU、冷却塔、压缩机/电机与传感器控制：
  `evaporator_waterside_fouling`, `condenser_tube_scaling`, `refrigerant_undercharge`, `chilled_water_flow_restriction`, `compressor_surge`, `bearing_wear_overheat`, `pump_cavitation`, `impeller_erosion`, `seal_leakage`, `ahu_damper_stuck`, `cooling_coil_valve_leak`, `supply_fan_belt_slip`, `filter_clogging_pressure_drop`, `cooling_tower_fan_imbalance`, `cooling_tower_fill_fouling`, `motor_stator_winding_overheat`, `lubrication_oil_degradation`, `sensor_calibration_drift`, `sensor_frozen_flatline`, `thermostat_setpoint_override`。
- **`messages[0]` 布局**：前半段为全场景共享常量 `GLOBAL_SYSTEM_PREFIX`（声明 10 个内置专家工具签名与 JSON 输出规范）；后半段 `SCENARIO_CONTEXT_JSON` 包含 `scenario_id`、`asset_info`、`assets`、`custom_tools`、`persistent_goals`、`permissions` 与 `fault_taxonomy = sorted(set(GLOBAL_FAULT_TAXONOMY) | set(scenario.get("extra_fault_taxonomy") or []))`。**严禁混入 `tool_catalog_data`（`fmea_manuals` 或 `historical_workorders`）**。
- **步内循环终止条件升级（`framework/run.py`）**：保持 v2.0 全部前缀不变性与 `pending_tool_result` 单次注入规则；内部步循环终止规则升级两处以支持高级治理流程：
  1. 当 `decision == "request_approval"` 时：若本次审批成功且 `effective_from_turn <= turn_idx`（即配置了 `delay_turns == 0` 当轮即刻生效），**不提前终止当前 tick**，允许后续 `step_idx` 同轮完成执行；否则（默认 `delay_turns >= 1`）立即结束当前 tick；
  2. 当 `decision == "execute"` 时：若调用的是声明了 `"terminal": false` 的前置准备类 `custom_tool`，**不提前终止当前 tick**；其余内置/终态执行工具完成后立即结束当前 tick。

### 4.2 线程局部 HTTP Keep-Alive 流式 `VLLMClient` (`framework/client.py`)
- 使用标准库 `http.client.HTTPConnection` / `HTTPSConnection` + `urllib.parse.urlsplit` + `threading.local()` 维护每工作线程持久 TCP/TLS 连接（请求头固定带 `"Connection": "keep-alive"`），消除逐请求建连抖动对 `ttft_ms` 的污染。
- 流式读取至 `data: [DONE]` 后排空响应尾包确保底层 socket 可安全复用；若服务端主动关闭空闲连接导致复用连接在首包前抛出断连异常（`RemoteDisconnected` / `BadStatusLine` / `CannotSendRequest` / `ConnectionError`），自动关闭旧连接、重建新连接并重发 1 次（重置 `t0`）。提供 `close()` 释放连接。

### 4.3 通用零硬编码 `MockClient` 完整 AssetOpsBench 能力扩展 (`framework/client.py`)
`MockClient`（`heuristic` 模式）仍**仅读取 `messages`**（零 `scenario_id` 硬编码、零 GT 访问），在保持内置 `task_01_chiller6` 与 `task_02_pump3` 行为 100% 不变的基础上，通用支持全部新增场景原型：
1. **既有越界 + 预测性越界双通道触发**：
   - 若设备某指标当前最新值已越界（且未受同轮 `planned_load_shift` 维护日志抑制）：按 v2.0 调 `check_tsfm_anomaly` 过滤单点毛刺，确认 `sustained_anomaly`；
   - 若设备当前指标尚在 `normal_ranges` 内，但存在 `>= 3` 个严格单调上升/下降读数且局部线性外推在 `predictive_horizon_minutes`（从资产配置读取，默认 `30.0` 分钟）内越过 `normal_ranges` 边界（或资产显式标记 `enable_predictive_check: true`）：自动调用 `forecast_tsfm_metric`；当返回 `predicted_breach == True` 时以 `predicted_value` 进入后续 FMEA 诊断流！
2. **结构化 + 自由文本 FMEA 兼容匹配**：调用 `lookup_fmea_manual` 后，优先匹配结构化 `symptom_metric` 与 `threshold_high`/`threshold_low`；若手册为自由文本条目（无 `symptom_metric`），按异常/预测指标名与关键词匹配 `symptoms`/`failure_mode`/`description` 并由 `recommended_tool` 或修复描述推断目标工具。
3. **纯诊断模式（`diagnosis_only`）自适应**：当 `permissions.allowed_decisions` 不含 `"execute"` 时，确认故障与 FMEA 后不发起任何执行或申批调用，而是在当步输出 `decision: "wait"`（`tool_call: null`）同时附带 `evidence_obs_ids` 与 `reported_needs` 完成零越权精准上报，后续轮次保持静默 `wait`。
4. **未结工单历史查重与 `update_workorder`**：当 `permissions.open_workorder_check is True` 或该设备已释放观测中存在 `modality == "workorder_history"` 提示时，在创建工单前先调用 `query_workorder_history(asset_id=..., status="open")`；若发现该设备已有 `"open"` 或 `"in_progress"` 工单，调用 `update_workorder` 追加诊断备注并提级，避免重复建单！
5. **条件审批前置动作自动编排**：当目标工具在 `approval_required_tools` 中且 `permissions.approval_policy[rec_tool].required_world_state` 尚未被当前会话已观测/已执行的 `mock_world_state` 满足时，自动从 `SCENARIO_CONTEXT_JSON.custom_tools` 查找 `state_effects` 能满足该前置条件的自定义工具（如 `start_standby_pump`）先行执行，待前置状态就绪后再调用 `submit_approval_request` 与最终高危工具。

### 4.4 CLI 3 `run.py` 批量参数与 CLI 7 `framework/compare.py` 契约

#### (1) `framework/run.py` 扩展
- 支持 `--scenarios <file...>` 和/或 `--scenarios-dir <dir>`（按文件名排序扫描 `*.scenario.json`），以及可选 `--tags <tag1> <tag2>...`（仅运行 `set(tags) & set(scenario.get("tags", []))` 非空的场景）。
- `run_summary.json` 的 `per_scenario[sc_id]` 新增 `"final_world_state": executor.world_state`。

#### (2) CLI 7：`framework/compare.py`（Cache OFF vs Cache ON 对比报告）
命令：
```bash
python3 -m framework.compare \
  --off-summary outputs/off/run_summary.json \
  --on-summary outputs/on/run_summary.json \
  [--off-log outputs/off/run_log.jsonl --on-log outputs/on/run_log.jsonl] \
  [--off-score outputs/off/score_report.json --on-score outputs/on/score_report.json] \
  [--off-replay outputs/off/replay_benchmark_report.json --on-replay outputs/on/replay_benchmark_report.json] \
  --target-ttft-reduction-pct 20.0 \
  --out outputs/comparison_report.json \
  [--out-md outputs/comparison_report.md]
```
输出 `comparison_report.json` 结构：
```json
{
  "schema_version": "3.0",
  "target_ttft_reduction_pct": 20.0,
  "latency_and_throughput": {
    "source": "replay_benchmark | run_summary",
    "off_median_ttft_ms": 142.0,
    "on_median_ttft_ms": 58.0,
    "median_ttft_reduction_pct": 59.155,
    "off_p95_ttft_ms": 210.0,
    "on_p95_ttft_ms": 95.0,
    "p95_ttft_reduction_pct": 54.762,
    "off_median_latency_ms": 480.0,
    "on_median_latency_ms": 390.0,
    "median_latency_reduction_pct": 18.75,
    "off_rps": 2.05,
    "on_rps": 2.54,
    "rps_speedup_ratio": 1.239,
    "off_wall_ms": 6820.0,
    "on_wall_ms": 5510.0,
    "wall_ms_reduction_pct": 19.208
  },
  "cache_efficiency": {
    "theoretical_intra_session_reuse_ratio": 0.784144,
    "off_empirical_cache_hit_rate": 0.0,
    "on_empirical_cache_hit_rate": 0.7612
  },
  "trajectory_determinism": {
    "paired_requests": 14,
    "total_max_requests": 14,
    "exact_request_match_rate": 1.0,
    "action_agreement_rate": 1.0
  },
  "behavioral_parity": {
    "off_micro_f1": 1.0,
    "on_micro_f1": 1.0,
    "micro_f1_delta": 0.0,
    "off_verified_count": 2,
    "on_verified_count": 2,
    "verified_non_regression": true,
    "off_unnecessary_turns": 0,
    "on_unnecessary_turns": 0,
    "off_unauthorized_count": 0,
    "on_unauthorized_count": 0,
    "quality_parity_pass": true
  },
  "hypothesis_evaluation": {
    "ttft_target_met": true,
    "quality_parity_pass": true,
    "overall_hypothesis_pass": true
  }
}
```
> 若同时传入 `--off-replay` 与 `--on-replay`，`latency_and_throughput` 与 `empirical_cache_hit_rate` 优先取受控固定轨迹重放报告（`source: "replay_benchmark"`），否则取 `run_summary.json`（`source: "run_summary"`）。指定 `--out-md` 时同步生成含性能对比表、缓存复用表与质量一致性表的标准 Markdown 报告。

---

## 5. `score.py` 多步/多资产/世界状态重构细则

1. **最终 `world_state` 确定性重构**：`score.py` 仅凭 `scenario.json` 与 `run_log.jsonl` 即可评分——初始化全新的 `ToolExecutor(scenario)`，按排序后的有效日志条目依次重放其中已成功授权执行的 `action`（或读取条目携带的快照），确定性还原每步及最终 `world_state`，供 `completion_check` 的 `"world_state"` 模式及 `duplicate_intervention` 增量完成度检查使用。
2. **多资产归属 `_targets_need` 扩展**：
   - `update_workorder` 若未显式传 `asset_id`，通过 `workorder_id` 反查所属 `asset_id` 精确归属；
   - `query_iot_metadata` / `query_workorder_history` 若 `asset_id is None`，结合 `evidence_obs_ids` 引用的设备集合判定归属。
3. **其余 5 大核心指标输出格式**：`score_report.json` 保持与 v2.0 完全相同的 `per_scenario` 与 `overall` 字段结构，确保任务 5 下游分析脚本零改动兼容。

---

## 6. 内置 2 个标准演示场景与根目录 7 步复跑流程

保持 `raw_data/` 下 2 个精炼快速的默认演示场景（补充 `tags`、`original_assetops_meta`、`sensor_units` 与 `historical_workorders` 已关单历史记录，离线全流程 `< 1s`，峰值 RSS `< 50 MB`）：
- **`task_01_chiller6`**：5 ticks，`chiller_6` + 干扰机 `chiller_7`，宽表 CSV 含待剥离 `anomaly_label` 列、`t1` 计划切负荷 `maintenance_log` 消警、`t2` `chiller_7` 单点毛刺过滤、`t3` 步内 3 步 FMEA 解锁与维护工单创建（8 次请求）。
- **`task_02_pump3`**：4 ticks，`pump_3` + 干扰泵 `pump_4`，大写 `Timestamp` 表头、高危工具 `create_urgent_shutdown_workorder` 跨轮审批生效与执行闭环（6 次请求）。

**仓库根目录完整 7 CLI 标准复跑命令（写入 `README.md`）**：
```bash
# 1. 转换原始数据为隔离的公开场景与隐藏 GT
python3 -m framework.convert --raw-dir raw_data --out-dir scenarios

# 2. 静态校验全部公开场景与 GT 的结构、引用一致性与零剧透/零标签泄露
python3 -m framework.validate --scenarios-dir scenarios/public --gt-dir scenarios/gt

# 3. 批量主动回放（支持 --scenarios-dir 与 --tags；默认 --concurrency 1）
python3 -m framework.run \
  --scenarios-dir scenarios/public \
  --backend mock \
  --cache-mode off \
  --max-steps-per-tick 3 \
  --concurrency 1 \
  --out-dir outputs/demo_off

# 4. 独立读取隐藏 GT 进行可组合确定性离线评分
python3 -m framework.score \
  --scenarios-dir scenarios/public \
  --gt-dir scenarios/gt \
  --log outputs/demo_off/run_log.jsonl \
  --out outputs/demo_off/score_report.json

# 5. 流式导出固定重放请求轨迹与精简执行结果
python3 -m framework.export \
  --log outputs/demo_off/run_log.jsonl \
  --out-requests outputs/demo_off/replay_requests.jsonl \
  --out-actions outputs/demo_off/exported_actions.jsonl

# 6. 跨会话并发固定轨迹重放压测
python3 -m framework.replay \
  --requests outputs/demo_off/replay_requests.jsonl \
  --backend mock \
  --cache-mode off \
  --concurrency 2 \
  --warmup-sessions 1 \
  --repeats 2 \
  --out outputs/demo_off/replay_benchmark_report.json

# 7. 生成 Cache OFF vs Cache ON 性能、缓存命中率与主动性质量对比报告（JSON + Markdown）
python3 -m framework.compare \
  --off-summary outputs/demo_off/run_summary.json \
  --on-summary outputs/demo_off/run_summary.json \
  --off-log outputs/demo_off/run_log.jsonl \
  --on-log outputs/demo_off/run_log.jsonl \
  --off-score outputs/demo_off/score_report.json \
  --on-score outputs/demo_off/score_report.json \
  --off-replay outputs/demo_off/replay_benchmark_report.json \
  --on-replay outputs/demo_off/replay_benchmark_report.json \
  --out outputs/demo_off/comparison_report.json \
  --out-md outputs/demo_off/comparison_report.md
```

---

## 7. 工程师交付验收清单（v3.0 Acceptance Criteria）

1. **纯标准库、7 个 CLI 全绿与 `< 50 MB` 低内存**：100% 仅用 Python 3.10+ 标准库；全部 7 个 CLI（`convert`, `validate`, `run`, `score`, `export`, `replay`, `compare`）与 `python3 -m unittest discover -s tests -v` 100% 通过；默认 `--concurrency 1` 下各 CLI 峰值内存均 `< 50 MB` RSS。
2. **v2.0 100% 向后兼容 + 零泄露校验**：原有双演示任务保持 `micro_need_discovery_f1 = 1.0`、`total_unnecessary_turns = 0`、`timely_count = 2`、`verified_count = 2`、`unauthorized_count = 0`；`validate.py` 对合法场景退出 `0`，对注入标签泄露、坏 `obs_id`、非递增 `replay_ticks` 或未知工具名的故障场景准确报错并退出非 `0`。
3. **AssetOpsBench 全专家域与高级治理单测全覆盖（`tests/test_pipeline.py`）**：
   - **Case 1（IoT 元数据与窗口聚合）**：验证 `query_iot_metadata` 与 `query_iot_history` 的 `start_time`、`metrics` 投影及 `mean`/`max` 聚合；
   - **Case 2（TSFM 趋势预测 + `diagnosis_only` 评分）**：传感器尚未越过静态上限但呈显著线性上升趋势，`MockClient` 调用 `forecast_tsfm_metric` 提前检出 `predicted_breach == True`、检索 FMEA 并以 `diagnosis_only` 模式完成零 `execute` 验证；
   - **Case 3（FMSR 原版自由文本手册关键词检索）**：对无结构化 `symptom_metric` 的自由文本 FMEA 条目通过 `lookup_fmea_manual(keyword=...)` 与 `MockClient` 文本回退完成匹配；
   - **Case 4（WorkOrder 查重去重更新 + `any_of` / `world_state` 评分）**：历史目录含已释放 `"open"` 工单，`MockClient` 先调 `query_workorder_history` 发现未结工单后改调 `update_workorder` 提级追加备注，通过 `any_of` 与 `world_state` 双模式验证；
   - **Case 5（声明式 `custom_tools` + 条件审批 `required_world_state` + `ordered_sequence`）**：备泵初始处于 `"standby"`，紧急停机审批要求备泵先 `"running"`；验证直接申批被拒（`unauthorized`），而 `MockClient` 自动先执行 `start_standby_pump`（含测试 `delay_turns == 0` 同轮闭环与 `delay_turns == 1` 跨轮闭环），并通过 `ordered_sequence` + `world_state` 验证；
   - **Case 6（窄表 `long` CSV + 自动 `tick_grid` + 标签黑名单剥离）**：端到端验证窄表传感器数据转换、`fmea_manuals.json`/`workorders.json` 外部文件加载、`tick_grid` 自动网格与 `allow_readonly_query_in_no_action_ticks`；
   - **Case 7（Keep-Alive 连接复用 + `compare.py` A/B 报告）**：通过本地 HTTP Server 计数服务端 accept 的 TCP 连接数，证明同一 `VLLMClient` 多次流式 SSE 请求真正复用单条 Keep-Alive 连接（含服务端断连自动重连测试），并验证 `compare.py` 生成准确的 JSON 与 Markdown A/B 对比报告。
