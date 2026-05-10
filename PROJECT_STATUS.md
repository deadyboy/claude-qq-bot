# QQ 机器人智能体改造项目状态

**更新日期**: 2026-05-02  
**项目**: claude-qq-bot  
**目标**: 从简单对话机器人升级为具备记忆/工具/规划能力的智能体

---

## 一、项目背景

### 原始状态
- 基于 nonebot2 + OneBot v11 的 QQ 机器人
- 接入中科大 LLM API（Qwen/DeepSeek 等）
- 仅有简单的 20 条消息会话记忆
- 被动应答，无任务执行能力

### 改造目标
1. 实现跨对话的永久记忆
2. 支持工具调用（文件/网络/代码）
3. 实现"用户监督，机器人干活"的工作模式

---

## 二、已完成工作

### 2.1 核心架构文件

| 文件 | 路径 | 状态 | 说明 |
|------|------|------|------|
| `memory_core.py` | `src/plugins/claude/` | ✅ 完成 | 混合记忆系统核心 |
| `agent.py` | `src/plugins/claude/` | ✅ 完成 | 智能体引擎框架 |
| `dialogue.py` | `src/plugins/claude/` | ✅ 升级 | 支持双模式切换 |
| `AGENT_MODE.md` | 项目根目录 | ✅ 完成 | 智能体模式使用说明 |
| `IMPLEMENTATION_SUMMARY.md` | 项目根目录 | ✅ 完成 | 技术实现总结 |

### 2.2 记忆系统

#### 短期记忆 (`ShortTermMemoryManager`)
- 存储：`data/sessions/*.json`
- 功能：
  - ✅ 会话消息增删改查
  - ✅ 重要消息标记
  - ✅ 动态长度控制（默认 50 条）
  - ✅ 超时自动失效（默认 2 小时）

#### 长期记忆 (`LongTermMemoryManager`)
- 存储：`data/longterm_memory/chroma.sqlite3`
- 功能：
  - ✅ ChromaDB 向量存储
  - ✅ 语义检索
  - ✅ 记忆类型分类（conversation/fact/skill/experience）
  - ⚠️ 降级模式：关键词搜索（向量模型下载慢）

#### 关键事实 (`KeyFactManager`)
- 存储：`data/key_facts.db` (SQLite)
- 功能：
  - ✅ 用户画像存储
  - ✅ 任务 CRUD
  - ✅ 事实验证机制
  - ✅ 按类型/主题查询

#### 统一管理 (`UnifiedMemoryManager`)
- ✅ 整合三层记忆 API
- ✅ 对话自动提取框架
- ✅ 用户画像查询
- ✅ 任务管理接口

### 2.3 智能体引擎

#### 意图识别 (`IntentRecognizer`)
- ✅ 意图分类：chat/task/query/command
- ✅ 关键词匹配（降级方案）
- ✅ LLM 辅助识别（框架）

#### 工具集 (`ToolRegistry`)
- ✅ 文件系统工具：
  - `read_file` - 读取文件
  - `write_file` - 写入文件
  - `list_files` - 列出目录
  - `search_files` - 搜索内容
- ✅ 代码执行工具：
  - `run_python` - 执行 Python 代码
  - `run_shell` - 执行 Shell 命令
- ⚠️ 网络工具（空壳）：
  - `web_search` - 需集成 WebSearch API
  - `web_fetch` - 需集成 WebFetch API

#### 任务规划 (`TaskPlanner`)
- ✅ 任务分解框架
- ✅ 依赖管理数据结构
- ❌ 执行引擎（未实现）

#### 监督日志 (`SupervisionLogger`)
- ✅ 日志级别：info/decision/action/warning/error
- ✅ 控制台实时输出
- ✅ JSONL 文件持久化
- ✅ 按级别/模块筛选

### 2.4 对话处理

| 功能 | 简单模式 | 智能体模式 |
|------|----------|------------|
| 基础对话 | ✅ | ✅ |
| 多轮记忆 | ✅ | ✅ |
| 意图识别 | ❌ | ✅ |
| 工具调用 | ❌ | ✅ |
| 任务管理 | ❌ | ✅ |
| 监督日志 | ❌ | ✅ |

### 2.5 测试脚本

| 文件 | 说明 | 状态 |
|------|------|------|
| `test_quick.py` | 快速测试记忆系统 | ✅ 通过 |
| `test_memory.py` | 完整测试（含 ChromaDB） | ⚠️ 网络问题 |

---

## 三、文件存放位置

### 项目根目录
```
F:\ClaudeSpace2\claude-qq-bot\
```

### 核心代码
```
F:\ClaudeSpace2\claude-qq-bot\src\plugins\claude\
├── memory_core.py      # 混合记忆系统
├── agent.py            # 智能体引擎
├── dialogue.py         # 对话处理（双模式）
├── api.py              # LLM API 调用（原有）
├── memory.py           # 简单会话管理（原有，保留）
├── formatter.py        # 消息格式化（原有）
├── config.py           # 模型切换配置（原有）
└── __init__.py         # 插件入口
```

### 文档
```
F:\ClaudeSpace2\claude-qq-bot\
├── PROJECT_STATUS.md       # 本文档（项目状态）
├── AGENT_MODE.md           # 智能体模式使用指南
├── IMPLEMENTATION_SUMMARY.md # 技术实现总结
├── README.md               # 原项目说明
├── DEPLOY.md               # 部署指南
├── INSTALL.md              # 安装说明
└── QUICKSTART.md           # 快速开始
```

### 数据目录
```
F:\ClaudeSpace2\claude-qq-bot\data\
├── sessions/           # 短期记忆（JSON 文件）
│   ├── private_xxx.json
│   └── group_xxx.json
├── longterm_memory/  # 长期记忆（ChromaDB）
│   └── chroma.sqlite3
├── key_facts.db      # 关键事实（SQLite）
└── logs/             # 监督日志
    └── supervision_YYYY-MM-DD.jsonl
```

### 测试文件
```
F:\ClaudeSpace2\claude-qq-bot\
├── test_quick.py     # 快速测试（推荐）
└── test_memory.py    # 完整测试
```

---

## 四、测试结果

### test_quick.py 运行结果
```
==================================================
QQ 机器人 - 记忆系统快速测试
==================================================
[1/3] 测试短期记忆...
      通过 - 共 3 条消息
[2/3] 测试关键事实...
      通过
[3/3] 测试统一记忆...
      通过

==================================================
结果汇总:
  短期记忆：[PASS]
  关键事实：[PASS]
  统一记忆：[PASS]
==================================================
所有测试通过！[OK]
```

---

## 五、待完成事项

### 高优先级
- [ ] **工具执行引擎**：实现任务调度循环
- [ ] **网络工具集成**：接入实际的 WebSearch/WebFetch
- [ ] **错误恢复机制**：工具失败重试/回退

### 中优先级
- [ ] **意图识别优化**：添加 Few-Shot 示例
- [ ] **记忆自动提取**：实现 `_auto_extract` 调用 LLM 分析
- [ ] **用户确认机制**：审批流程实现

### 低优先级
- [ ] **定时任务调度器**：周期性任务支持
- [ ] **多模态支持**：图片理解
- [ ] **性能优化**：缓存/批量操作

---

## 六、启用智能体模式

### 当前状态
- 默认：`AGENT_MODE = False`（简单模式）

### 启用状态
旧 `AGENT_MODE` 已标记为 legacy，不建议再通过改开关启用；后续应以受控 Agent Mode 重构替代。

### 验证
- 发送 `/status` 查看智能体状态
- 发送 `/tasks` 查看任务列表

---

## 七、依赖变更

### 新增依赖
```toml
# pyproject.toml
chromadb>=0.4.0  # 向量数据库
```

### 安装
```bash
pip install -e .
```

---

## 八、关键设计决策

### 1. 混合记忆架构
- **为什么**：单一存储无法满足所有场景
- **方案**：JSON（轻量）+ ChromaDB（语义）+ SQLite（结构化）

### 2. 双模式设计
- **为什么**：保证向后兼容，降低使用门槛
- **方案**：`AGENT_MODE` 开关切换

### 3. 监督日志
- **为什么**：用户需要透明化决策过程
- **方案**：JSONL 格式，按级别分类

---

## 九、联系方式

项目位置：`F:\ClaudeSpace2\claude-qq-bot\`

如有疑问，请查阅：
- 使用指南：`AGENT_MODE.md`
- 技术文档：`IMPLEMENTATION_SUMMARY.md`
