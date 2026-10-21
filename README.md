# Employee Report Analyzer / 员工汇报分析工具

[English](README_EN.md) | 简体中文

一个面向 GitHub Copilot CLI 的开源 Skill 与本地 Python 工具链：从 MP3、WAV 或 JSON 转写中提取逐人证据，生成独立 Markdown/DOCX 报告，并指导 Copilot 撰写可追溯、需人工审核的主管评价草稿。

> **重要边界：** 本项目生成证据包和 AI 评价草稿，不替代主管判断，不应自动作出转正、晋升、绩效或用工决定。

## 核心能力

- **真实音频入口**：Faster-Whisper 本地 ASR，支持 Windows CPU INT8、VAD、`base`/`small` 模型。
- **说话人候选时间轴**：无时间范围时，先进行匿名 diarization，再由人工确认姓名与范围。
- **稳健分块**：30–60 秒分块、独立 worker、硬超时、失败关闭及可疑块模型升级。
- **幻觉检测**：识别重复文本、常见幻觉短语、异常语速、低置信度等风险。
- **独立报告**：每位说话人分别输出 Markdown 和 DOCX，同时生成摘要。
- **证据化评价**：按时间戳区分事实、评价解释、差距、待核验项及行动建议。
- **隐私优先**：默认本地处理；JSON 工作流无需云端令牌。

## 处理流程

```text
音频/JSON → 匿名时间轴 → 人工确认范围和身份 → 分块转写
          → 幻觉检查 → 每人证据包 → Copilot主管评价草稿 → 人工审批
```

## 快速开始

### 1. 安装

需要 Python 3.9+。

```powershell
# 基础功能：JSON输入、报告生成和测试
pip install -e .

# 推荐：真实音频转写 + 轻量本地diarization
pip install -e ".[audio]"

# 开发测试
pip install -e ".[dev]"
```

可选的高质量说话人区分：

```powershell
pip install -e ".[diarize-speechbrain]"
# 或使用需要 Hugging Face 授权的 pyannote
pip install -e ".[diarize-pyannote]"
```

### 2. 查看匿名候选时间轴

```powershell
python -m zhrreport.cli diarize meeting.mp3 `
  --range 00:00:00-00:02:00 `
  --diarization-backend numpy_acoustic `
  --num-speakers 2
```

`SPEAKER_00` 等标签不是已验证身份，必须人工映射到姓名。

### 3. 确认范围后生成独立报告

```powershell
python -m zhrreport.cli report meeting.mp3 `
  --backend faster-whisper `
  --range 00:05:26-00:19:01 `
  --speaker-name KY `
  --model base `
  --chunk-duration 45 `
  --timeout 120 `
  --output-dir output\KY
```

输出包括 `KY.md`、`KY.docx`、`_summary.md` 和 `_summary.docx`。

### 4. 使用已有 JSON 转写

```powershell
python -m zhrreport.cli report sample\sample_transcript.json `
  --confirm-full --output-dir output\sample
```

### 5. 让 Copilot 合成主管评价草稿

将仓库作为 Copilot CLI 工作区打开后，`.github/skills/zhrreport/SKILL.md` 会提供评价规则。可提出：

```text
读取 output 中每个人的证据报告，按照 zhrreport Skill 生成独立的主管评价草稿。
```

评价结构包括总体评价、阶段标准、四个证据维度、技术案例、职业化素养、30–60 天行动及证据质量附录。

## Diarization 后端

| 后端 | 凭据 | 特点 | 限制 |
|---|---|---|---|
| `numpy_acoustic` | 无 | 轻量、可本地执行 | 声学聚类，不是说话人身份模型 |
| `speechbrain` | 无 | ECAPA speaker embeddings | 依赖和模型较大 |
| `pyannote` | HF token/模型权限 | 通常质量最高 | 需要授权，环境较重 |

## 常用诊断

```powershell
python -m zhrreport.cli backends
python -m zhrreport.cli diarizers
python -m zhrreport.cli timeline sample\sample_transcript.json
python -m zhrreport.cli check-hallucination sample\sample_transcript.json
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

真实音频集成测试会在依赖或测试音频缺失时明确跳过。

## Copilot 集成

- Skill：`.github/skills/zhrreport/SKILL.md`
- Extension：`.github/extensions/zhrreport/extension.mjs`
- CLI：`python -m zhrreport.cli`

扩展提供 `zhrreport_generate`、`zhrreport_timeline`、`zhrreport_check_hallucination` 和 `zhrreport_backends`。具体可用性取决于 Copilot CLI 版本及扩展运行环境；CLI 始终可独立使用。

## 安全、隐私与公平性

- 不要提交真实音频、转写、员工报告或客户信息到公开仓库。
- 自动转写可能错误识别人名、客户名、数字、命令和根因；必须回源核验。
- 匿名聚类标签不代表真实身份。
- AI 输出只能作为草稿，正式人事决策必须由有权限的人员依据完整信息作出。
- 详见 [SECURITY.md](SECURITY.md)。

## 许可证

[MIT](LICENSE)
