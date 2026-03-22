# -*- coding: utf-8 -*-
import pathlib

OUT = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')

PART1 = """\
# 实时语音转写系统 — 架构设计文档 v2.0

**版本**: v2.0 | **日期**: 2026-03-21

---

## 目录

1. [系统定位与核心目标](#一系统定位与核心目标)
2. [整体架构图](#二整体架构图)
3. [核心数据流](#三核心数据流一句话如何被识别)
4. [异步并发模型](#四异步并发模型详解)
5. [PostgreSQL 表设计](#五postgresql-表设计)
6. [存储层设计](#六存储层设计)
7. [模型选型决策](#七模型选型决策)
8. [项目结构说明](#八项目结构说明)
9. [关键配置速查](#九关键配置速查)

---

## 一、系统定位与核心目标

多人实时语音转写系统，核心能力：

| 能力 | 实现方式 | 指标 |
|------|----------|---------|
| 实时转写 | WebSocket 流式推送 | 停顿后 1~3s 内出字 |
| 说话人分离 | ERes2Net + 在线聚类 | 自动标注 speaker_0/1/2 |
| 姓名绑定 | REST API 编辑 | 历史片段同步更新 |
| 多路并发 | SessionManager 隔离 | 每会话独立 VAD+聚类实例 |
| 离线存档 | MinIO 对象存储 | 音频+文本永久保存 |
| 语义检索 | Milvus 向量库(可选) | 跨会话语义搜索 |
"""

PART2 = """
---

## 二、整体架构图

```
+---------------------------------------------------------------+
|                    客户端（浏览器）                             |
|                                                               |
|  麦克风 --[16kHz PCM]--> base64编码 --[WebSocket]--> 后端    |
|  <--[WebSocket]-- {speaker,text,start,end}  实时transcript  |
|                                                               |
|  REST API:                                                    |
|    PUT /api/sessions/{id}/speakers   编辑说话人姓名         |
|    GET /api/sessions/{id}/download   获取预签名下载链接  |
|    GET /api/sessions/{id}/transcript 查询完整转写         |
+----------------------------+----------------------------------+
                             | WebSocket + HTTP
+----------------------------v----------------------------------+
|              FastAPI 后端（单进程 asyncio）                     |
|                                                               |
|  WebSocket Handler          REST API (sessions.py)           |
|  ws_handler.py              编辑说话人/下载/查询           |
|       |                                                       |
|       v                                                       |
|  +-------------------------------------------------------+   |
|  |                    Services 层                        |   |
|  |                                                       |   |
|  |  VADService    ASRService       SpeakerService        |   |
|  |  Silero VAD    FunASR(默认)     ERes2Net声纹提取    |   |
|  |  每会话独立  /Whisper(可选)  +OnlineSpeakerCluster  |   |
|  |               Semaphore限流    余弦相似度聚类        |   |
|  |                                                       |   |
|  |  StorageService           MilvusService(可选)       |   |
|  |  MinIO 音频+文本            语义向量存储+检索       |   |
|  +-------------------------------------------------------+   |
|       |                                                       |
|  +-------------------------------------------------------+   |
|  |  SessionManager（全局单例）                             |   |
|  |  Dict[session_id -> SpeechSession]                    |   |
|  |  每个会话独立: VAD实例+聚类实例+音频缓冲+片段列表  |   |
|  +-------------------------------------------------------+   |
+----------+------------------+--------------------+-----------+
           |                  |                    |
    +------+------+  +--------+-------+  +---------+------+
    | PostgreSQL  |  |     MinIO      |  |    Milvus      |
    | sessions表  |  | raw_audio.wav  |  | 384维向量       |
    | speakers表  |  | transcript.txt |  | HNSW索引        |
    | segments表  |  | 预签名直接下载  |  | (可选)          |
    +-------------+  +----------------+  +----------------+
```
"""

PART3 = """
---

## 三、核心数据流：一句话如何被识别

### 3.1 音频采集到文字推送完整链路

```
STEP 1  用户开口说话
           |
           v
        [浏览器 AudioContext - ScriptProcessor]
          每 256ms 采集 4096 个样本点 (16kHz)
          Float32 -> Int16 -> base64 -> WebSocket发送
           |
STEP 2  [WebSocket Handler - ws_handler.py]
          base64解码 -> PCM bytes
           |
STEP 3  [VADService.process_chunk(bytes)]  每会话独立实例
          |
          +-- 未检测到语音结束 --> 返回空列表 --> 