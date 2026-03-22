# -*- coding: utf-8 -*-
import pathlib

SEC4 = '''## 四、异步并发模型详解

### 4.1 为何 ASR 需要 2~5 秒却不阻塞接收新音频？

```
asynio 事件循环（单线程）时间轴

t=0ms    收到音频包#1 → VAD → 检测到语音段A结束
         create_task(处理段A)  ← 登记任务，耗时 < 0.1ms，立即返回
t=1ms    await receive_text()  ← 协程挂起，等待下一包

t=256ms  收到音频包#2 → VAD → 段B 尚未结束
t=512ms  收到音频包#3 → VAD → 检测到段B结束
         create_task(处理段B)  ← 立即返回
t=513ms  继续等待下一个音频包

         ┌────────────────────────────────────────────┐
         │  线程池并行（不占用事件循环）               │
         │  段A: t=1ms   → ASR(2s)‖声纹(0.5s)        │
         │       → t=2001ms 完成                      │
         │  段B: t=513ms → ASR(2s)‖声纹(0.5s)        │
         │       → t=2513ms 完成                      │
         └────────────────────────────────────────────┘

t=2001ms 段A完成 → 事件循环调度 → ws.send_json(transcript_A)
t=2513ms 段B完成 → 事件循环调度 → ws.send_json(transcript_B)

✅ 结论：t=1ms ~ t=2001ms 期间，事件循环持续接收新音频包
         ASR 在线程池运行，完全不占用事件循环线程
```

### 4.2 Semaphore 防止 GPU 显存溢出

```
设 ASR_MAX_CONCURRENT = 2（默认）

同时有 5 个语音段等待转写：

段A → acquire(count=1) → 进线程池 → GPU 推理中...
段B → acquire(count=2) → 进线程池 → GPU 推理中...
段C → acquire → ⏸ 等待（已达上限 2）
段D → acquire → ⏸ 等待
段E → acquire → ⏸ 等待

段A 完成 → release → 段C 进入线程池
段B 完成 → release → 段D 进入线程池

✅ 效果：GPU 同时只跑 2 个 ASR 推理，防止显存 OOM
        等待中的段 C/D/E 以协程挂起形式等待，不阻塞事件循环

⚙️ 调整建议：
   GPU 显存 >= 16GB → ASR_MAX_CONCURRENT=3
   纯 CPU 推理     → ASR_MAX_CONCURRENT=4
```

### 4.3 fire-and-forget 模式（实时性优先）

```
_process_segment 内部执行顺序：

  1. await ASR + 声纹（并行，必须等）
  2. 聚类匹配 speaker（内存操作，< 1ms）
  3. ✅ await ws.send_json(transcript)   ← 先推前端，不等存储
  4. create_task(_write_segment_to_db)   ← PostgreSQL 异步写
  5. create_task(milvus.insert_segment)  ← Milvus 异步写

优先级：实时体验 > 存储延迟
容错性：PG/Milvus 写入失败不影响客户端接收结果
```

### 4.4 并发模型总览

| 组件 | 执行方式 | 是否阻塞事件循环 | 并发控制 |
|------|---------|----------------|----------|
| WebSocket 接收 | asyncio 原生协程 | 否 | 无限制 |
| VAD 处理 | 同步（< 1ms/帧）| 否（极短）| 无需限制 |
| ASR 转写 | asyncio.to_thread | 否 | Semaphore(2) |
| 声纹提取 | asyncio.to_thread | 否 | 同上（gather）|
| 声纹聚类 | 同步（< 1ms）| 否（极短）| 无需限制 |
| PostgreSQL 写 | asyncio + asyncpg | 否 | 连接池 |
| Milvus 写 | asyncio.to_thread | 否 | fire-and-forget |
| MinIO 上传 | asyncio.to_thread | 否 | 会话结束触发 |

---
'''

pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/scripts/_arch_sec4.txt').write_text(SEC4, encoding='utf-8')
print('sec4 len:', len(SEC4))
print('done')
