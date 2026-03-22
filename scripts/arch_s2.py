# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u4e8c\u3001\u6574\u4f53\u67b6\u6784\u56fe

```
+----------------------------------------------------------------+
|                   \u5ba2\u6237\u7aef\uff08\u6d4f\u89c8\u5668\uff09                            |
|  \u9ea6\u514b\u98ce --(16kHz PCM)--> base64 --[WebSocket]--> \u540e\u7aef          |
|  <--[WebSocket]-- {speaker,text,start,end} \u5b9e\u65f6\u8bc6\u522b\u7ed3\u679c     |
|  REST: PUT /speakers  GET /download  GET /transcript          |
+--------------------------+---------------------------------+---+
                           | WebSocket + HTTP
+--------------------------v---------------------------------+---+
|              FastAPI \u540e\u7aef\uff08\u5355\u8fdb\u7a0b asyncio\uff09                      |
|                                                                |
|  +-----------------+     +-------------------------------+    |
|  | WebSocket\u5904\u7406\u5668  |     |     REST API (sessions.py) |    |
|  | ws_handler.py   |     | \u7f16\u8f91\u8bf4\u8bdd\u4eba/\u4e0b\u8f7d/\u67e5\u8be2/\u5217\u8868  |    |
|  +-----------------+     +-------------------------------+    |
|          |                                                     |
|          v                                                     |
|  +----------------------------------------------------+        |
|  |               Services \u5c42                        |        |
|  |  VADService    ASRService     SpeakerService        |        |
|  |  Silero VAD    FunASR(\u9ed8\u8ba4)   ERes2Net\u58f0\u7eb9\u63d0\u53d6  |        |
|  |  \u6bcf\u4f1a\u8bdd\u72ec\u7acb   /Whisper(\u53ef\u9009) +\u5728\u7ebf\u805a\u7c7b      |        |
|  |               Semaphore\u9650\u6d41  \u4f59\u5f26\u76f8\u4f3c\u5ea6\u5339\u914d   |        |
|  |  StorageService       MilvusService(\u53ef\u9009)      |        |
|  |  MinIO \u97f3\u9891+\u6587\u672c        \u8bed\u4e49\u5411\u91cf+\u68c0\u7d22         |        |
|  +----------------------------------------------------+        |
|          |                                                     |
|  +----------------------------------------------------+        |
|  |  SessionManager \u5168\u5c40\u5355\u4f8b                          |        |
|  |  Dict[session_id -> SpeechSession]                 |        |
|  |  \u6bcf\u4f1a\u8bdd: VAD\u5b9e\u4f8b+\u805a\u7c7b+\u97f3\u9891\u7f13\u51b2+\u7247\u6bb5\u5217\u8868          |        |
|  +----------------------------------------------------+        |
+----------+-----------------+-----------------+----------------+
           |                 |                 |
  +--------+------+  +-------+-------+  +------+--------+
  | PostgreSQL    |  |     MinIO     |  |    Milvus     |
  | sessions      |  | raw_audio.wav |  | 384\u7ef4\u5411\u91cf      |
  | speakers      |  | transcript    |  | HNSW\u7d22\u5f15     |
  | segments      |  | \u9884\u7b7e\u540dURL\u4e0b\u8f7d  |  | (\u53ef\u9009)       |
  +---------------+  +---------------+  +---------------+
```
'''
f.write_text(t, encoding='utf-8')
print('s2', len(t))
