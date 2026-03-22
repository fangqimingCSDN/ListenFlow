# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u516b\u3001\u9879\u76ee\u7ed3\u6784\u8bf4\u660e

### 8.1 \u76ee\u5f55\u7ed3\u6784

```
speech_proj/
  backend/
    main.py              FastAPI\u5165\u53e3\uff0clifespan\u9884\u52a0\u8f7d\u6240\u6709\u6a21\u578b
    core/
      config.py          pydantic-settings\u7edf\u4e00\u914d\u7f6e\uff0c\u6240\u6709\u53c2\u6570\u4ece.env\u8bfb\u53d6
      logging.py         loguru\uff0c\u6587\u4ef6+\u63a7\u5236\u53f0\u53cc\u8f93\u51fa
    db/
      database.py        asyncpg\u5f02\u6b65\u5f15\u64ce\uff0cAsyncSessionLocal
      models.py          3\u5f20\u8868\uff0c\u65e0ForeignKey\u65e0relationship
    services/
      vad_service.py     Silero VAD\uff0c\u6bcf\u4f1a\u8bdd\u72ec\u7acb\u5b9e\u4f8b
      asr_service.py     FunASR/Whisper\u53cc\u540e\u7aef\uff0cSemaphore\u5e76\u53d1\u9650\u6d41
      speaker_service.py ERes2Net\u58f0\u7eb9+OnlineSpeakerCluster
      storage_service.py MinIO\u4e0a\u4f20/\u4e0b\u8f7d/\u9884\u7b7e\u540dURL
      session_manager.py SpeechSession+SessionManager\u5168\u5c40\u5355\u4f8b
      vector_service.py  MilvusService+TextEmbedder(\u53ef\u9009)
    api/
      ws_handler.py      WebSocket\u4e3b\u5904\u7406\u5668
      sessions.py        REST API\u8def\u7531
    migrations/          Alembic\u8fc1\u79fb\u811a\u672c
  frontend/
    index.html           \u5355\u9875\u6f14\u793a UI
    app.js               WebSocket\u5f55\u97f3+\u6ce2\u5f62\u53ef\u89c6\u5316+\u8bf4\u8bdd\u4eba\u7f16\u8f91
  docker/
    docker-compose.yml   PG+MinIO+Milvus+Backend
    Dockerfile.backend   CUDA 11.8+Python 3.11
  scripts/               \u5de5\u5177\u811a\u672c
  ARCHITECTURE.md        \u672c\u6587\u6863
  README.md
```

### 8.2 Services \u5c42\u8bbe\u8ba1\u539f\u5219

**\u5355\u4e00\u804c\u8d23**: \u6bcf\u4e2aService\u53ea\u505a\u4e00\u4ef6\u4e8b

| Service | \u804c\u8d23 |
|---------|------|
| VADService | \u53ea\u505a\u8bed\u97f3\u7aef\u70b9\u68c0\u6d4b |
| ASRService | \u53ea\u505a\u6587\u672c\u8f6c\u5199 |
| SpeakerService | \u53ea\u505a\u58f0\u7eb9\u63d0\u53d6\u548c\u805a\u7c7b |
| StorageService | \u53ea\u505a\u6587\u4ef6\u5b58\u53d6 |
| MilvusService | \u53ea\u505a\u5411\u91cf\u5b58\u53d6 |

**\u5168\u5c40\u5355\u4f8b**: \u6a21\u578b\u52a0\u8f7d\u8017\u65f6(5~30\u79d2)\uff0cFastAPI lifespan\u9884\u52a0\u8f7d\u4e00\u6b21\uff0c\u6240\u6709\u8bf7\u6c42\u590d\u7528

**\u7ebf\u7a0b\u5b89\u5168**: \u6a21\u578b\u63a8\u7406\u5747\u4e3a\u540c\u6b65\u963b\u585e\uff0c\u901a\u8fc7 asyncio.to_thread() \u6d3e\u53d1\u7ebf\u7a0b\u6c60\uff0c\u4e0d\u963b\u585e\u4e8b\u4ef6\u5faa\u73af

### 8.3 SessionManager \u8bbe\u8ba1

```
SessionManager (\u5168\u5c40\u5355\u4f8b)
  sessions: Dict[session_id, SpeechSession]
  \u5b9a\u671f\u6e05\u7406\u8d85\u65f6\u7a7a\u95f2\u4f1a\u8bdd (\u9ed8\u8ba4600s)

SpeechSession (\u6bcf\u4e2aWebSocket\u8fde\u63a5\u4e00\u4e2a)
  vad          \u72ec\u7acbVAD\u72b6\u6001\u673a\uff0c\u4e0d\u540c\u4f1a\u8bdd\u4e92\u4e0d\u5e72\u6270
  cluster      \u72ec\u7acb\u8bf4\u8bdd\u4eba\u805a\u7c7b\uff0c\u4e0d\u540c\u4f1a\u8bdd\u4e0d\u6df7\u6dc6
  audio_buffer \u539f\u59cbPCM\u7d2f\u79ef\uff0c\u4f1a\u8bdd\u7ed3\u675f\u540e\u6253\u5305WAV\u4e0a\u4f20
  segments     \u5df2\u5b8c\u6210\u8f6c\u5199\u7247\u6bb5\u5217\u8868
  speaker_names  speaker_label -> display_name \u6620\u5c04
  db_session_id  \u5bf9\u5e94PG\u4e2d\u7684 session.id
```
'''
f.write_text(t, encoding='utf-8')
print('s8', len(t))
