# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u4e5d\u3001\u5173\u952e\u914d\u7f6e\u901f\u67e5

| \u73af\u5883\u53d8\u91cf | \u9ed8\u8ba4\u5024 | \u8bf4\u660e |
|----------|--------|------|
| ASR_BACKEND | funasr | funasr \u6216 whisper |
| ASR_MAX_CONCURRENT | 2 | GPU\u5e76\u53d1\u63a8\u7406\u4e0a\u9650 |
| VAD_SILENCE_DURATION_MS | 800 | \u9759\u97f3\u65ad\u53e5\u9608\u5024(ms) |
| VAD_MAX_SEGMENT_DURATION | 30 | \u6700\u5927\u7247\u6bb5\u65f6\u957f(s) |
| VAD_THRESHOLD | 0.5 | VAD\u8bed\u97f3\u6982\u7387\u9608\u5024 |
| ENABLE_VECTOR_STORE | false | \u662f\u5426\u542f\u7528Milvus |
| VECTOR_BACKEND | milvus | milvus \u6216 qdrant |
| MILVUS_HOST | localhost | Milvus\u5730\u5740 |
| WHISPER_MODEL_SIZE | large-v3 | Whisper\u6a21\u578b\u5927\u5c0f |
| WHISPER_COMPUTE_TYPE | float16 | GPU:float16 / CPU:int8 |
| DATABASE_URL | postgresql+asyncpg://... | PG\u8fde\u63a5\u4e32 |
| MINIO_ENDPOINT | localhost:9000 | MinIO\u5730\u5740 |

---

## \u5341\u3001\u5feb\u901f\u542f\u52a8\u6307\u5357

```bash
# 1. \u542f\u52a8\u57fa\u7840\u670d\u52a1
cd docker
docker-compose up -d postgres minio

# 2. \u521d\u59cb\u5316\u6570\u636e\u5e93
cd backend
alembic upgrade head

# 3. \u590d\u5236\u914d\u7f6e\u6587\u4ef6
cp .env.example .env
# \u7f16\u8f91 .env \u586b\u5199\u5bc6\u7801\u548c\u6a21\u578b\u8def\u5f84

# 4. \u542f\u52a8\u540e\u7aef
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. (\u53ef\u9009) \u542f\u52a8 Milvus
docker-compose up -d milvus
# .env \u8bbe\u7f6e ENABLE_VECTOR_STORE=true

# 6. \u6253\u5f00\u524d\u7aef
# \u6d4f\u89c8\u5668\u8bbf\u95ee frontend/index.html
```
'''
f.write_text(t, encoding='utf-8')
print('s9 done, total chars:', len(t))
