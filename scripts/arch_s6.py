# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u516d\u3001\u5b58\u50a8\u5c42\u8bbe\u8ba1

| \u5c42 | \u7cfb\u7edf | \u5185\u5bb9 | \u8bbf\u95ee |
|---|--------|---------|-------|
| \u5143\u6570\u636e\u5c42 | PostgreSQL | \u4f1a\u8bdd\u72b6\u6001/\u8bf4\u8bdd\u4eba/\u7247\u6bb5\u6587\u672c | SQLAlchemy asyncpg |
| \u6587\u4ef6\u5c42 | MinIO | \u539f\u59cb\u5f55\u97f3WAV + \u8f6c\u5199TXT/JSON | \u9884\u7b7e\u540dURL\u76f4\u63a5\u4e0b\u8f7d |
| \u5411\u91cf\u5c42 | Milvus(\u53ef\u9009) | \u6587\u672c\u8bed\u4e49\u5411\u91cf | REST\u8bed\u4e49\u641c\u7d22 |

MinIO \u5b58\u50a8\u7ed3\u6784:
```
Bucket: speech-audio
  {session_id}/raw_audio.wav      -- \u5b8c\u6574\u539f\u59cb\u5f55\u97f3
Bucket: speech-text
  {session_id}/transcript.txt     -- \u7eaf\u6587\u672c\u8f6c\u5199
  {session_id}/transcript.json    -- \u7ed3\u6784\u5316JSON\uff08\u542b\u65f6\u95f4\u6233\uff09
```

Milvus\u96c6\u5408 speech_transcripts:
```
  id            INT64 PK auto
  session_id    VARCHAR(64)       \u6309\u4f1a\u8bdd\u8fc7\u6ee4
  speaker_label VARCHAR(32)       \u6309\u8bf4\u8bdd\u4eba\u8fc7\u6ee4
  text          VARCHAR(2000)
  embedding     FLOAT_VECTOR(384) MiniLM-L12-v2 384\u7ef4
  \u7d22\u5f15: HNSW metric=COSINE M=16
```
'''
f.write_text(t, encoding='utf-8')
print('s6', len(t))
