# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u4e94\u3001PostgreSQL \u8868\u8bbe\u8ba1

### 5.1 \u4e09\u8868\u5173\u7cfb\uff08\u903b\u8f91\u5173\u8054\uff0c\u65e0\u6570\u636e\u5e93\u5916\u952e\uff09

```
sessions \u8868
  id (UUID PK)
  title, status, language
  audio_object_key      -- MinIO \u8def\u5f84
  transcript_object_key -- MinIO \u8def\u5f84
  audio_duration_sec    -- \u5196\u4f59\u5b57\u6bb5\uff0c\u907f\u514d\u5b9e\u65f6SUM\u8ba1\u7b97
  speaker_count         -- \u5196\u4f59\u5b57\u6bb5\uff0c\u907f\u514d\u6bcf\u6b21COUNT
  created_at / completed_at
      |
      | \u901a\u8fc7 speakers.session_id = sessions.id \u903b\u8f91\u5173\u8054
      v
speakers \u8868
  id (UUID PK)
  session_id   (\u65e0\u5916\u952e\uff0c\u5efa\u7d22\u5f15)
  speaker_label     -- \u7cfb\u7edf\u5206\u914d: speaker_0, speaker_1
  display_name      -- \u7528\u6237\u7f16\u8f91: \u5f20\u67d0\u67d0\uff08\u53ef\u7a7a\uff09
  embedding (JSON)  -- \u58f0\u7eb9\u5907\u4efd\uff0c\u4e3b\u5b58 Milvus
  \u8054\u5408\u552f\u4e00\u7d22\u5f15: (session_id, speaker_label)
      |
      | \u901a\u8fc7 segments.speaker_id = speakers.id \u903b\u8f91\u5173\u8054
      v
transcript_segments \u8868
  id (UUID PK)
  session_id  (\u65e0\u5916\u952e)
  speaker_id  (\u65e0\u5916\u952e\uff0c\u53ef\u7a7a)
  start_time, end_time  -- Float \u79d2\uff0c\u7cbe\u5ea6 3 \u4f4d
  text                  -- \u8f6c\u5199\u6587\u672c
  sequence_no           -- \u7247\u6bb5\u5e8f\u53f7\uff0c\u4fdd\u8bc1\u6b63\u786e\u6392\u5e8f
  confidence            -- \u7f6e\u4fe1\u5ea6 0~1
  words (JSON)          -- \u8bcd\u7ea7\u65f6\u95f4\u6233\uff08Whisper \u63d0\u4f9b\uff09
  is_final              -- \u533a\u5206\u6d41\u5f0f\u4e2d\u95f4\u7ed3\u679c\u548c\u6700\u7ec8\u7ed3\u679c
```

### 5.2 \u4e3a\u4f55\u4e0d\u7528\u6570\u636e\u5e93\u5916\u952e\uff1f

| \u7ef4\u5ea6 | \u6709\u5916\u952e | \u65e0\u5916\u952e\uff08\u672c\u9879\u76ee\uff09 |
|------|--------|------------------|
| \u5199\u5165\u6027\u80fd | \u6bcf\u6b21INSERT\u9700\u9501\u5b9a\u7236\u8868\u884c\u9a8c\u8bc1 | \u76f4\u63a5\u5199\u5165\u65e0\u9501 |
| \u9ad8\u5e76\u53d1 | \u884c\u9501\u7ade\u4e89\u4e25\u91cd | \u7ebf\u6027\u6269\u5c55 |
| \u5206\u5e93\u5206\u8868 | \u8de8\u5e93\u5916\u952e\u4e0d\u53ef\u7ef4\u62a4 | \u65e0\u969c\u788d |
| \u4e00\u81f4\u6027 | \u6570\u636e\u5e93\u5f3a\u5236 | \u4ee3\u7801\u903b\u8f91\u4fdd\u8bc1 |

### 5.3 \u7d22\u5f15\u8bbe\u8ba1

```sql
-- speakers \u8868
INDEX (session_id, speaker_label) UNIQUE  -- \u540c\u4f1a\u8bdd\u5185\u6807\u7b7e\u4e0d\u91cd\u590d

-- transcript_segments \u8868
INDEX (session_id, sequence_no)   -- \u6700\u9ad8\u9891\uff1a\u987a\u5e8f\u8bfb\u5168\u90e8\u8f6c\u5199
INDEX (session_id, start_time)    -- \u65f6\u95f4\u533a\u95f4\u67e5\u8be2\uff08\u64ad\u653e\u5b9a\u4f4d\uff09
INDEX (speaker_id)                -- \u6309\u8bf4\u8bdd\u4eba\u7edf\u8ba1\u53d1\u8a00\u91cf
```
'''
f.write_text(t, encoding='utf-8')
print('s5', len(t))
