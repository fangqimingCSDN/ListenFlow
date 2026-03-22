# -*- coding: utf-8 -*-
import pathlib
f = pathlib.Path(r'd:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
t = f.read_text(encoding='utf-8')
t += '''
---

## \u4e09\u3001\u6838\u5fc3\u6570\u636e\u6d41\uff1a\u4e00\u53e5\u8bdd\u5982\u4f55\u88ab\u8bc6\u522b

### 3.1 \u5b8c\u6574\u5904\u7406\u94fe\u8def\uff089 \u6b65\uff09

```
STEP1  \u7528\u6237\u5f00\u53e3\u8bf4\u8bdd
       [\u6d4f\u89c8\u5668 AudioContext ScriptProcessor]
       \u6bcf 256ms \u91c7\u96c6 4096 \u6837\u672c\uff0816kHz\uff09
       Float32 -> Int16 -> base64 -> WebSocket

STEP2  [ws_handler.py] \u6536\u5230 {"type":"audio","data":"<base64>"}
       base64 \u89e3\u7801 -> PCM bytes

STEP3  [VADService.process_chunk(bytes)]  \u6bcf\u4f1a\u8bdd\u72ec\u7acb\u5b9e\u4f8b
       |
       +-- \u672a\u68c0\u6d4b\u5230\u8bed\u97f3\u7ed3\u675f -> \u8fd4\u56de\u7a7a\u5217\u8868 -> \u7ee7\u7eed\u63a5\u6536\u4e0b\u4e00\u5e27
       |
       +-- \u68c0\u6d4b\u5230\u8bed\u97f3\u7ed3\u675f\uff08\u4e24\u4e2a\u89e6\u53d1\u6761\u4ef6\u4e4b\u4e00\uff09
             \u6761\u4ef6A: \u9759\u97f3\u6301\u7eed > VAD_SILENCE_DURATION_MS\uff08\u9ed8\u8ba4800ms\uff09
             \u6761\u4ef6B: \u5355\u6bb5\u65f6\u957f > VAD_MAX_SEGMENT_DURATION\uff08\u9ed8\u8ba430s\uff09
             \u8fd4\u56de [(start_sec, end_sec, audio_np_array)]

STEP4  asyncio.create_task(_process_segment(...))
       \u7acb\u5373\u8fd4\u56de\uff0c\u4e0d\u963b\u585e\u63a5\u6536\u5faa\u73af
       WebSocket \u7ee7\u7eed\u63a5\u6536\u4e0b\u4e00\u4e2a\u97f3\u9891\u5305

STEP5  asyncio.gather \u5e76\u884c\u6267\u884c

       +---------------------------+---------------------------+
       |                           |                           |
       v                           v                           |
  [ASRService.transcribe]  [SpeakerService.extract_embedding] |
  asyncio.to_thread          asyncio.to_thread                |
  \u7ebf\u7a0b\u6c60\u6267\u884c 2~5s         \u7ebf\u7a0b\u6c60\u6267\u884c 0.3~0.5s             |
  FunASR paraformer-zh      ERes2Net 192\u7ef4\u58f0\u7eb9\u5d4c\u5165              |
  \u8fd4\u56de: text, confidence    \u8fd4\u56de: np.ndarray                      |
       |                           |                           |
       +---------------------------+                           |
       | \u4e24\u8005\u90fd\u5b8c\u6210\u540e\u624d\u7ee7\u7eed                              |
       v

STEP6  [OnlineSpeakerCluster.update(embedding)]
       \u4e0e\u5386\u53f2\u58f0\u7eb9\u4f59\u5f26\u76f8\u4f3c\u5ea6\u5339\u914d
       \u76f8\u4f3c\u5ea6 > 0.55 -> \u5f52\u5165\u5df2\u6709 speaker_N  (\u540c\u4e00\u4eba)
       \u76f8\u4f3c\u5ea6 < 0.55 -> \u65b0\u5efa speaker_N+1    (\u65b0\u8ba4\u8bc6\u7684\u4eba)
       \u67e5 speaker_names \u6620\u5c04 -> \u8fd4\u56de ("speaker_0", "\u5f20\u67d0\u67d0")

STEP7  SpeechSession.add_segment(text, speaker, start, end)
       \u5199\u5165\u5185\u5b58\u7247\u6bb5\u5217\u8868

STEP8  fire-and-forget \u5f02\u6b65\u5199\u5b58\u50a8
       create_task(_write_segment_to_db)   -> \u5199 PostgreSQL
       create_task(milvus.insert_segment)  -> \u5199 Milvus\uff08\u5982\u5df2\u5f00\u542f\uff09
       \u4e0d\u7b49\u5b83\u4eec\u5b8c\u6210\uff0c\u76f4\u63a5\u8fdb\u884c\u4e0b\u4e00\u6b65

STEP9  WebSocket.send_json \u7acb\u5373\u63a8\u9001
       {
         "type": "transcript",
         "seq": 3,
         "text": "\u8fd9\u4e2a\u65b9\u6848\u7684\u4ef7\u683c\u600e\u4e48\u6837",
         "speaker_label": "speaker_0",
         "speaker_display": "\u5f20\u67d0\u67d0",
         "start": 12.34,
         "end": 14.89
       }
       \u524d\u7aef\u6e32\u67d3: [\u5f20\u67d0\u67d0] 12:34  \u8fd9\u4e2a\u65b9\u6848\u7684\u4ef7\u683c\u600e\u4e48\u6837
```

### 3.2 \u4f1a\u8bdd\u751f\u547d\u5468\u671f\u72b6\u6001\u673a

```
   \u5ba2\u6237\u7aef\u53d1 {"type":"start"}
           |
           v
     [ recording ]  <------+
           |               |
     \u53d1\u9001 pause        \u53d1\u9001 resume
           v               |
      [ paused ] ---------+
           |
     \u53d1\u9001 stop \u6216 WS \u65ad\u5f00
           |
           v
    [ finalizing ] -- \u540e\u53f0\u5f02\u6b65 -->
        PCM -> WAV -> \u4e0a\u4f20 MinIO
        \u751f\u6210 TXT/JSON -> \u4e0a\u4f20 MinIO
        PG: sessions.status = completed
           |
           v
     [ completed ]
     \u63a8\u9001 session_completed\uff08\u542b\u4e0b\u8f7d\u94fe\u63a5\uff09
```
'''
f.write_text(t, encoding='utf-8')
print('s3', len(t))
