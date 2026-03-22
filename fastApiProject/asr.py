from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import threading
from threading import Lock
import base64
import numpy as np
import torch
from funasr import AutoModel
from silero_vad import load_silero_vad, VADIterator
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import time
import threading
from collections import deque
from pydantic import BaseModel
import uvicorn
from typing import Optional
import os
import json
import zipfile
import io
import soundfile as sf
from datetime import datetime
import shutil
from pydub import AudioSegment
import tempfile

app = FastAPI(title="SenseVoice实时语音识别API", version="1.0.0")

# 创建APIRouter并设置统一前缀
api_router = APIRouter(prefix="/sensevoice/api/v1")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD要求的样本数
BASE_DATA_DIR = "user_sessions"  # 用户数据存储目录

# 确保数据目录存在
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = "G:\\f_models"
# 初始化模型
print("正在加载模型...")
# model = AutoModel(model='iic/SenseVoiceSmall', trust_remote_code=True,
#                   device="cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )
# res = model.generate(input=f"{model.model_path}/example/asr_example.wav",
#                      batch_size_s=300,
#                      hotword='魔搭')
vad = load_silero_vad()
vad_iter = VADIterator(vad, threshold=0.3)
print("模型加载完成")
cam = AutoModel(model="cam++"
                # , model_revision="v2.0.2"
                )

# online_cluster.py
import numpy as np
from typing import List, Dict


class OnlineSpeakerCluster:
    """
    流式说话人聚类
    usage:
        clu = OnlineSpeakerCluster(thr=0.78, ema=0.5)
        for emb in embs:          # emb: [C] numpy
            spk_id = clu.update(emb)
            print(spk_id)
    """

    def __init__(self, thr: float = 0.58, ema: float = 0.9):
        """
        thr: 余弦相似度阈值，超过则归到已有类
        ema: 更新中心时的指数滑动平均系数 α
             center = α * new + (1-α) * center
        """
        self.thr = thr
        self.ema = ema
        self.centers: List[np.ndarray] = []   # 每个说话人中心

    def update(self, x: np.ndarray) -> int:
        """单步更新，返回当前片段说话人 ID（从 0 开始）"""
        if len(self.centers) == 0:            # 第一个片段
            self.centers.append(x.copy())
            return 0

        # 与所有已有中心算余弦相似度
        x_norm = x / (np.linalg.norm(x) + 1e-12)
        centers_norm = np.stack([c / (np.linalg.norm(c) + 1e-12) for c in self.centers])
        sims = np.dot(centers_norm, x_norm)   # [n_spk]

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.thr:              # 归入最近类
            # EMA 更新中心
            self.centers[best_idx] = self.ema * x + (1 - self.ema) * self.centers[best_idx]
            return best_idx
        else:                                 # 新建说话人
            self.centers.append(x.copy())
            return len(self.centers) - 1

    def reset(self):
        """清空，重新开始"""
        self.centers.clear()

# 定义请求模型
class AudioRequest(BaseModel):
    audio: str
    session_id: Optional[str] = None

class SpeakerNameUpdateRequest(BaseModel):
    speaker_mapping: Dict[str, str]


# 定义响应模型
class AudioResponse(BaseModel):
    text: Optional[str] = None
    status: str
    error: Optional[str] = None


# 用户会话状态管理
user_sessions = {}
session_locks = {}  # 会话锁，避免多个会话同时访问
THRESH   = 0.8         # 聚类阈值
# 移除全局cluster，改为每个会话独立的cluster

# 获取会话锁的辅助函数
def get_session_lock(session_id):
    """获取会话锁，确保同一会话的请求串行处理"""
    if session_id not in session_locks:
        session_locks[session_id] = Lock()
    return session_locks[session_id]

# 会话管理锁
session_management_lock = Lock()

@api_router.post("/sessions/{session_id}/update_speaker_names")
async def update_speaker_names(request: SpeakerNameUpdateRequest, session_id: str):
    """更新说话人名称映射"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        user_sessions[session_id].update_speaker_names(request.speaker_mapping)
        return {"status": "success", "message": "说话人名称映射已更新", "speaker_mapping": request.speaker_mapping}

@api_router.get("/sessions/{session_id}/download_named_transcript")
async def download_named_transcript(session_id: str):
    """下载带有真实姓名的转录文本文件"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = user_sessions[session_id]
        named_file = session.save_named_transcript()
        
        if not named_file or not os.path.exists(named_file):
            raise HTTPException(status_code=404, detail="无法生成带姓名的转录文本文件")
        
        return FileResponse(
            named_file,
            media_type="text/plain",
            filename=f"{session_id}_named_transcriptions.txt"
        )

@api_router.get("/sessions/{session_id}/speaker_names")
async def get_speaker_names(session_id: str):
    """获取当前会话的说话人名称映射"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = user_sessions[session_id]
        return {
            "session_id": session_id,
            "speaker_names": session.speaker_names,
            "message": "获取成功"
        }

@api_router.get("/sessions/{session_id}/preview_named_transcript")
async def preview_named_transcript(session_id: str):
    """预览带真实姓名的转录文本内容"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = user_sessions[session_id]
    named_file = session.save_named_transcript()
    
    if not named_file or not os.path.exists(named_file):
        raise HTTPException(status_code=404, detail="无法生成带姓名的转录文本文件")
    
    # 读取文件内容并返回
    with open(named_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        "session_id": session_id,
        "content": content,
        "filename": f"{session_id}_named_transcriptions.txt",
        "speaker_mapping": session.speaker_names
    }
from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2net_sv_zh-cn_16k-common',
    # model_revision='v1.0.5'
)

class UserSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.audio_chunks = []
        self.prev_audio_chunks = deque(maxlen=10)  # 保存最近的10个chunk
        self.speaking = False
        self.last_active = time.time()
        self.residual = np.zeros(0, dtype=np.float32)
        self.session_dir = os.path.join(BASE_DATA_DIR, session_id)
        self.utterance_count = 0
        self.combined_audio = None
        self.speaker_names = {}  # 说话人名称映射，如 {"speaker_0": "张三", "speaker_1": "李四"}
        self.is_recording = True  # 录音状态：True表示正在录音，False表示已停止
        self.is_paused = False   # 暂停状态：True表示已暂停，False表示正常进行

        # 为每个会话创建独立的聚类和VAD实例
        self.cluster = OnlineSpeakerCluster(thr=0.4, ema=0.9)
        self.vad_iter = VADIterator(vad, threshold=0.3)
        
        # 创建用户目录
        os.makedirs(self.session_dir, exist_ok=True)

        # 创建元数据文件
        self.metadata_file = os.path.join(self.session_dir, "metadata.json")
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "utterances": [],
                    "speaker_names": self.speaker_names,
                    "is_recording": self.is_recording
                }, f, indent=2)

    def add_audio_data(self, audio_data):
        # 检查录音状态，如果已停止或已暂停则不再处理新的音频数据
        if not self.is_recording or self.is_paused:
            return None
            
        self.last_active = time.time()

        # 确保音频数据长度正确
        if len(audio_data) != CHUNK_SIZE:
            raise ValueError(f"音频数据长度必须为{CHUNK_SIZE}，实际为{len(audio_data)}")

        # 处理音频数据，使用会话独立的VAD实例
        speech_dict = self.vad_iter(audio_data)

        if speech_dict:
            if not self.speaking and 'start' in speech_dict:
                self.speaking = True
                print(f"会话 {self.session_id}: 检测到语音开始")

        if self.speaking:
            self.audio_chunks.append(audio_data)
        else:
            self.prev_audio_chunks.append(audio_data)
            # 限制历史缓冲区大小
            if len(self.prev_audio_chunks) > 10:
                self.prev_audio_chunks.popleft()

        if speech_dict and 'end' in speech_dict:
            self.speaking = False
            if self.audio_chunks:
                # 合并音频片段（历史缓冲区 + 当前语音）
                all_chunks = list(self.prev_audio_chunks) + self.audio_chunks
                if all_chunks:
                    audio = np.concatenate(all_chunks)

                    # 执行语音识别
                    try:
                        # result = model.inference(audio * 32768, language='zh', disable_pbar=True)
                        result = model.generate(input=audio * 32768,
                                             batch_size_s=300, language='zh')
                        # emb = np.array(cam.generate(input=audio * 32768)[0]["spk_embedding"]).squeeze()
                        # print(emb.shape)
                        # 使用会话独立的聚类实例
                        emb = sv_pipeline([audio * 32768], output_emb=True)['embs'].squeeze()
                        speaker = self.cluster.update(emb)
                        # print(result)
                        print(f"会话 {self.session_id}: spk{speaker}")
                        # print('spk',result[0]['sentence_info'][0]['spk'])
                        # speaker = result[0]['sentence_info'][0]['spk']
                        text = rich_transcription_postprocess(result[0]['text'])
                        print(f"会话 {self.session_id}: 识别结果 - {text}")

                        # 保存音频和文本
                        self.save_utterance(audio, text, speaker)

                        # 清空缓冲区
                        self.audio_chunks = []
                        self.prev_audio_chunks.clear()

                        return 'speaker_' + str(speaker) + '              ' + text
                    except Exception as e:
                        print(f"会话 {self.session_id}: 识别错误 - {str(e)}")
                        return None

        return None

    def stop_recording(self):
        """停止录音"""
        if self.is_recording:
            self.is_recording = False
            # 更新元数据文件
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['is_recording'] = self.is_recording
                    metadata['stopped_at'] = datetime.now().isoformat()
                    f.seek(0)
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    f.truncate()
            
            # 处理最后一个未完成的语音片段
            if self.speaking and self.audio_chunks:
                # 合并音频片段（历史缓冲区 + 当前语音）
                all_chunks = list(self.prev_audio_chunks) + self.audio_chunks
                if all_chunks:
                    audio = np.concatenate(all_chunks)

                    # 执行语音识别
                    try:
                        result = model.generate(input=audio * 32768, batch_size_s=300, language='zh')
                        emb = sv_pipeline([audio * 32768], output_emb=True)['embs'].squeeze()
                        speaker = self.cluster.update(emb)
                        text = rich_transcription_postprocess(result[0]['text'])
                        print(f"会话 {self.session_id}: 最后的识别结果 - {text}")

                        # 保存音频和文本
                        self.save_utterance(audio, text, speaker)
                    except Exception as e:
                        print(f"会话 {self.session_id}: 最后语音识别错误 - {str(e)}")

            # 清空缓冲区
            self.audio_chunks = []
            self.prev_audio_chunks.clear()
            self.speaking = False
            
            print(f"会话 {self.session_id}: 录音已停止")
            return True
        return False

    def pause_recording(self):
        """暂停录音"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            # 更新元数据文件
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['is_paused'] = self.is_paused
                    metadata['paused_at'] = datetime.now().isoformat()
                    f.seek(0)
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    f.truncate()
            
            print(f"会话 {self.session_id}: 录音已暂停")
            return True
        return False

    def resume_recording(self):
        """恢复录音"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            # 更新元数据文件
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['is_paused'] = self.is_paused
                    metadata['resumed_at'] = datetime.now().isoformat()
                    f.seek(0)
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    f.truncate()
            
            print(f"会话 {self.session_id}: 录音已恢复")
            return True
        return False

    def save_utterance(self, audio_data, text,speaker):
        """保存语音片段和识别文本"""
        self.utterance_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"utterance_{self.utterance_count:04d}_{timestamp}"

        # 临时保存为WAV
        wav_file = os.path.join(self.session_dir, f"{filename}.wav")
        sf.write(wav_file, audio_data, SAMPLE_RATE)

        # 转换为MP3
        mp3_file = os.path.join(self.session_dir, f"{filename}.mp3")
        audio_segment = AudioSegment.from_wav(wav_file)
        audio_segment.export(mp3_file, format="mp3", bitrate="64k")

        # 删除临时WAV文件
        os.remove(wav_file)

        # 保存文本文件
        text_file = os.path.join(self.session_dir, f"{filename}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)

        # 更新完整会话音频
        self.update_combined_audio(audio_segment)

        # 更新元数据
        with open(self.metadata_file, 'r+', encoding='utf-8') as f:
            metadata = json.load(f)
            metadata["utterances"].append({
                "id": self.utterance_count,
                "timestamp": datetime.now().isoformat(),
                "audio_file": f"{filename}.mp3",
                "text_file": f"{filename}.txt",
                "text": text,
                "speaker": speaker,
                "duration": len(audio_segment) / 1000.0  # 秒
            })
            f.seek(0)
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            f.truncate()

    def update_combined_audio(self, audio_segment):
        """更新完整的会话音频"""
        if self.combined_audio is None:
            self.combined_audio = audio_segment
        else:
            self.combined_audio += audio_segment

        # 保存完整的MP3文件
        combined_mp3 = os.path.join(self.session_dir, "complete_session.mp3")
        self.combined_audio.export(combined_mp3, format="mp3", bitrate="64k")

    def update_speaker_names(self, speaker_mapping):
        """更新说话人名称映射"""
        self.speaker_names.update(speaker_mapping)
        # 更新元数据文件
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                metadata = json.load(f)
                metadata['speaker_names'] = self.speaker_names
                f.seek(0)
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.truncate()

    def save_named_transcript(self):
        """保存带有真实姓名的转录文本文件，一行一句"""
        # 读取元数据
        if not os.path.exists(self.metadata_file):
            return None
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 创建带姓名映射的文本文件
        named_file = os.path.join(self.session_dir, "named_transcriptions.txt")
        
        with open(named_file, 'w', encoding='utf-8') as f:
            # 只保留对话内容，不要头信息
            for utterance in metadata['utterances']:
                speaker_id = utterance.get('speaker', '')
                speaker_key = f"speaker_{speaker_id}"
                
                # 获取说话人姓名，如果没有映射则使用原始ID
                speaker_name = self.speaker_names.get(speaker_key, speaker_key)
                text = utterance.get('text', '')
                
                # 按照要求格式：姓名 + 制表符 + 句子
                f.write(f"{speaker_name}\t{text}\n")
        
        return named_file

    def update_speaker_names(self, speaker_mapping):
        """更新说话人名称映射"""
        self.speaker_names.update(speaker_mapping)
        # 更新元数据文件
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                metadata = json.load(f)
                metadata['speaker_names'] = self.speaker_names
                f.seek(0)
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.truncate()

    def save_named_transcript(self):
        """保存带有真实姓名的转录文本文件，一行一句"""
        # 读取元数据
        if not os.path.exists(self.metadata_file):
            return None
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 创建带姓名映射的文本文件
        named_file = os.path.join(self.session_dir, "named_transcriptions.txt")
        
        with open(named_file, 'w', encoding='utf-8') as f:
            # 只保留对话内容，不要头信息
            for utterance in metadata['utterances']:
                speaker_id = utterance.get('speaker', '')
                speaker_key = f"speaker_{speaker_id}"
                
                # 获取说话人姓名，如果没有映射则使用原始ID
                speaker_name = self.speaker_names.get(speaker_key, speaker_key)
                text = utterance.get('text', '')
                
                # 按照要求格式：姓名 + 制表符 + 句子
                f.write(f"{speaker_name}\t{text}\n")
        
        return named_file

    def save_complete_transcript(self):
        """保存完整的转录文本文件"""
        transcript_file = os.path.join(self.session_dir, "complete_transcript.txt")
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"会话ID: {self.session_id}\n")
            f.write(f"创建时间: {metadata['created_at']}\n")
            f.write(f"语音片段数量: {len(metadata['utterances'])}\n")
            f.write(f"总时长: {sum(utterance['duration'] for utterance in metadata['utterances']):.2f}秒\n")
            f.write("=" * 50 + "\n\n")

            for i, utterance in enumerate(metadata['utterances'], 1):
                f.write(f"片段 {i} [{utterance['timestamp']}]:\n")
                f.write(f"{utterance['text']}\n")
                f.write(f"时长: {utterance['duration']:.2f}秒\n")
                f.write("-" * 30 + "\n")


# 清理过期会话的线程
def cleanup_sessions():
    while True:
        time.sleep(60)  # 每分钟检查一次
        current_time = time.time()
        expired_sessions = []

        for session_id, session in user_sessions.items():
            if current_time - session.last_active > 300:  # 5分钟无活动视为过期
                # 保存完整转录文本
                session.save_complete_transcript()
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            # 保留用户数据，只从内存中移除会话
            del user_sessions[session_id]
            print(f"已清理过期会话: {session_id}")


# 启动清理线程
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()


@api_router.post("/asr", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    try:
        # 1. 取会话
        session_id = request.session_id or "default"
        if session_id not in user_sessions:
            user_sessions[session_id] = UserSession(session_id)
        session = user_sessions[session_id]

        # 2. base64 → numpy
        audio_bytes = base64.b64decode(request.audio)
        if not audio_bytes:
            return AudioResponse(status="listening")

        # 3. 字节长度对齐 4 的倍数（float32 元素大小）
        remainder = len(audio_bytes) % 4
        if remainder:
            audio_bytes += b'\0' * (4 - remainder)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # 4. 拼上上一包剩余
        audio_array = np.concatenate([session.residual, audio_array])

        # 5. 按 512 样本一批批喂 VAD
        while len(audio_array) >= CHUNK_SIZE:
            chunk = audio_array[:CHUNK_SIZE]
            audio_array = audio_array[CHUNK_SIZE:]
            result_text = session.add_audio_data(chunk)
            if result_text:  # 一旦识别出文字立即返回
                return AudioResponse(text=result_text, status="complete")

        # 6. 剩余样本留到下一包
        session.residual = audio_array
        return AudioResponse(status="listening")

    except Exception as e:
        return AudioResponse(status="error", error=str(e))


@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "active_sessions": len(user_sessions)
    }


@api_router.get("/sessions")
async def list_sessions():
    sessions_info = {}
    for session_id, session in user_sessions.items():
        sessions_info[session_id] = {
            "last_active": session.last_active,
            "speaking": session.speaking,
            "audio_chunks": len(session.audio_chunks),
            "prev_chunks": len(session.prev_audio_chunks),
            "utterance_count": session.utterance_count
        }
    return sessions_info


@api_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in user_sessions:
        del user_sessions[session_id]
        # 同时删除用户数据目录
        session_dir = os.path.join(BASE_DATA_DIR, session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@api_router.get("/sessions/{session_id}/download")
async def download_session_data(session_id: str):
    """下载完整的MP3和文本文件（不包含片段文件）"""
    session_dir = os.path.join(BASE_DATA_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session data not found")

    # 确保有完整的转录文本
    complete_transcript = os.path.join(session_dir, "complete_transcript.txt")
    complete_audio = os.path.join(session_dir, "complete_session.mp3")
    complete_metadata = os.path.join(session_dir, "metadata.json")

    # 如果会话还在内存中，保存完整转录
    if session_id in user_sessions:
        user_sessions[session_id].save_complete_transcript()

    # 检查文件是否存在
    if not os.path.exists(complete_transcript) or not os.path.exists(complete_audio):
        raise HTTPException(status_code=404, detail="Complete files not found")

    # 创建内存中的ZIP文件，只包含完整文件
    def generate_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 添加完整音频文件
            zip_file.write(complete_audio, f"{session_id}_complete.mp3")
            # 添加完整文本文件
            zip_file.write(complete_transcript, f"{session_id}_transcript.txt")
            zip_file.write(complete_metadata, f"{session_id}_metadata.json")

        zip_buffer.seek(0)
        return zip_buffer

    # 使用StreamingResponse返回ZIP文件
    zip_buffer = generate_zip()
    return StreamingResponse(
        io.BytesIO(zip_buffer.getvalue()),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={session_id}_complete_files.zip",
            "Content-Length": str(zip_buffer.getbuffer().nbytes)
        }
    )


@api_router.get("/sessions/{session_id}/metadata")
async def get_session_metadata(session_id: str):
    metadata_file = os.path.join(BASE_DATA_DIR, session_id, "metadata.json")
    if not os.path.exists(metadata_file):
        raise HTTPException(status_code=404, detail="Session metadata not found")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


@api_router.get("/sessions/{session_id}/complete_audio")
async def get_complete_audio(session_id: str):
    """下载完整的会话音频MP3文件"""
    complete_mp3 = os.path.join(BASE_DATA_DIR, session_id, "complete_session.mp3")
    if not os.path.exists(complete_mp3):
        raise HTTPException(status_code=404, detail="Complete audio not found")

    return FileResponse(
        complete_mp3,
        media_type="audio/mpeg",
        filename=f"{session_id}_complete.mp3"
    )


@api_router.get("/sessions/{session_id}/complete_transcript")
async def get_complete_transcript(session_id: str):
    """下载完整的转录文本文件"""
    transcript_file = os.path.join(BASE_DATA_DIR, session_id, "complete_transcript.txt")
    if not os.path.exists(transcript_file):
        # 如果文件不存在，尝试生成
        metadata_file = os.path.join(BASE_DATA_DIR, session_id, "metadata.json")
        if not os.path.exists(metadata_file):
            raise HTTPException(status_code=404, detail="Session data not found")

        # 生成完整转录
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"会话ID: {session_id}\n")
            f.write(f"创建时间: {metadata['created_at']}\n")
            f.write(f"语音片段数量: {len(metadata['utterances'])}\n")
            f.write(f"总时长: {sum(utterance['duration'] for utterance in metadata['utterances']):.2f}秒\n")
            f.write("=" * 50 + "\n\n")

            for i, utterance in enumerate(metadata['utterances'], 1):
                f.write(f"片段 {i} [{utterance['timestamp']}]:\n")
                f.write(f"{utterance['text']}\n")
                f.write(f"时长: {utterance['duration']:.2f}秒\n")
                f.write("-" * 30 + "\n")

    return FileResponse(
        transcript_file,
        media_type="text/plain",
        filename=f"{session_id}_transcript.txt"
    )


@api_router.get("/sessions/{session_id}/files")
async def list_session_files(session_id: str):
    """列出会话中的所有文件"""
    session_dir = os.path.join(BASE_DATA_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session data not found")

    files = []
    for file in os.listdir(session_dir):
        file_path = os.path.join(session_dir, file)
        if os.path.isfile(file_path):
            files.append({
                "name": file,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })

    return {"files": files}


@api_router.post("/sessions/{session_id}/stop_recording")
async def stop_recording(session_id: str):
    """停止录音并结束当前会话"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = user_sessions[session_id]
    
    # 停止录音
    stopped = session.stop_recording()
    
    if stopped:
        # 保存完整转录文本
        session.save_complete_transcript()
        
        # 如果需要，也可以从内存中删除会话（可选）
        # del user_sessions[session_id]
        
        return {
            "status": "success",
            "message": "录音已停止，会话已结束",
            "session_id": session_id,
            "utterance_count": session.utterance_count,
            "is_recording": session.is_recording
        }
    else:
        return {
            "status": "info",
            "message": "录音已经处于停止状态",
            "session_id": session_id,
            "is_recording": session.is_recording
        }


@api_router.get("/sessions/{session_id}/recording_status")
async def get_recording_status(session_id: str):
    """获取会话录音状态"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = user_sessions[session_id]
        
        return {
            "session_id": session_id,
            "is_recording": session.is_recording,
            "is_paused": session.is_paused,
            "utterance_count": session.utterance_count,
            "speaking": session.speaking,
            "last_active": session.last_active
        }

@api_router.post("/sessions/{session_id}/pause_recording")
async def pause_recording(session_id: str):
    """暂停录音"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = user_sessions[session_id]
        
        # 暂停录音
        paused = session.pause_recording()
        
        if paused:
            return {
                "status": "success",
                "message": "录音已暂停，会话保持活跃",
                "session_id": session_id,
                "is_recording": session.is_recording,
                "is_paused": session.is_paused,
                "utterance_count": session.utterance_count
            }
        else:
            return {
                "status": "info",
                "message": "录音已经处于暂停状态或录音已停止",
                "session_id": session_id,
                "is_recording": session.is_recording,
                "is_paused": session.is_paused
            }

@api_router.post("/sessions/{session_id}/resume_recording")
async def resume_recording(session_id: str):
    """恢复录音"""
    with get_session_lock(session_id):
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = user_sessions[session_id]
        
        # 恢复录音
        resumed = session.resume_recording()
        
        if resumed:
            return {
                "status": "success",
                "message": "录音已恢复",
                "session_id": session_id,
                "is_recording": session.is_recording,
                "is_paused": session.is_paused,
                "utterance_count": session.utterance_count
            }
        else:
            return {
                "status": "info",
                "message": "录音已经处于进行状态或录音已停止",
                "session_id": session_id,
                "is_recording": session.is_recording,
                "is_paused": session.is_paused
            }


# 包含API路由
app.include_router(api_router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
