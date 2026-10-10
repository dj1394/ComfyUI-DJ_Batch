# -*- coding: utf-8 -*-
"""
ComfyUI-DJ_Batch · 批量批次音频 — 批量音频导入节点（逐个加载模式）

关键约束 & 踩坑记录：
1. 每次执行只加载一个音频文件，用 SQLite 数据库存储进度计数器。
   和 WAS "Load Image Batch" 同一套机制：
   - IS_CHANGED 返回 NaN → 强制每次执行都重新运行，不走 ComfyUI 缓存
   - 每次执行后 index+1 存回数据库，下次执行加载下一个文件
   - 到末尾后自动回到开头循环
   踩坑：之前一次性读完全部文件输出 batch tensor，下游节点只处理第一个，
         导致"无论正序倒序输出都是同一个文件"。
2. 输出 ComfyUI AUDIO dict：{"waveform": [1, C, S], "sample_rate": int}
   - batch 维固定为 1（单个文件）
   - SoundFlow 等下游节点要求 waveform 是 3D [B, C, S]
   - key 用 "sample_rate"（SoundFlow 检查的拼写），不是官方 _io.py 里的 "sampler_rate"
3. 解码用 PyAV（import av），和 ComfyUI 官方 LoadAudio 节点一致。
   踩坑：之前用 torchaudio.load()，新版走 torchcodec 后端，torchcodec 没装时
   import 成功但调用报 "TorchCodec is required"，全部文件解码失败。
   PyAV 底层自带 ffmpeg，不依赖 torchcodec，ComfyUI 必装，稳定可用。
4. zip 解压到临时目录，用完即删。
5. 路径参数用 STRING 类型（手动输入），不用 FILE 类型。
6. 自然排序用自写 natsort 逻辑，不引入第三方包。
"""

import os
import random
import re
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import torch
import av

# 随机模式说明（与 批量批次图片 同一套）：
# 用模块级 random.shuffle，每次执行都重新洗牌 → 真随机、不复现。
# 不用 random.Random(种子) 那种固定种子的写法，否则每次洗出同样顺序，"随机"名不副实。


# ─── SQLite 进度数据库 ──────────────────────────────────────────
# 和 WAS Load Image Batch 一样，用 SQLite 存每个 label 的进度计数器。
# 数据库文件放在本插件目录下（基于 __file__ 相对推导，保证目录可迁移）。
_PLUGIN_DIR = Path(__file__).resolve().parent
_DB_PATH = _PLUGIN_DIR / "batch_audio_progress.db"


def _get_db() -> sqlite3.Connection:
    """获取 SQLite 连接，自动建表。"""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        'CREATE TABLE IF NOT EXISTS progress ('
        '  label TEXT PRIMARY KEY,'
        '  path TEXT NOT NULL,'
        '  pattern TEXT NOT NULL DEFAULT \'*\','
        '  "index" INTEGER NOT NULL DEFAULT 0'
        ')'
    )
    return conn


def _get_next_index(label: str, path: str, sort_mode: str) -> int:
    """
    获取下一个文件索引，并更新数据库。
    如果路径或排序方式任一变了，重置 index 为 0。
    """
    conn = _get_db()
    try:
        row = conn.execute(
            'SELECT path, pattern, "index" FROM progress WHERE label = ?', (label,)
        ).fetchone()

        if row is None:
            # 第一次使用这个 label，从 0 开始
            new_index = 0
            conn.execute(
                'INSERT INTO progress (label, path, pattern, "index") VALUES (?, ?, ?, ?)',
                (label, path, sort_mode, new_index),
            )
        else:
            old_path, old_sort, old_index = row
            # 路径或排序方式任一变了，都重置（排序方式变了文件顺序完全不同）
            if old_path != path or old_sort != sort_mode:
                new_index = 0
            else:
                new_index = old_index
            conn.execute(
                'UPDATE progress SET path = ?, pattern = ?, "index" = ? WHERE label = ?',
                (path, sort_mode, new_index, label),
            )

        conn.commit()
        return new_index
    finally:
        conn.close()


def _save_index(label: str, path: str, sort_mode: str, new_index: int):
    """保存新的索引到数据库。"""
    conn = _get_db()
    try:
        conn.execute(
            'UPDATE progress SET path = ?, pattern = ?, "index" = ? WHERE label = ?',
            (path, sort_mode, new_index, label),
        )
        conn.commit()
    finally:
        conn.close()


# ─── 自然排序（不依赖 natsort 包）──────────────────────────────
# 支持：阿拉伯数字、中文小写数字(一二三...)、中文大写数字(壹贰叁...)
# 括号归一化：中英文括号都转成半角 ()，避免 (1) 和 （1）排不到一起

# 中文数字映射（小写 + 大写统一映射到 int）
_CN_DIGIT_MAP = {
    '零': 0, '〇': 0, '○': 0,
    '一': 1, '壹': 1,
    '二': 2, '贰': 2, '两': 2,
    '三': 3, '叁': 3,
    '四': 4, '肆': 4,
    '五': 5, '伍': 5,
    '六': 6, '陆': 6,
    '七': 7, '柒': 7,
    '八': 8, '捌': 8,
    '九': 9, '玖': 9,
}
# 中文数字单位
_CN_UNIT_MAP = {
    '十': 10, '拾': 10,
    '百': 100, '佰': 100,
    '千': 1000, '仟': 1000,
    '万': 10000,
}

# 匹配数字段：阿拉伯数字 或 中文数字（含单位组合如"二十三"、"壹佰零伍"）
_NUM_PATTERN = re.compile(
    r'(\d+)'                          # 阿拉伯数字
    r'|([零〇○一二两三四五六七八九十壹贰叁肆伍陆柒捌玖拾佰仟百千万]+)'  # 中文数字
)


def _cn_to_int(s: str) -> int:
    """把中文数字字符串转成 int。支持"二十三"、"壹佰零伍"等组合。"""
    if not s:
        return 0
    total = 0
    current = 0
    for ch in s:
        if ch in _CN_DIGIT_MAP:
            current = _CN_DIGIT_MAP[ch]
        elif ch in _CN_UNIT_MAP:
            unit = _CN_UNIT_MAP[ch]
            if current == 0:
                current = 1  # "十" = 1×10
            total += current * unit
            current = 0
        else:
            # 非数字字符，忽略
            pass
    total += current
    return total


def _normalize_brackets(s: str) -> str:
    """中英文括号统一转半角 ()。"""
    return (s.replace('（', '(').replace('）', ')')
              .replace('【', '[').replace('】', ']')
              .replace('｛', '{').replace('｝', '}'))


def _natural_sort_key(filepath):
    """
    把文件名拆成数字/非数字段，让 file2 < file10。
    支持阿拉伯数字、中文小写/大写数字、中英文括号归一化。
    """
    name = Path(filepath).stem  # 不含扩展名
    name = _normalize_brackets(name)

    result = []
    last_end = 0
    for m in _NUM_PATTERN.finditer(name):
        # 数字前的非数字段
        if m.start() > last_end:
            text = name[last_end:m.start()].lower()
            result.append((1, text))
        # 数字段：阿拉伯 or 中文，统一转 int
        if m.group(1) is not None:
            val = int(m.group(1))
        else:
            val = _cn_to_int(m.group(2))
        result.append((0, val))
        last_end = m.end()

    # 尾部非数字段
    if last_end < len(name):
        text = name[last_end:].lower()
        result.append((1, text))

    return result


# ─── 支持的音频扩展名────────────────────────────────────────────
AUDIO_EXTS = {'.wav', '.mp3', '.aac'}


def _scan_audio_files(directory: str) -> list[Path]:
    """扫描目录（不递归），返回按自然排序的音频文件列表。"""
    files = [
        f for f in Path(directory).iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    ]
    files.sort(key=_natural_sort_key)
    return files


def _extract_zip(zip_path: str, dest_dir: str) -> list[Path]:
    """解压 zip 到 dest_dir，只提取音频文件，返回排序后的列表。"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        audio_members = [
            m for m in zf.namelist()
            if not m.endswith('/') and Path(m).suffix.lower() in AUDIO_EXTS
        ]
        if not audio_members:
            raise ValueError(f"ZIP 里没找到音频文件（支持 {', '.join(AUDIO_EXTS)}）")
        zf.extractall(dest_dir, members=audio_members)

    all_files = [
        f for f in Path(dest_dir).rglob('*')
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    ]
    all_files.sort(key=_natural_sort_key)
    return all_files


def _decode_audio_file(path: Path):
    """
    用 PyAV（和 ComfyUI 官方 LoadAudio 一致）解码单个音频文件。
    返回 (waveform, sample_rate)：
      waveform: [channels, samples] float32 tensor
      sample_rate: int
    """
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"没有音频流：{path.name}")

        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in container.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError(f"没有解码出音频帧：{path.name}")

        wav = torch.cat(frames, dim=1)
        # 确保 float32（PyAV 可能返回 int16/int32）
        if wav.dtype == torch.int16:
            wav = wav.float() / (2 ** 15)
        elif wav.dtype == torch.int32:
            wav = wav.float() / (2 ** 31)
        else:
            wav = wav.float()

    return wav, sr


class BatchAudioImport_DJ:
    """
    批量导入音频文件（逐个加载模式）。
    每次执行加载一个文件，自动循环，和 WAS Load Image Batch 同一套机制。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {
                    "default": "",
                    "placeholder": "输入文件夹路径或 .zip 文件路径",
                    "multiline": False,
                }),
                "标签": ("STRING", {
                    "default": "Batch 001",
                    "placeholder": "批次标签（不同标签独立计数）",
                    "multiline": False,
                }),
                "排序方式": (["自然排序", "文件名正序", "文件名倒序", "随机"], {
                    "default": "文件名正序",
                }),
                "最大文件数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                }),
                "重置进度": (["否", "是"], {
                    "default": "否",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "文件名")
    CATEGORY = "DJ/Batch"
    FUNCTION = "run"

    # 信息面板说明（双击节点 / 悬停 / 右键→信息 可见）。
    # 支持格式从 AUDIO_EXTS 常量动态拼接，避免和扩展名白名单不同步。
    DESCRIPTION = (
        "批量导入音频：每次执行加载一个文件，自动循环，和 WAS Load Image Batch 同一套机制。\n"
        "路径可填文件夹或 .zip；不同标签独立计数。\n"
        f"支持格式：{', '.join(sorted(AUDIO_EXTS))}"
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 返回 NaN 强制每次执行都重新运行，不走 ComfyUI 缓存。
        # 和 WAS Load Image Batch 的 IS_CHANGED 一致。
        return float("NaN")

    def run(self, 路径: str, 标签: str = "Batch 001",
            排序方式: str = "文件名正序", 最大文件数: int = 0,
            重置进度: str = "否"):
        # strip 各种引号：半角 " ' 和全角 “ ” ‘ ’，防止粘贴时带入
        路径 = 路径.strip()
        for q in ('"', "'", '“', '”', '‘', '’'):
            路径 = 路径.strip(q)
        if not 路径:
            raise ValueError("路径不能为空，请输入文件夹或 zip 文件路径")

        # ── 重置进度：删掉当前标签的数据库记录 ──
        if 重置进度 == "是":
            conn = _get_db()
            try:
                conn.execute('DELETE FROM progress WHERE label = ?', (标签,))
                conn.commit()
                print(f"[ComfyUI-DJ_Batch·批量批次音频] [{标签}] 进度已重置")
            finally:
                conn.close()

        p = Path(路径)
        temp_dir = None

        try:
            # ── 确定音频文件列表 ──
            if p.suffix.lower() == '.zip':
                if not p.is_file():
                    raise FileNotFoundError(f"ZIP 文件不存在：{路径}")
                temp_dir = tempfile.mkdtemp(prefix="dj_batch_audio_")
                audio_files = _extract_zip(str(p), temp_dir)

            elif p.is_dir():
                audio_files = _scan_audio_files(str(p))

            else:
                raise FileNotFoundError(f"路径不存在或不是文件夹/ZIP：{路径}")

            if not audio_files:
                raise ValueError(
                    f"没找到音频文件（支持 {', '.join(AUDIO_EXTS)}）：{路径}"
                )

            # ── 排序 ──
            # 正序/倒序基于自然排序（数字按数值排，file2 < file10）；
            # "自然排序" = 同正序；随机模式每次执行 random.shuffle 重洗 → 真随机、不复现。
            if 排序方式 == "文件名倒序":
                audio_files.sort(key=_natural_sort_key, reverse=True)
            elif 排序方式 == "随机":
                random.shuffle(audio_files)
            else:
                # "自然排序" 和 "文件名正序" 都是升序
                audio_files.sort(key=_natural_sort_key)

            # ── 截断 ──
            if 最大文件数 > 0 and len(audio_files) > 最大文件数:
                audio_files = audio_files[:最大文件数]

            total = len(audio_files)

            # ── 从数据库读取进度，取当前文件 ──
            index = _get_next_index(标签, 路径, 排序方式)

            # 循环：到末尾后回到开头
            if index >= total:
                index = 0

            current_file = audio_files[index]

            # ── 解码当前文件 ──
            wav, sr = _decode_audio_file(current_file)

            # [channels, samples] → [1, channels, samples]（加 batch 维）
            # SoundFlow 等下游节点要求 3D tensor
            wav_batched = wav.unsqueeze(0)

            # ── 保存下一个进度 ──
            next_index = index + 1
            if next_index >= total:
                next_index = 0  # 循环回开头
            _save_index(标签, 路径, 排序方式, next_index)

            print(
                f"[ComfyUI-DJ_Batch·批量批次音频] [{标签}] "
                f"{index + 1}/{total}: {current_file.name} "
                f"({sr}Hz, {wav.shape[0]}ch, {wav.shape[1]}samples)"
            )

            # 输出 ComfyUI AUDIO dict + 文件名
            return ({"waveform": wav_batched, "sample_rate": sr}, current_file.name)

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
