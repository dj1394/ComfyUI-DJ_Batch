# -*- coding: utf-8 -*-
"""
ComfyUI-DJ_Batch · 批量批次图片 — 图片批量加载节点（逐个加载模式）

关键约束 & 踩坑记录：
1. 每次执行只加载一张图片，用 SQLite 数据库存储进度计数器。
   和 WAS "Load Image Batch"、本包 批量批次音频 同一套机制：
   - IS_CHANGED 返回 NaN → 强制每次执行都重新运行，不走 ComfyUI 缓存
   - 每次执行后 index+1 存回数据库，下次执行加载下一张图
   - 到末尾后自动回到开头循环
   踩坑：之前一次性读完全部文件输出 batch tensor，下游节点只处理第一个，
         导致"无论怎么排输出都是同一张图"。
2. 输出 ComfyUI IMAGE：[1, H, W, C] float32 tensor（PIL → tensor，0~1）。
   - batch 维固定为 1（单张图）
   - 输出格式可选 RGB / RGBA
3. 解码用 PIL + exif_transpose（和 WAS Load Image Batch 一致），
   EXIF 方向自动纠正，手机拍的照片不会躺倒。
4. 路径参数用 STRING 类型（手动输入），不用 FILE 类型。
5. 自然排序用自写 natsort 逻辑（和 批量批次音频 同一套），
   不引入第三方包。支持阿拉伯数字、中文小写/大写数字、中英文括号归一化。
6. 进度数据库放在本插件目录下（基于 __file__ 相对推导，保证目录可迁移）。
"""

import os
import re
import random
import sqlite3
from pathlib import Path

from PIL import Image, ImageOps

# 随机模式说明（与 批量批次音频 同一套）：
# 用模块级 random.shuffle，每次执行都重新洗牌 → 真随机、不复现。
# 不用 random.Random(种子) 那种固定种子的写法，否则每次洗出同样顺序，"随机"名不副实。


# ─── SQLite 进度数据库 ──────────────────────────────────────────
# 和 WAS Load Image Batch、批量批次音频 一样，用 SQLite 存每个 label 的进度计数器。
# 数据库文件放在本插件目录下（基于 __file__ 相对推导，保证目录可迁移）。
_PLUGIN_DIR = Path(__file__).resolve().parent
_DB_PATH = _PLUGIN_DIR / "batch_image_progress.db"


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
    
    重置规则：
    - 路径变了 → 重置 index 为 0
    - 排序方式从"固定"切到非固定 → 保留当前 index（从固定文件开始轮询）
    - 排序方式从非固定切到"固定" → 保留当前 index（停留在当前文件）
    - 其他排序方式之间切换 → 重置 index 为 0（顺序完全不同）
    （pattern 存进 pattern 列，复用批量批次音频的表结构，不改表）
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
            old_path, old_pattern, old_index = row
            # 路径变了，重置
            if old_path != path:
                new_index = 0
            # 排序方式切换逻辑
            elif old_pattern == "固定" and sort_mode != "固定":
                # 从固定切到非固定：保留当前 index（从固定文件开始轮询）
                new_index = old_index
            elif old_pattern != "固定" and sort_mode == "固定":
                # 从非固定切到固定：保留当前 index（停留在当前文件）
                new_index = old_index
            elif old_pattern != sort_mode:
                # 其他排序方式之间切换：重置（顺序完全不同）
                new_index = 0
            else:
                # 相同排序方式：保持进度
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


# ─── 自然排序（和批量批次音频同一套）──────────────────────────────
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


# ─── 支持的图片扩展名（比 WAS 多了 .avif）──────────────────────
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.avif'}


def _pil2tensor(image: Image.Image):
    """PIL → ComfyUI IMAGE tensor：[1, H, W, C] float32，0~1。"""
    import numpy as np
    import torch
    arr = np.array(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        # 灰度图 → 单通道
        arr = arr[:, :, None]
    return torch.from_numpy(arr)[None, ...]


def _scan_image_files(directory: str, pattern: str) -> list[Path]:
    """
    扫描目录（不递归），按通配符过滤 + 扩展名白名单，返回自然排序后的列表。
    pattern 支持 glob 通配符（如 *.png、图*.jpg），和 WAS 的 pattern 参数一致。
    """
    import fnmatch
    files = [
        f for f in Path(directory).iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and fnmatch.fnmatch(f.name.lower(), pattern.lower())
    ]
    files.sort(key=_natural_sort_key)
    return files


def _load_image_file(path: Path) -> Image.Image:
    """PIL 打开 + EXIF 方向纠正（和 WAS Load Image Batch 一致）。"""
    img = Image.open(str(path))
    img = ImageOps.exif_transpose(img)
    return img


class BatchImageRotate_DJ:
    """
    图片批量加载：一个文件夹里的图片，每次执行加载一张，自动循环。
    和 WAS Load Image Batch、批量批次音频 同一套机制。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {
                    "default": "",
                    "placeholder": "输入图片文件夹路径",
                    "multiline": False,
                }),
                "标签": ("STRING", {
                    "default": "Batch 001",
                    "placeholder": "批次标签（不同标签独立计数）",
                    "multiline": False,
                }),
                "文件过滤": ("STRING", {
                    "default": "*",
                    "placeholder": "通配符，如 *.png、图*.jpg；* = 全部",
                    "multiline": False,
                }),
                "排序方式": (["文件名正序", "固定", "文件名倒序", "随机"], {
                    "default": "文件名正序",
                }),
                "最大文件数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                }),
                "RGBA透明通道": (["开", "关"], {
                    "default": "关",
                }),
                "输出带扩展名": (["开", "关"], {
                    "default": "关",
                }),
                "重置进度": (["开", "关"], {
                    "default": "关",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "文件名")
    CATEGORY = "DJ/Batch"
    FUNCTION = "run"

    # 信息面板说明（双击节点 / 悬停 / 右键→信息 可见）。
    # 支持格式从 IMAGE_EXTS 常量动态拼接，避免和扩展名白名单不同步。
    DESCRIPTION = (
        "批量加载图片：每次执行加载一张，自动循环，和 WAS Load Image Batch、批量批次音频 同一套机制。\n"
        "路径填图片文件夹；可用通配符过滤（如 *.png）；不同标签独立计数。\n"
        f"支持格式：{', '.join(sorted(IMAGE_EXTS))}"
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 返回 NaN 强制每次执行都重新运行，不走 ComfyUI 缓存。
        # 和 WAS Load Image Batch、批量批次音频 的 IS_CHANGED 一致。
        return float("NaN")

    def run(self, 路径: str, 标签: str = "Batch 001", 文件过滤: str = "*",
            排序方式: str = "文件名正序",
            RGBA透明通道: str = "关", 输出带扩展名: str = "关",
            最大文件数: int = 0, 重置进度: str = "关"):
        # strip 各种引号：半角 " ' 和全角 “ ” ‘ ’，防止粘贴时带入
        路径 = 路径.strip()
        for q in ('"', "'", '“', '”', '‘', '’'):
            路径 = 路径.strip(q)
        if not 路径:
            raise ValueError("路径不能为空，请输入图片文件夹路径")

        # ── 重置进度：删掉当前标签的数据库记录 ──
        if 重置进度 == "开":
            conn = _get_db()
            try:
                conn.execute('DELETE FROM progress WHERE label = ?', (标签,))
                conn.commit()
                print(f"[ComfyUI-DJ_Batch·批量批次图片] [{标签}] 进度已重置")
            finally:
                conn.close()

        p = Path(路径)
        if not p.is_dir():
            raise FileNotFoundError(f"路径不存在或不是文件夹：{路径}")

        # ── 扫描图片文件列表 ──
        image_files = _scan_image_files(str(p), 文件过滤 or "*")

        if not image_files:
            raise ValueError(
                f"没找到图片文件（支持 {', '.join(sorted(IMAGE_EXTS))}，"
                f"过滤规则：{文件过滤}）：{路径}"
            )

        # ── 排序 ──
        # 正序/倒序基于自然排序（数字按数值排，file2 < file10，支持中文数字）；
        # 随机模式每次执行 random.shuffle 重洗 → 真随机、不复现。
        if 排序方式 == "文件名倒序":
            image_files.sort(key=_natural_sort_key, reverse=True)
        elif 排序方式 == "随机":
            random.shuffle(image_files)
        else:
            # "文件名正序" 升序
            image_files.sort(key=_natural_sort_key)

        # ── 截断 ──
        if 最大文件数 > 0 and len(image_files) > 最大文件数:
            image_files = image_files[:最大文件数]

        total = len(image_files)

        # ── 从数据库读取进度，取当前文件 ──
        index = _get_next_index(标签, 路径, 排序方式)

        # 循环：到末尾后回到开头
        if index >= total:
            index = 0

        current_file = image_files[index]

        # ── 解码当前文件 ──
        img = _load_image_file(current_file)

        # RGBA透明通道：关 → 强制转 RGB；开 → 保持原样（有透明才 RGBA，没透明就 RGB）
        if RGBA透明通道 == "关":
            img = img.convert("RGB")

        # ── 文件名输出 ──
        filename = current_file.name
        if 输出带扩展名 == "关":
            filename = os.path.splitext(filename)[0]

        # ── 保存下一个进度 ──
        if 排序方式 == "固定":
            # 固定模式：不推进 index，停留在当前文件
            _save_index(标签, 路径, 排序方式, index)
        else:
            # 正常模式：推进到下一个文件
            next_index = index + 1
            if next_index >= total:
                next_index = 0  # 循环回开头
            _save_index(标签, 路径, 排序方式, next_index)

        print(
            f"[ComfyUI-DJ_Batch·批量批次图片] [{标签}] "
            f"{index + 1}/{total}: {current_file.name} "
            f"({img.width}x{img.height})"
        )

        # 输出 ComfyUI IMAGE tensor [1,H,W,C] + 文件名
        return (_pil2tensor(img), filename)
