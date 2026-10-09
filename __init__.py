# ComfyUI-DJ_Batch — 批次工具箱（批量批次音频 + 批量批次图片）
# 多节点包：Node ID = ComfyUI-DJ_中文功能名，显示名与 ID 相同（用户破例定的格式）
# 坑：注册ID 不能和内置/已装插件重名，否则静默丢弃
# 坑：两个节点共用 CATEGORY "DJ/Batch"，在节点树里归到同一个文件夹下

from .batch_audio import BatchAudioImport_DJ
from .batch_image import BatchImageRotate_DJ

NODE_CLASS_MAPPINGS = {
    "ComfyUI-DJ_批量批次音频": BatchAudioImport_DJ,
    "ComfyUI-DJ_批量批次图片": BatchImageRotate_DJ,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # 显示名与注册ID逐字符相同（用户定的格式），映射必须保留
    "ComfyUI-DJ_批量批次音频": "ComfyUI-DJ_批量批次音频",
    "ComfyUI-DJ_批量批次图片": "ComfyUI-DJ_批量批次图片",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
