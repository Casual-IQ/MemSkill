import json
from typing import Dict, List, Any
from .base import DataProcessor, ChunkMode, register_processor


@register_processor('fotobot_traj')
class FotoBotProcessor(DataProcessor):
    """Data processor for FotoBot camera trajectory logs."""

    supported_chunk_modes = [
        ChunkMode.TURN,
        ChunkMode.TURN_PAIR,
        ChunkMode.FULL_SESSION,
        ChunkMode.PARAGRAPH,
        ChunkMode.FIXED_LENGTH
    ]

    def extract_chunks(self, data: Dict) -> List[str]:
        """将 JSONL 中的单条记录提取为有意义的文本 Chunk"""
        if not isinstance(data, dict):
            return [str(data)]

        # 1. 如果存在显式的 action_history 优先使用
        actions = data.get("action_history", [])
        if actions:
            return [
                json.dumps(act, ensure_ascii=False) if isinstance(act, dict) else str(act)
                for act in actions
            ]

        # 2. 适配你的 session.jsonl 结构：提取 role、content 和 tool_calls
        role = data.get("role", "")
        content = data.get("content", "")
        tool_calls = data.get("tool_calls", None)

        chunk_text = ""
        if role:
            chunk_text += f"[{role.upper()}]\n"
        if content:
            chunk_text += f"{content}\n"
        if tool_calls:
            chunk_text += f"Tool Calls: {json.dumps(tool_calls, ensure_ascii=False)}"

        return [chunk_text.strip()] if chunk_text.strip() else [json.dumps(data, ensure_ascii=False)]

    def get_sample_id(self, data: Dict) -> str:
        if isinstance(data, dict):
            return str(data.get('sample_id', data.get('key', data.get('id', 'fotobot_sample_0'))))
        return 'fotobot_sample_0'

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """构建基础 QA 用于计算 Reward，驱动 Skill 生成"""
        if isinstance(data, dict) and 'qa_list' in data and data['qa_list']:
            return data['qa_list']

        # 如果 JSONL 里没有 QA，自动提取/构造一条打分问题
        return [{
            "question": "What standard camera framing or action should be used in this scene?",
            "answers": [data.get("content", "adjust camera")],  # 用参考动作作为 ground truth
            "key": self.get_sample_id(data)
        }]