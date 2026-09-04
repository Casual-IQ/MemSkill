import json
from typing import Dict, List, Any
from .base import DataProcessor, ChunkMode, register_processor


@register_processor('fotobot_traj')
class FotoBotProcessor(DataProcessor):
    """Data processor for FotoBot real-time inference inputs and memory context."""

    supported_chunk_modes = [
        ChunkMode.TURN,
        ChunkMode.TURN_PAIR,
        ChunkMode.FULL_SESSION,
        ChunkMode.FIXED_LENGTH
    ]

    def extract_chunks(self, data: Dict) -> List[str]:
        """Extract conversation context chunks for memory retrieval, ignoring metadata."""
        if not isinstance(data, dict):
            return [str(data)]

        # Filter out session metadata rows
        if data.get("_type") == "metadata":
            return []

        # Handle pre-aggregated session structure: {"key": "...", "messages": [...]}
        messages = data.get("messages", [])
        if messages:
            return self._extract_session_chunks(messages)

        # Handle single message streaming/line-by-line input
        formatted_text = self._format_single_message(data)
        return [formatted_text] if formatted_text else []

    def _format_single_message(self, msg: Dict) -> str:
        """Format a single message into plain text with role and reasoning metadata."""
        if msg.get("_type") == "metadata":
            return ""

        role = msg.get("role", "")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        tool_calls = msg.get("tool_calls", None)

        parts = []
        if role:
            parts.append(f"[{role.upper()}]")
        if reasoning:
            parts.append(f"[Reasoning/CoT]\n{reasoning}")
        if content:
            content_str = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
            parts.append(f"[Content]\n{content_str}")
        if tool_calls:
            parts.append(f"[Tool Calls Executed]\n{json.dumps(tool_calls, ensure_ascii=False, indent=2)}")

        return "\n".join(parts).strip()

    def _extract_session_chunks(self, messages: List[Dict]) -> List[str]:
        """Aggregate chunks based on configured ChunkMode."""
        valid_msgs = [m for m in messages if m.get("_type") != "metadata"]

        if self.chunk_mode == ChunkMode.FULL_SESSION:
            session_text = "\n\n".join([self._format_single_message(m) for m in valid_msgs if self._format_single_message(m)])
            return [session_text] if session_text else []

        elif self.chunk_mode == ChunkMode.TURN_PAIR:
            chunks = []
            for i in range(0, len(valid_msgs), 2):
                pair = valid_msgs[i:i+2]
                pair_text = "\n\n".join([self._format_single_message(m) for m in pair if self._format_single_message(m)])
                if pair_text:
                    chunks.append(pair_text)
            return chunks

        return [self._format_single_message(m) for m in valid_msgs if self._format_single_message(m)]

    def get_sample_id(self, data: Dict) -> str:
        """Extract sample identifier."""
        if isinstance(data, dict):
            return str(data.get('key', data.get('sample_id', data.get('id', 'fotobot_sample_0'))))
        return 'fotobot_sample_0'

    def get_qa_list(self, data: Dict) -> List[Dict[str, Any]]:
        """Construct QA items for evaluation and prompt generation, capturing selected skills."""
        if isinstance(data, dict) and 'qa_list' in data and data['qa_list']:
            return data['qa_list']

        # Skip metadata and raw tool response logs from QA generation
        if not isinstance(data, dict) or data.get("_type") == "metadata" or data.get("role") == "tool":
            return []

        # Extract ground truth if available (optional)
        answers = []
        if data.get("tool_calls"):
            answers.append(json.dumps(data["tool_calls"], ensure_ascii=False))

        selected_skill_text = data.get("selected_skill_text", "")
        selected_skill_name = data.get("selected_skill_name", "")

        return [{
            "question": "What is the expected camera framing tool action or response given the current visual and trajectory context?",
            "answers": answers,
            "key": self.get_sample_id(data),
            "selected_skill_text": selected_skill_text,
            "selected_skill_name": selected_skill_name
        }]