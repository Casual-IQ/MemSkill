import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from .base import Evaluator, register_evaluator


@register_evaluator("fotobot_traj")
class FotoBotEvaluator(Evaluator):
    """
    Evaluator for FotoBot camera trajectory logs and photographic skill extraction.
    Supports both offline JSON-action evaluation and real-time aesthetic tool reward parsing.
    """

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """Filter valid QA items that contain a prompt, question, task, or sample key."""
        valid_qa = []
        if not qa_list:
            return valid_qa

        for i, qa in enumerate(qa_list):
            if isinstance(qa, dict) and (qa.get('question') or qa.get('task') or qa.get('prompt') or qa.get('key')):
                valid_qa.append((i, qa))
        return valid_qa

    def prepare_eval_args(self) -> Any:
        """Set default generation arguments for evaluation."""
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 512
        return eval_args

    def build_prompt(self, question: str, retrieved_memories: List[str], qa_item: Dict) -> str:
        """Construct the evaluation prompt incorporating retrieved composition skills."""
        if len(retrieved_memories) > 0:
            context_parts = [f"[Composition Skill {i}]\n{mem}" for i, mem in enumerate(retrieved_memories, 1)]
            context = "\n\n".join(context_parts)
        else:
            context = "No specific photography skills retrieved."

        return (
            f"Active Photography Guidelines & Skills:\n{context}\n\n"
            f"Task Description & Scene Constraints:\n{question}\n\n"
            f"Provide the recommended camera adjustment steps or framing decision in standard tool_calls format:"
        )

    def get_ground_truth(self, qa_item: Dict) -> List[str]:
        """Extract ground truth answers from QA item if available."""
        if not qa_item:
            return []
        answers = qa_item.get('answers', qa_item.get('answer', ''))
        if isinstance(answers, list):
            return [str(a) for a in answers]
        return [str(answers)] if answers else []

    def get_episode_reward(self, data: Dict) -> float:
        """
        Extract real-time physical reward from 'evaluate_current_aesthetic' tool logs.
        Calculates a weighted aesthetic score based on rule, ven, and clip metrics.
        """
        if not isinstance(data, dict):
            return 0.0

        # 1. Parse reward from 'evaluate_current_aesthetic' tool response log
        if data.get("role") == "tool" and data.get("name") == "evaluate_current_aesthetic":
            content = data.get("content", "")
            try:
                content_json = json.loads(content) if isinstance(content, str) else content
                scores = content_json.get("scores", {})

                rule_score = scores.get("rule", 0.0)
                ven_score = scores.get("ven", 0.0)
                clip_score = scores.get("clip", 0.0)

                # Weighted combination: Rule (0.3), VEN (0.4), CLIP (0.3)
                return float(0.3 * rule_score + 0.4 * ven_score + 0.3 * clip_score)
            except (json.JSONDecodeError, AttributeError):
                return 0.0

        # 2. General fallback for direct reward fields
        for key in ("reward", "score", "episode_reward", "aesthetic_score"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0

    def _extract_action(self, response: str) -> str:
        """Extract clean JSON or tool-call payloads from model response, filtering out reasoning content."""
        if not response:
            return ""

        # 1. Extract markdown json code blocks
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()

        # 2. Extract explicit Tool Calls blocks
        tool_call_match = re.search(r'Tool Calls:\s*(\{.*?\})', response, re.DOTALL)
        if tool_call_match:
            return tool_call_match.group(1).strip()

        # 3. Fallback to extracting outermost dictionary structure
        dict_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if dict_match:
            return dict_match.group(0).strip()

        return response.strip()

    def compute_f1(self, prediction: str, ground_truth, qa_item: Dict = None) -> float:
        """
        重写 F1 方法作为统一 Reward 接口。
        结合 位姿偏差 (Pose Error) 与 美学打分 (Aesthetic Score)
        """
        if not prediction or not ground_truth:
            return 0.0

        # 1. 解析预测值与 Ground Truth 中的相机参数
        pred_params = self._extract_camera_params(str(prediction))
        gt_params = self._extract_camera_params(str(ground_truth))

        if not pred_params or not gt_params:
            return 0.0

        # 2. 计算数值位姿误差 (距离、角度)
        pos_err = np.linalg.norm(np.array(pred_params['pos']) - np.array(gt_params['pos']))
        rot_err = abs(pred_params['angle'] - gt_params['angle'])
        
        # 高斯核衰减函数，将误差映射到 [0, 1]
        pose_reward = np.exp(-0.5 * pos_err) * np.exp(-0.1 * rot_err)

        # 3. 提取场景审美/构图得分 (如果仿真器或 QA 中存在)
        aesthetic_score = 0.5
        if qa_item and isinstance(qa_item, dict):
            aesthetic_score = float(qa_item.get('aesthetic_score', 0.5))

        # 4. 综合 Reward (权重可调)
        total_reward = 0.6 * pose_reward + 0.4 * aesthetic_score
        return float(np.clip(total_reward, 0.0, 1.0))

    def _extract_camera_params(self, text: str) -> Dict[str, Any]:
        """从文本或 Tool Call 字符串中提取数值参数"""
        try:
            # 优先解析 JSON 类型的 Tool Call
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'pos': data.get('position', [0.0, 0.0, 0.0]),
                    'angle': data.get('angle', 0.0)
                }
        except Exception:
            pass
        return {}

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """Extract metadata for evaluation logging."""
        return {
            'scene_key': qa.get('key', 'unknown_scene'),
            'action_type': qa.get('type', 'camera_pose'),
        }