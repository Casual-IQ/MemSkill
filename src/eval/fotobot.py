import re
from typing import List, Dict, Any, Tuple
from .base import Evaluator, register_evaluator


@register_evaluator("fotobot_traj")
class FotoBotEvaluator(Evaluator):
    """
    Evaluator for FotoBot camera trajectory logs and photographic skill extraction.
    """

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        valid_qa = []
        if not qa_list:
            return valid_qa

        for i, qa in enumerate(qa_list):
            if isinstance(qa, dict) and (qa.get('question') or qa.get('task') or qa.get('prompt')):
                valid_qa.append((i, qa))
        return valid_qa

    def prepare_eval_args(self) -> Any:
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 512
        return eval_args

    def build_prompt(self, question: str, retrieved_memories: List[str], qa_item: Dict) -> str:
        if len(retrieved_memories) > 0:
            context_parts = [f"[Composition Skill {i}]\n{mem}" for i, mem in enumerate(retrieved_memories, 1)]
            context = "\n\n".join(context_parts)
        else:
            context = "No specific photography skills retrieved."

        return (
            f"Active Photography Guidelines & Skills:\n{context}\n\n"
            f"Task Description & Scene Constraints:\n{question}\n\n"
            f"Provide the recommended camera adjustment steps or framing decision:"
        )

    def get_ground_truth(self, qa_item: Dict) -> List[str]:
        if not qa_item:
            return []
        answers = qa_item.get('answers', qa_item.get('answer', ''))
        if isinstance(answers, list):
            return [str(a) for a in answers]
        return [str(answers)] if answers else []

    def compute_f1(self, prediction: str, ground_truth, qa_item: Dict = None) -> float:
        if not prediction or not ground_truth:
            return 0.0

        pred_tokens = set(re.findall(r'\w+', str(prediction).lower()))
        gt_str = " ".join([str(g) for g in ground_truth]) if isinstance(ground_truth, list) else str(ground_truth)
        gt_tokens = set(re.findall(r'\w+', gt_str.lower()))

        if not pred_tokens or not gt_tokens:
            return 0.0

        intersection = pred_tokens.intersection(gt_tokens)
        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        return {
            'scene_key': qa.get('key', 'unknown_scene'),
            'action_type': qa.get('type', 'camera_pose'),
        }