"""Instruction prompts, answer templates and answer parsing for the LLM baseline.

The wording follows MentalGLM (C-IMHI): the user turn is "考虑以下这个帖子，<question>\n<text>"
(the format of the released evaluation predictions) and the answer opens with a first sentence
that names the label, e.g. "这篇帖子体现了高自杀风险。". MentalGLM's own evaluation reads the label
from that first sentence (Evaluation/LLM/evaluate.py); `parse_answer` does the same.

For MATCHA there is no explanation data (posts may not be sent to GPT-4), so the fine-tuning
targets are the label sentence only. The Chinese names below are the verbalizers: each label
id maps to one name, and no name is a substring of another, so parsing is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

DISORDERS_ZH = {
    "ADHD": "注意缺陷多动障碍",
    "ASD": "孤独症谱系障碍",
    "BD": "双相情感障碍",
    "DEP": "抑郁症",
    "GAD": "广泛性焦虑障碍",
    "OCD": "强迫症",
    "SZ": "精神分裂症",
}

# SocialCD-3K label order (same as data/tasks/cognitive/meta.json); names as in C-IMHI
COGNITIVE_ZH = ["非此即彼", "以偏概全", "心理过滤", "否定正面思考", "读心术", "先知错误",
                "放大", "情绪化推理", "应该句式", "乱贴标签", "罪责归己", "罪责归他"]


@dataclass(frozen=True)
class TaskPrompt:
    question: str           # the user turn; "{text}" is replaced by the post or timeline
    verbalizers: list[str]  # Chinese name of each label id, in label order
    answer: str             # the answer's first sentence; "{label}" is replaced by the name(s)
    multi_label: bool = False

    def user_turn(self, text: str) -> str:
        return self.question.replace("{text}", text)

    def target(self, label) -> str:
        """Training answer for a label id (single-label) or a 0/1 vector (multi-label)."""
        if self.multi_label:
            names = [n for n, y in zip(self.verbalizers, label) if int(y) == 1]
            return self.answer.replace("{label}", ", ".join(names) if names else "无")
        return self.answer.replace("{label}", self.verbalizers[int(label)])


def _options(names) -> str:
    return "、".join(names)


_POST = "考虑以下这个帖子，"
_USER = "考虑以下这位用户最近发布的微博（多条微博用“｜”分隔），"
_MATCHA_DISORDERS = [DISORDERS_ZH[k] for k in ["ADHD", "ASD", "BD", "DEP", "GAD", "OCD", "SZ"]]
_MATCHA_GROUPS = [d + "组" for d in _MATCHA_DISORDERS] + ["对照组"]  # user level: "the depression group"

PROMPTS: dict[str, TaskPrompt] = {
    # MentalGLM's own tasks, worded exactly as in their released evaluation
    "suicide": TaskPrompt(
        _POST + "帖子体现了高自杀风险还是低自杀风险？\n{text}",
        ["低自杀风险", "高自杀风险"],
        "这篇帖子体现了{label}。"),
    "cognitive": TaskPrompt(
        _POST + "帖子体现了什么认知歪曲？\n{text}",
        COGNITIVE_ZH,
        "这篇帖子体现了{label}的认知歪曲。",
        multi_label=True),
    # MATCHA tasks (label order as in data/tasks/<task>/meta.json)
    "matcha_disorder_post": TaskPrompt(
        _POST + f"帖子来自哪个精神障碍的超话社区？选项：{_options(_MATCHA_DISORDERS)}。\n{{text}}",
        _MATCHA_DISORDERS,
        "这篇帖子来自{label}社区。"),
    "matcha_user_binary": TaskPrompt(
        _USER + "该用户属于精神障碍组还是对照组？\n{text}",
        ["对照组", "精神障碍组"],
        "这位用户属于{label}。"),
    "matcha_user_multiclass": TaskPrompt(
        _USER + f"该用户属于哪一组？选项：{_options(_MATCHA_GROUPS)}。\n{{text}}",
        _MATCHA_GROUPS,
        "这位用户属于{label}。"),
}
PROMPTS["matcha_disorder_post_masked"] = PROMPTS["matcha_disorder_post"]
PROMPTS["matcha_user_binary_masked"] = PROMPTS["matcha_user_binary"]
PROMPTS["matcha_user_multiclass_masked"] = PROMPTS["matcha_user_multiclass"]


def get_prompt(task: str) -> TaskPrompt:
    if task not in PROMPTS:
        raise KeyError(f"no LLM prompt for task {task!r}; add one to matcha_bench.llm.prompts")
    return PROMPTS[task]


def first_sentence(answer: str) -> str:
    return answer.strip().split("。")[0]


def parse_answer(answer: str, prompt: TaskPrompt):
    """Label from the first sentence of a generated answer.

    Single-label: the label id whose name occurs first, or -1 if none does (counted as wrong).
    Multi-label: a 0/1 list with every named label (all zeros if none).
    """
    s = first_sentence(answer)
    if prompt.multi_label:
        return [int(name in s) for name in prompt.verbalizers]
    hits = [(s.find(name), i) for i, name in enumerate(prompt.verbalizers) if name in s]
    return min(hits)[1] if hits else -1
