"""阶段 4 · 多智能体实验用的编码任务与判分器。

判分不靠模型"感觉对不对"，而是把生成的代码和一组**隐藏测试**一起
放进独立子进程里跑。测试通过 = 成功，否则 = 失败。
这就是阶段 5 eval 的雏形：先有一个确定、可重复的判分器。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeTask:
    slug: str
    spec: str        # 给模型的自然语言需求（不含测试）
    tests: str       # 隐藏测试：模型看不到，判分用


@dataclass(frozen=True)
class TestResult:
    ok: bool
    output: str

    def brief(self, limit: int = 800) -> str:
        text = self.output.strip() or "（无输出）"
        return text if len(text) <= limit else text[:limit] + "\n…（已截断）"


TASKS: list[CodeTask] = [
    CodeTask(
        slug="merge_intervals",
        spec="""\
实现 merge_intervals(intervals: list[list[int]]) -> list[list[int]]。
- 每个区间是 [start, end]，且 start <= end。
- 合并所有重叠或首尾相接的区间；例如 [1,3] 和 [3,6] 要合并成 [1,6]。
- 返回值按 start 升序排列。
- 不修改传入的列表。
- 空列表返回空列表。
示例：[[1,3],[2,6],[8,10],[15,18]] -> [[1,6],[8,10],[15,18]]
""",
        tests="""\
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[5,7],[1,3],[2,4]]) == [[1,4],[5,7]]
original = [[1,3],[2,6]]
_ = merge_intervals(original)
assert original == [[1,3],[2,6]], "不应修改输入"
print("ALL TESTS PASSED")
""",
    ),
    CodeTask(
        slug="parse_duration",
        spec="""\
实现 parse_duration(text: str) -> int，把时长字符串解析成秒数。
- 支持单位 h（小时）、m（分钟）、s（秒），可任意组合，例如 "1h30m"、"2m10s"、"45s"。
- 单位必须按 h、m、s 的顺序出现，每个单位最多一次。
- 空字符串返回 0。
- 格式非法（重复单位、未知字符、顺序错误、数字缺失）时抛出 ValueError。
- 只处理整数，不处理小数。
示例："1h30m" -> 5400；"45m" -> 2700；"2h" -> 7200。
""",
        tests="""\
assert parse_duration("1h30m") == 5400
assert parse_duration("45m") == 2700
assert parse_duration("2h") == 7200
assert parse_duration("0s") == 0
assert parse_duration("") == 0
assert parse_duration("1h2m3s") == 3723
for bad in ["1m1h", "1h1h", "abc", "10", "1x"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} 应该抛出 ValueError")
print("ALL TESTS PASSED")
""",
    ),
    CodeTask(
        slug="chunk_list",
        spec="""\
实现 chunk_list(items: list, size: int) -> list[list]。
- 按顺序把列表切成若干块，每块最多 size 个元素，最后一块可以不满。
- size <= 0 时抛出 ValueError。
- 空列表返回空列表。
- 不修改原列表。
示例：[1,2,3,4,5], 2 -> [[1,2],[3,4],[5]]
""",
        tests="""\
assert chunk_list([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunk_list([], 3) == []
assert chunk_list([1,2], 5) == [[1,2]]
original = [1,2,3]
_ = chunk_list(original, 2)
assert original == [1,2,3]
for bad_size in [0, -1]:
    try:
        chunk_list([1], bad_size)
    except ValueError:
        pass
    else:
        raise AssertionError("size <= 0 应该抛出 ValueError")
print("ALL TESTS PASSED")
""",
    ),
    CodeTask(
        slug="eval_expr",
        spec="""\
实现 eval_expr(text: str) -> float，计算只含非负整数、+ - * / **、括号和空白的表达式。
- 支持一元负号，且一元负号优先级低于 **：-2**2 == -4，(-2)**2 == 4。
- ** 是右结合：2**3**2 == 512。
- 除法返回 float，例如 5/2 == 2.5。
- 除以 0 抛出 ValueError。
- 空串、括号不匹配、连续运算符等非法输入抛出 ValueError。
- 不允许使用 eval 或 exec。
""",
        tests="""\
assert eval_expr("1+2*3") == 7
assert eval_expr("(1+2)*3") == 9
assert eval_expr("2**3**2") == 512, "** 必须右结合"
assert eval_expr("-2**2") == -4, "一元负号优先级低于 **"
assert eval_expr("(-2)**2") == 4
assert eval_expr("5/2") == 2.5
assert eval_expr(" 10 - 2 * (3 + 1) ") == 2
for bad in ["", "1+", "(1+2", "1++2", "abc", "1/0"]:
    try:
        eval_expr(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} 应该抛出 ValueError")
print("ALL TESTS PASSED")
""",
    ),
    CodeTask(
        slug="spiral_order",
        spec="""\
实现 spiral_order(matrix: list[list[int]]) -> list[int]，按顺时针螺旋顺序返回所有元素。
- 从左上角出发，先向右，再向下、向左、向上，逐层向内。
- 空矩阵返回空列表。
- 支持非方阵（每行列数相同，但行数可以不等于列数）。
- 不修改输入。
示例：[[1,2,3],[4,5,6],[7,8,9]] -> [1,2,3,6,9,8,7,4,5]
""",
        tests="""\
assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]
assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]
assert spiral_order([[1],[2],[3]]) == [1,2,3]
assert spiral_order([[1,2,3]]) == [1,2,3]
assert spiral_order([]) == []
original = [[1,2],[3,4]]
_ = spiral_order(original)
assert original == [[1,2],[3,4]]
print("ALL TESTS PASSED")
""",
    ),
    CodeTask(
        slug="top_k_frequent",
        spec="""\
实现 top_k_frequent(nums: list[int], k: int) -> list[int]。
- 返回出现次数最多的 k 个元素。
- 次数相同时，数值小的排在前面（保证结果唯一）。
- k <= 0 或 k 大于不同元素个数时抛出 ValueError。
示例：[1,1,1,2,2,3], 2 -> [1,2]
""",
        tests="""\
assert top_k_frequent([1,1,1,2,2,3], 2) == [1,2]
assert top_k_frequent([4,4,5,5,6], 2) == [4,5], "次数相同按数值升序"
assert top_k_frequent([7], 1) == [7]
for bad in [0, -1, 3]:
    try:
        top_k_frequent([1,2], bad)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 k 应该抛出 ValueError")
print("ALL TESTS PASSED")
""",
    ),
]


def extract_code(text: str) -> str:
    """从模型回复里取出代码：优先取 ```python 围栏，去掉围栏外的解释。"""
    text = text.strip()
    if "```" in text:
        blocks = text.split("```")
        # 奇数下标是围栏内容；优先带 python 标记的块
        candidates: list[str] = []
        for index in range(1, len(blocks), 2):
            block = blocks[index]
            first_line, _, rest = block.partition("\n")
            if first_line.strip().lower() in {"python", "py", ""}:
                candidates.append(rest if first_line.strip() else block)
            else:
                candidates.append(block)
        if candidates:
            return max(candidates, key=len).strip()
    return text


def run_tests(code: str, task: CodeTask, timeout: float = 15.0) -> TestResult:
    """把代码和隐藏测试拼成一个文件，在独立子进程里执行。"""
    program = code.rstrip() + "\n\n# ---- hidden tests ----\n" + task.tests
    with tempfile.TemporaryDirectory(prefix="stage4-code-") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            completed = subprocess.run(
                # -X utf8：让子进程的 stdout/stderr 也按 UTF-8 编码，
                # 否则 Windows 控制台会把中文断言信息打成乱码。
                [sys.executable, "-X", "utf8", "-s", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, f"超时（>{timeout}s），可能存在死循环")
    output = (completed.stdout or "") + (completed.stderr or "")
    ok = completed.returncode == 0 and "ALL TESTS PASSED" in output
    return TestResult(ok, output)
