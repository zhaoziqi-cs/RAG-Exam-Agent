import re
from pathlib import Path
from typing import List, Tuple

from src.schemas import ExamQuestion
    
QUESTION_HEADER_RE = re.compile(r"^\s*(\d+)[、\.．]\s*(.+?)\s*$")
OPTION_LINE_RE = re.compile(
    r"^\s*([a-zA-ZＡ-Ｚ])(?:[、\.．\)\）:：]\s*|\s+)(.+?)\s*$"
)
MULTIPLE_MARK = "【多选题】"

def normalize_whitespace(text: str) -> str:
    """压缩多余空白，保留正文语义。"""
    return re.sub(r"\s+", " ", text).strip()
    
def read_input_text(file_path: str) -> str:
    """读取 input.txt 文本。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return path.read_text(encoding="utf-8")

def preclean_text(text: str) -> str:
    """
    轻量清洗：
    1. 统一换行
    2. 去掉 BOM
    3. 去掉每行行尾空白
    4. 保留正文结构，不做过度清洗
    """
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()

def split_question_blocks(text: str) -> List[str]:
    """
    按单独一行的 * 切分题块。
    允许文件开头/中间/结尾出现 *，空块会自动跳过。
    """
    blocks: List[str] = []
    current_lines: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        if line == "*":
            if current_lines:
                blocks.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines).strip())

    return blocks

def parse_question_header(header_line: str) -> Tuple[str, str, str]:
    """
    从题头提取:
    - question_id
    - question
    - question_type
    """
    match = QUESTION_HEADER_RE.match(header_line)
    if not match:
        raise ValueError(f"题干首行格式不合法: {header_line}")

    question_id, raw_question = match.groups()

    question_type = "multiple" if MULTIPLE_MARK in raw_question else "single"
    question = raw_question.replace(MULTIPLE_MARK, "")
    question = normalize_whitespace(question)

    if not question:
        raise ValueError(f"题干为空: {header_line}")

    return question_id, question, question_type

def split_question_and_option_lines(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    支持题干跨行：
    - 第一个选项出现前的行，视为题干续行
    - 第一个选项出现后的行，归入选项区域
    """
    question_lines: List[str] = []
    option_lines: List[str] = []
    seen_option = False

    for line in lines:
        if OPTION_LINE_RE.match(line):
            seen_option = True
        if seen_option:
            option_lines.append(line)
        else:
            question_lines.append(line)

    return question_lines, option_lines


def strip_option_prefix(option_line: str) -> str:
    """
    去掉 A、 / A. / A) 之类的选项前缀，只保留纯文本内容。
    """
    match = OPTION_LINE_RE.match(option_line)
    if not match:
        raise ValueError(f"不是合法选项行: {option_line}")

    option_text = match.group(2)
    option_text = normalize_whitespace(option_text)

    if not option_text:
        raise ValueError(f"选项内容为空: {option_line}")

    return option_text

def parse_options(lines: List[str]) -> List[str]:
    """
    解析选项区域，支持选项跨行：
    - 遇到新的 A/B/C/D 行，开启新选项
    - 非选项前缀行，拼接到上一个选项后面
    """
    if not lines:
        raise ValueError("未找到任何选项行")

    options: List[str] = []
    current_option: str | None = None

    for line in lines:
        if OPTION_LINE_RE.match(line):
            if current_option is not None:
                options.append(normalize_whitespace(current_option))
            current_option = strip_option_prefix(line)
        else:
            if current_option is None:
                raise ValueError(f"选项区域出现无法归属的文本: {line}")
            current_option = f"{current_option} {line.strip()}"

    if current_option is not None:
        options.append(normalize_whitespace(current_option))

    if len(options) < 2:
        raise ValueError(f"选择题至少需要 2 个选项，当前仅解析到 {len(options)} 个")

    return options

def parse_question_block(block: str, block_index: int, source_file: str) -> ExamQuestion:
    """
    将单个题块解析为 ExamQuestion。
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        raise ValueError("空题块")

    question_id, question, question_type = parse_question_header(lines[0])

    question_lines, option_lines = split_question_and_option_lines(lines[1:])
    if question_lines:
        question = normalize_whitespace(" ".join([question] + question_lines))

    options = parse_options(option_lines)

    exam_question = ExamQuestion(
        question=question,
        options=options,
        question_type=question_type,
        question_id=question_id,
        metadata={
            "source": source_file,
            "block_index": block_index,
            "raw_block": block,
        },
    )
    exam_question.validate()
    return exam_question

def parse_input_text(text: str, source_name: str = "input.txt") -> List[ExamQuestion]:
    """
    直接从原始文本解析，方便后面做单元测试。
    """
    cleaned_text = preclean_text(text)
    blocks = split_question_blocks(cleaned_text)

    if not blocks:
        raise ValueError("未解析到任何题块，请检查 input.txt 格式")

    questions: List[ExamQuestion] = []

    for idx, block in enumerate(blocks, start=1):
        try:
            question = parse_question_block(
                block=block,
                block_index=idx,
                source_file=source_name,
            )
            questions.append(question)
        except Exception as exc:
            raise ValueError(
                f"解析第 {idx} 个题块失败: {exc}\n"
                f"原始题块如下:\n{block}"
            ) from exc

    return questions

def parse_input_file(file_path: str) -> List[ExamQuestion]:
    """
    对外主入口：
    input.txt -> List[ExamQuestion]
    """
    text = read_input_text(file_path)
    source_name = Path(file_path).name
    return parse_input_text(text, source_name=source_name)


if __name__ == "__main__":
    questions = parse_input_file("input.txt")
    print(f"共解析出 {len(questions)} 道题")

    for q in questions[:3]:
        print("-" * 60)
        print(f"question_id   : {q.question_id}")
        print(f"question_type : {q.question_type}")
        print(f"question      : {q.question}")
        print(f"options       : {q.options}")
