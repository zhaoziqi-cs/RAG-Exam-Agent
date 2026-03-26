import json
import re
from typing import Callable, Dict, List, Optional

from src.augmentor import ExamAugmentor
from src.config import settings
from src.llm import call_llm
from src.retriever import VectorRetriever
from src.schemas import AugmentedContext, ExamQuestion, QAResult

class RAGQAPipeline:
    """
    组装完整 QA 流程：
    1. 接收结构化题目 ExamQuestion
    2. 调用 augmentor 构造检索 query
    3. 调用 retriever 检索知识库
    4. 调用 augmentor 构造最终 context
    5. 构建 prompt    调用 LLM
    6. 解析并规范化输出
    7. 返回 QAResult
    """
    def __init__(
            self,
            retriever: Optional[VectorRetriever] = None,
            augmentor: Optional[ExamAugmentor] = None,
            llm_fn: Optional[Callable[[str], str]] = None,
            top_k: int = settings.top_k,
    ) -> None:
        self.retriever = retriever or VectorRetriever()
        self.augmentor = augmentor or ExamAugmentor()
        self.llm_fn = llm_fn or call_llm
        self.top_k = top_k

    def answer(self,exam_question: ExamQuestion) -> QAResult:
        """
        主入口：输入一道题，输出结构化答题结果。
        """
        exam_question.validate()
        retrieval_query = self.augmentor.build_retrieval_query(exam_question)
        retrieved_chunks = self.retriever.retrieve(retrieval_query, top_k=self.top_k)
        augmented_context = self.augmentor.build_augmented_context(exam_question, retrieved_chunks)
        prompt = self.build_prompt(exam_question,augmented_context)

        raw_output = self.llm_fn(prompt)
        parsed_output = self.parse_llm_output(raw_output)
        normalized_output = self.normalize_answer(
            result = parsed_output,
            exam_question = exam_question,
            augmented_context = augmented_context
        )

        return QAResult(
            answer=normalized_output["answer"],
            reason=normalized_output["reason"],
            source=normalized_output["source"],
            retrieved_chunks=augmented_context.references,
            metadata={
                "question_id": exam_question.question_id,
                "question_type": exam_question.question_type,
                "retrieval_query": retrieval_query,
                "raw_llm_output": raw_output,
            },
        )
    
    def answer_from_raw(
            self,
        question: str,
        options: List[str],
        question_type: str = "single",
        question_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> QAResult:
        """
        方便外部直接传原始参数，不用手动先构造 ExamQuestion。
        """
        exam_question = ExamQuestion(
            question=question,
            options=options,
            question_type=question_type,
            question_id=question_id,
            metadata=metadata or {},
        )
        return self.answer(exam_question)
    
    def build_prompt(
            self,
            exam_question: ExamQuestion,
            augmented_context: AugmentedContext,
    ) -> str:
        """
        把题目 + 选项 + 增强后的上下文，拼成最终给 LLM 的 prompt。
        """
        options_text = "\n".join(f"{chr(ord('A')+i)}. {option}"
                                for i, option in enumerate(exam_question.options)
        )
        question_type_text = (
            "单选题（只能选择一个答案）"
            if exam_question.question_type == "single"
            else "多选题（可以选择多个答案）"
        )
        return f"""你是一个企业内部考试助手，请严格依据“知识库参考”回答问题。

【任务要求】
1. 必须基于知识库参考作答，不要凭空补充制度细节。
2. 单选题只能返回一个选项；多选题可以返回多个选项。
3. 如果知识库证据不足，也要基于最相关内容给出最合理答案。
4. 输出必须是严格 JSON，不能附加任何解释性文字、代码块标记或前后缀。

【题目类型】
{question_type_text}

【题目】
{exam_question.question}

【选项】
{options_text}

【知识库参考】
{augmented_context.context}

请返回如下 JSON 格式：
{{
  "answer": ["A"],
  "reason": "简要解释你为什么选择这些答案，说明依据了哪些知识点。",
  "source": "来源文件名 + 页码（如果能判断）"
}}
"""
        
    @staticmethod
    def parse_llm_output(output: str) -> Dict:
        """
        优先直接解析 JSON；
        如果模型返回了额外文字，则尝试提取其中的 JSON 对象。
        """
        if not output or not output.strip():
            return {
                "answer": [],
                "reason": "模型未返回有效内容。",
                "source": "",
            }
        text = output.strip()

        # 1. 直接解析
        try :
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 去掉 ```json ... ``` 代码块包装后再试
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
         # 3. 从整段文本里提取最外层 JSON 对象
        json_candidate = RAGQAPipeline._extract_json_object(text)
        if json_candidate:
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

        return {
            "answer": [],
            "reason": "模型输出解析失败。",
            "source": "",
        }

        
    def normalize_answer(
            self,
            result: Dict,
            exam_question: ExamQuestion,
            augmented_context: AugmentedContext,
    ) -> Dict[str,object]:
        """
        规范化模型输出：
        - answer 强制转成大写字母列表
        - 过滤非法选项
        - 单选题只保留一个答案
        - source 为空时做兜底
        """
        valid_labels = {chr(ord("A") + i) for i in range(len(exam_question.options))}
        raw_answer = result.get("answer",[])
        answers = self._normalize_answer_list(raw_answer)

        # 只保留合法选项
        answers = [a for a in answers if a in valid_labels]

        # 单选题只保留一个
        if exam_question.question_type == "single" and len(answers) > 1:
            answers = answers[:1]
        reason = str(result.get("reason", "") or "").strip()
        if not reason:
            reason = "模型未给出有效解析。"
        source = str(result.get("source", "") or "").strip()
        if not source:
            source = self._build_fallback_source(augmented_context)

        return {
            "answer": answers,
            "reason": reason,
            "source": source,
        }
        
    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """
        从混杂文本中提取第一个完整 JSON 对象。
        用简单括号栈处理，比纯正则更稳一些。
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None
        
    @staticmethod
    def _normalize_answer_list(raw_answer) -> List[str]:
        """
        兼容以下几种情况：
        - ["A", "C"]
        - "A"
        - "A,C"
        - "答案是 A 和 C"
        """
        if not raw_answer:
            return []
        items = raw_answer if isinstance(raw_answer,list) else [raw_answer]

        normalized: List[str] = []
        seen = set()
        for item in items:
            text = str(item).upper()
            letters = re.findall(r"[A-Z]", text)
            for letter in letters:
                if letter not in seen:
                    seen.add(letter)
                    normalized.append(letter)
        return normalized
       
    @staticmethod
    def _build_fallback_source(augmented_context: AugmentedContext) -> str:
        """
        当模型没返回 source 时，用 references 第一条做兜底。
        """
        if not augmented_context.references:
            return ""
        
        first_ref = augmented_context.references[0]
        source = first_ref.source or "unknown"
        page = first_ref.metadata.get("page")

        if page is not None:
            return f"{source} 第 {page} 页"
        return source