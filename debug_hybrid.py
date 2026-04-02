from src.retriever import VectorRetriever
from src.hybrid_retriever import HybridRetriever
from src.augmentor import ExamAugmentor
from src.schemas import ExamQuestion

augmentor = ExamAugmentor()
vector_ret = VectorRetriever()
hybrid_ret = HybridRetriever()

question11 = ExamQuestion(
        question='资产分类中，逾期1年以上的应收账款属于哪类资产？',
        options=[
            "A、良性资产",
            "B、一般资产",
            "C、低效资产",
            "D、无效资产"
        ],
        question_type="single",
        question_id="demo_011"
    )  # 你的 Q11 或 Q25 ExamQuestion 对象

question15 = ExamQuestion(
        question='以下哪项属于"1245"工作思路中的"4个抓手"？',
        options=[
            "A、党建引领",
            "B、现金流管理",
            "C、数智赋能",
            "D、风险防控"
        ],
        question_type="single",
        question_id="demo_015"
    )  # 你的 Q11 或 Q25 ExamQuestion 对象

question25 = ExamQuestion(
        question='资以下哪些属于资产分类中的"良性资产"？',
        options=[
            "A、可自由支取的货币资金",
            "B、正常收款的应收账款",
            "C、未逾期的其他应收款",
            "D、正常使用的固定资产"
        ],
        question_type="multiple",
        question_id="demo_025"
    )  # 你的 Q11 或 Q25 ExamQuestion 对象


query = augmentor.build_retrieval_query(question15)

for name, ret in [("vector", vector_ret), ("hybrid", hybrid_ret)]:
    print(f"\n===== {name} =====")
    results = ret.retrieve(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(i, r.source, r.metadata.get("page"), r.chunk_id, r.score)
        print(r.text[:200])
        print("-" * 60)