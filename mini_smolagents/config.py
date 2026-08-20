"""集中管理默认配置常量。"""

# 步数
DEFAULT_MAX_STEPS = 5
CODE_MAX_STEPS = 8
SUB_AGENT_MAX_STEPS = 5

# 上下文窗口 / 预算
DEFAULT_WINDOW_SIZE = 10
DEFAULT_TOKEN_BUDGET = 8000

# 执行 / 超时 / 重试
WEB_SEARCH_TIMEOUT = 20
BAIDU_PAGE_TIMEOUT = 6
PYTHON_INTERPRETER_TIMEOUT = 10
CODE_EXEC_TIMEOUT = 30
TOOL_RETRY_ATTEMPTS = 3
WEB_SEARCH_MAX_RESULTS = 10
SEARCH_CACHE_SIZE = 16
SEARCH_CACHE_TTL = 900
BAIDU_MIN_INTERVAL = 3.0
SEARCH_SEMANTIC_DUP_THRESHOLD = 0.76
SEARCH_STOP_WORDS = frozenset({
    # 中文高频修饰/虚词（2 字为主）
    "今日", "今天", "明天", "昨天", "昨日", "每天", "最新", "最新价",
    "价格", "股价", "股票", "行情", "走势", "数据", "信息", "内容",
    "结果", "情况", "多少", "怎么", "如何", "为何", "查询", "搜索",
    "详细", "全部", "包括", "时间", "地点", "城市", "推荐", "排行",
    "排名", "对比", "区别", "预测", "分析", "报告", "新闻", "热点",
    "动态", "快讯", "消息", "相关", "多少钱", "怎么样", "是什么", "有哪些",
    # 英文高频修饰
    "today", "yesterday", "latest", "price", "prices", "stock", "stocks",
    "news", "how", "much", "search", "result", "results", "data", "info",
})
BING_PAGE_TIMEOUT = 5
BING_MIN_INTERVAL = 1.0

# 记忆
MEMORY_TOP_K = 3
EXPERIENCE_TOP_K = 3
EXPERIENCE_MIN_COUNT = 2
MEMORY_DEDUP_THRESHOLD = 0.9
MEMORY_MATCH_THRESHOLD = 0.6
MEMORY_HALF_LIFE_DAYS = 30

# 档案卡（L2）
DEFAULT_PROFILE_MAX_BYTES = 4096
PROFILE_FILENAME = "user_profile.json"

# 档案卡 facts 向量召回（L2.5）
FACT_DEDUP_THRESHOLD = 0.9
FACTS_TOP_K = 3

# 截断长度
TRUNC_SHORT = 200
TRUNC_MEDIUM = 500
TRUNC_LONG = 800