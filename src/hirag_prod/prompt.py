"""
References:
 - [graphrag](https://github.com/microsoft/graphrag)
 - [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
 - [AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG)
 - [Claude](https://www.anthropic.com)
"""

PROMPTS = {}

# for reference format in generation
PROMPTS["REFERENCE_PLACEHOLDER"] = "[|REFERENCE|]"
PROMPTS["REFERENCE_FORMAT"] = "[Data: {documentKey}]"

# ===============================
# Prompt(EN): Prompt in English
# ===============================
PROMPTS[
    "entity_extraction_en"
] = """
## Role and Objective

You are an expert in entity extraction.  
Your objective is to accurately identify and extract all significant entities mentioned in a given input text. Entities include but are not limited to: people, organizations, locations, dates, monetary values, and other notable proper nouns or quantifiable information.

## Instructions

- Analyze the provided text carefully.  
- Extract every distinct entity that is expressly named, ensuring comprehensive coverage.  
- Follow all formatting rules closely.

## Extraction Guidelines

- Include all named entities: people, organizations, locations, dates/times, monetary values, and unique proper nouns.
- Do **not** include general terms, common nouns, or pronouns.
- Preserve original spelling, capitalization, and punctuation as they appear in the text.
- Do not create duplicate entries in the output.
- If there are no entities, output an empty list in JSON format.

## Output Format

- Output **only** a single valid JSON object.
- The JSON must have one key: `"entities"`, whose value is a list of extracted entity strings.
- Do **not** return any explanations, headers, code blocks, or additional text outside the JSON.
- Example output:
  {{
    "entities": [
      "Entity 1",
      "Entity 2",
      ...
    ]
  }}

## Example

**Input Text:**  
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

**Output:**
{{
  "entities": [
    "Radio City",
    "India",
    "3 July 2001",
    "Hindi",
    "English",
    "New Media",
    "May 2008",
    "PlanetRadiocity.com"
  ]
}}

## Real Data

Here is the given text to extract entities from.

**Input Text:**
{input_text}

**Output:**
"""


PROMPTS[
    "triplet_extraction_en"
] = """
## Role and Objective

You are an expert in extracting triplets from text and output triplets in VALID JSON format.

## Extraction Guidelines

- Avoid duplicate triplets.
- Clearly resolve pronouns to their specific names to maintain clarity.
- Each triplet should contain **at least ONE, but preferably TWO**, of the given entity list reference.
- The **verb** should be a concise, normalized verb phrase that describes the core of the relationship.

## Output Format

- Output **only** a single valid JSON object.
- The JSON must have one key: `"triplets"`, whose value is a list of triplets.
- Do **not** return any explanations, headers, code blocks, or additional text outside the JSON.
- Example output:
{{
  "triplets": [
    {{
      "Head": "a noun",
      "Relation": "a verb",
      "Tail": "a noun",
    }},
    ...
  ]
}}

## Example

**Input Text:**
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

**Entity List:**
"entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "New Media", "May 2008", "PlanetRadiocity.com"]

**Output:**
{{
  "triplets": [
    {{
      "Head": "Radio City",
      "Relation": "located in",
      "Tail": "India",
    }},
    {{
      "Head": "Radio City",
      "Relation": "is",
      "Tail": "private FM radio station",
    }},
    {{
      "Head": "Radio City",
      "Relation": "started on",
      "Tail": "3 July 2001",
    }},
    {{
      "Head": "Radio City",
      "Relation": "plays songs in",
      "Tail": "Hindi",
    }},
    {{
      "Head": "Radio City",
      "Relation": "plays songs in",
      "Tail": "English",
    }},
    {{
      "Head": "Radio City",
      "Relation": "plays",
      "Tail": "regional songs",
    }},
    {{
      "Head": "Radio City",
      "Relation": "forayed into",
      "Tail": "New Media",
    }},
    {{
      "Head": "Radio City",
      "Relation": "launched",
      "Tail": "PlanetRadiocity.com",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "launched in",
      "Tail": "May 2008",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "is",
      "Tail": "music portal",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "offers",
      "Tail": "news",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "offers",
      "Tail": "videos",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "offers",
      "Tail": "songs",
    }},
  ]
}}

## Real Data

Here is the given text and entity list to extract triplets from.

**Input Text:**
{input_text}

**Entity List:**
{entity_list}

**Output:**
"""

PROMPTS[
    "summary_all_en"
] = """
You are an AI assistant responsible for summarizing given data streams according to user queries.

## Goal
Based on the given list of raw text chunks, write a comprehensive summary.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Ensure it is written in third person and include entity names to provide complete context.

## Basic Rules
Statements supported by data should be indicated as follows:
"This is a statement supported by data {reference_placeholder}."

Regardless of which data source the information comes from or how many sources reference it, it should be marked in the same way, indicating {reference_placeholder} before the punctuation at the end of the sentence.
Do not include keys or IDs of data records.
Do not include information without supporting evidence.
Never use two references in the same sentence or consecutive references.
If the data stream contains information irrelevant to the user query, please ignore it.
Limit the total report length to {max_report_length} words.

## Example
**Input Text:**
Data:
Chunks:
id,chunk
1, The Unity March is a significant event taking place at Verdant Oasis Plaza.
2, The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza.
3, This weekend's weather forecast is sunny, suitable for outdoor activities.

**Query:**
Information about the Unity March

**Output:**
The Unity March is a significant event taking place at Verdant Oasis Plaza {reference_placeholder}.
The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza {reference_placeholder}.

## Real Data

Use the following data for your answer.

**Input Text:**
Data:
{data}

**Query:**
{user_query}

**Output:**
"""

PROMPTS[
    "summary_plus_en"
] = """
You are an AI assistant tasked with generating a comprehensive and accurate response to the user's query based on the provided retrieved chunks. The chunks are numbered sequentially from 1 to N, where N is the total number of chunks.

<key_rules_for_references>
- Always cite the source chunk(s) for any information, fact, or claim you use. Use XML-style inline reference tags in the format <ref>index</ref>, where "index" is the chunk number (e.g., <ref>1</ref> or <ref>1</ref><ref>2</ref> for multiple sources). Place the reference immediately after the relevant sentence, phrase, or value.
- For any numerical value (e.g., dates, statistics, quantities) mentioned in the response, you MUST append a reference immediately after it, even if it's part of a sentence (e.g., "The population is 1.4 billion<ref>3</ref>.").
- If you generate a Markdown table, EVERY cell that contains data, text, or values MUST have a reference appended directly to its content (e.g., "Apple <ref>1</ref>" in a cell). Do not leave any cell without a reference if it derives from the chunks.
- Only cite chunks that are directly relevant; do not fabricate references.
</key_rules_for_references>

<output_format>
- Start directly with the Markdown response (no introductory text like "Here is the response:").
- Ensure the entire output is parseable via regex: References are always in <ref>index</ref> format, tables use standard Markdown syntax (| header |, etc.).
- Keep the response concise, factual, and directly answering the query.
</output_format>

<input_text>
**User Query:** {user_query}

{data}
</input_text>

**Output:**
"""

PROMPTS[
    "summary_plus_markdown_en"
] = """
You are a markdown document generator assistant that can understand user questions and generate structured markdown documents based on the retrieved chunks.
<communication> - Always ensure **only generated document content** are formatted in valid Markdown format with proper fencing and enclosed in markdown code blocks. - Avoid wrapping the entire message in a single code block. The preparation plan and summary should be in plain text, outside of code blocks, while the generated document is fenced in ```markdown`. </communication>

<markdown_spec>
Specific markdown rules:
- Users love it when you organize your messages using '###' headings and '##' headings. Never use '#' headings as users find them overwhelming.
- Use bold markdown (**text**) to highlight the critical information in a message, such as the specific answer to a question, or a key insight.
- Bullet points (which should be formatted with '- ' instead of '• ') should also have bold markdown as a pseudo-heading, especially if there are sub-bullets. Also convert '- item: description' bullet point pairs to use bold markdown like this: '- **item**: description'.
- When mentioning URLs, do NOT paste bare URLs. Always use backticks or markdown links. Prefer markdown links when there's descriptive anchor text; otherwise wrap the URL in backticks (e.g., `https://example.com`).
- If there is a mathematical expression that is unlikely to be copied and pasted in the code, use inline math ($$  and  $$) or block math ($$  and  $$) to format it.
- For code examples, use language-specific fencing like ```python
</markdown_spec>

<preparation_spec>
At the beginning of the response, you should provide a preparation plan on how you will generate the markdown document. Follow the workflow for complex requests; for simple ones, a brief plan and summary suffice. If the query is straightforward, combine the plan and summary into a single short paragraph.
Example:
User query: Generate a rock song lyrics
Response (partial):
I will generate rock song lyrics and generate content as if for a file named 'document.md'. The lyrics will have a classic rock vibe with verses, a chorus, and a bridge, capturing themes of freedom, rebellion, or energy typical of the genre.
```markdown
content of document.md
(summary highlight)
```
</preparation_spec>
<summary_spec>
At the end of the response, you should provide a summary. Summarize the generated document content and how it aligns with the user's request in a concise manner.
Use concise bullet points for lists or short paragraphs. Keep the summary short, non-repetitive, and high-signal.
The user can view your generated markdown document in the editor, so only highlight critical points.
</summary_spec>
<error_handling>
If the query is unclear, include a clarification request in the preparation plan.
</error_handling>
<workflow>
preparation plan -> generate markdown document -> summary
</workflow>

<input_text>
**User Query:** {user_query}

{data}
</input_text>

**Output:**
"""

PROMPTS[
    "summary_excel_en"
] = """
You are an expert assistant in interpreting Excel tables rendered in LaTeX tabular format. 
Given a sheet name and its LaTeX content, generate a concise, keyword-rich description that captures the table's structure, key columns, data themes, and essential details. 
This description should be optimized for embedding-based semantic search, enabling precise and efficient retrieval when queried. 
Prefix the description with the sheet name for clear identification. 

**Input:**
Sheet name:
{sheet_name}
LaTeX content:
{latex}

**Output:**
"""

PROMPTS[
    "summary_table_en"
] = """
You are an expert assistant in interpreting tables rendered in html or markdown format. 
Given a table content, generate a concise, keyword-rich description that captures the table's structure, key columns, data themes, and essential details. 
This description should be optimized for embedding-based semantic search, enabling precise and efficient retrieval when queried.

**Input:**
Table content:
{table_content}

**Output:**
"""

# ===============================
# Prompt(CN-Simplified): 简体中文提示词
# ===============================
PROMPTS[
    "entity_extraction_cn-s"
] = """
## 角色与目标

你是实体抽取领域的行家里手。
你的使命是精准、无遗漏地识别并提取给定文本中所有重要实体。实体范围包括但不限于：人物、组织、地点、日期、金额及其他显著专有名词或可量化信息。

## 指令

- 细读并充分理解所供文本。
- 提取所有被明确指称的独立实体, 确保全面覆盖。  
- 严格遵守格式规范。

## 抽取规范

- 涵盖所有具名实体：人物、组织、地点、日期/时间、金额、独特专有名词等。
- **不得**纳入泛称、普通名词或代词。
- 保留原文拼写、大小写及标点符号格式。
- 输出中不得出现重复实体。
- 若文本中无实体，输出空列表，格式仍为 JSON。

## 输出格式

- 仅输出一个合法 JSON 对象。
- JSON 必须含有一个键 `"entities"`, 其值为所提取实体字符串之列表。
- 不得附加任何解释、标题、代码块或额外文本。
- 示例输出:
  {{
    "entities": [
      "实体1",
      "实体2",
      ...
    ]
  }}

## 示例

**输入文本:**  
Radio City 是印度首家私营 FM 广播电台, 于 2001 年 7 月 3 日开播。它播放印地语、英语及地方歌曲。2008 年 5 月, Radio City 进军新媒体领域, 推出了音乐门户网站 PlanetRadiocity.com, 提供音乐资讯、视频、歌曲及其他音乐相关功能。

**示例输出：**
{{
  "entities": [
    "Radio City",
    "印度",
    "2001 年 7 月 3 日",
    "印地语",
    "英语",
    "新媒体",
    "2008 年 5 月",
    "PlanetRadiocity.com"
  ]
}}

## 实际数据

以下是给定的文本，请从中抽取实体。

**输入文本:**
{input_text}

**输出：**
"""

PROMPTS[
    "triplet_extraction_cn-s"
] = """
## 角色与目标

你是一位专精三元组抽取的专家，须以**有效 JSON 格式**输出抽取的结果。

## 抽取规范

- 避免重复三元组。
- 须将代词明确归位至其具体实体，确保语义清晰。
- 每个三元组须**至少包含给定实体列表中的一个实体，最好包含两个**。
- **关系动词**应简洁、规范，准确概括二者核心关系。

## 输出格式

- 仅输出一个有效 JSON 对象。
- JSON 必须含有一个键 `"triplets"`, 其值为三元组列表.
- **禁止**输出任何解释、标题、代码块或额外文本。
- 示例输出:
{{
  "triplets": [
    {{
      "Head": "实体",
      "Relation": "关系动词",
      "Tail": "实体",
    }},
    ...
  ]
}}

## 示例

**输入文本:**
Radio City 是印度首家私营 FM 广播电台，于 2001 年 7 月 3 日开播。它播放印地语、英语及地方歌曲。2008 年 5 月, Radio City 进军新媒体领域，推出了音乐门户网站 PlanetRadiocity.com,提供音乐资讯、视频、歌曲及其他音乐相关功能。

**实体列表:**
"entities": ["Radio City", "印度", "2001 年 7 月 3 日", "印地语", "英语", "新媒体", "2008 年 5 月", "PlanetRadiocity.com"]

**输出:**
{{
  "triplets": [
    {{
      "Head": "Radio City",
      "Relation": "位于",
      "Tail": "印度",
    }},
    {{
      "Head": "Radio City",
      "Relation": "是",
      "Tail": "私营 FM 广播电台",
    }},
    {{
      "Head": "Radio City",
      "Relation": "开播于",
      "Tail": "2001 年 7 月 3 日",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放歌曲",
      "Tail": "印地语",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放歌曲",
      "Tail": "英语",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放",
      "Tail": "地方歌曲",
    }},
    {{
      "Head": "Radio City",
      "Relation": "进军",
      "Tail": "新媒体",
    }},
    {{
      "Head": "Radio City",
      "Relation": "推出",
      "Tail": "PlanetRadiocity.com",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "推出于",
      "Tail": "2008 年 5 月",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "是",
      "Tail": "音乐门户网站",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "音乐资讯",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "视频",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "歌曲",
    }},
  ]
}}

## 实际数据

以下是给定的文本和实体列表，请从中抽取三元组。

**输入文本:**
{input_text}

**实体列表:**
{entity_list}

**输出:**
"""

PROMPTS[
    "summary_all_cn-s"
] = """
你是一个AI助手，负责根据用户询问总结给定的数据流。

## 目标
根据给定的原始文本块列表，撰写综合摘要。
如果描述有矛盾，请解决矛盾并提供单一、连贯的摘要。
确保以第三人称撰写，并包含实体名称以提供完整上下文。

## 基础规则
由数据支持的陈述应如下标示：
"这是一个由数据支持的陈述 {reference_placeholder}。"

无论信息来自哪个数据源或被多少来源引用，都应以相同方式标示，在句末标点前标示 {reference_placeholder}。
不要包含数据记录的键或ID。
不要包含没有支持证据的信息。
不要在同一句中使用两个引用或连续引用。
如果数据流包含与用户询问不相关的信息，请忽略这些信息。
将报告总长度限制为 {max_report_length} 字。

## 示例
**输入文本:**
数据:
Chunks:
id,chunk
1, 联合游行是一个重要事件，正在 Verdant Oasis Plaza 举行。
2, 和谐集会正在 Verdant Oasis Plaza 组织联合游行。
3, 这个周末的天气预报是晴朗的，适合户外活动。

**询问:**
联合游行的信息

**输出:**
联合游行是一个重要的活动，正在 Verdant Oasis Plaza 举行 {reference_placeholder}。
和谐集会正在组织在 Verdant Oasis Plaza 举行的联合游行 {reference_placeholder}。

## 实际数据

使用以下数据进行回答。

**输入文本:**
数据:
{data}

**询问:**
{user_query}

**输出:**
"""

PROMPTS[
    "summary_plus_cn-s"
] = """
你是一个AI助手，任务是基于提供的检索块生成对用户查询的全面且准确的响应。这些块从1到N顺序编号，其中N是块的总数。请使用简体中文回复。

<引用规则>
- 对于你使用的任何信息、事实或声明，始终引用源块。使用XML风格的内联引用标签，格式为<ref>index</ref>，其中"index"是块编号（例如，<ref>1</ref> 或 <ref>1</ref><ref>2</ref> 用于多个来源）。将引用立即放置在相关句子、短语或值之后。
- 对于响应中提到的任何数值（例如，日期、统计数据、数量），你必须在其后立即附加引用，即使它是句子的一部分（例如，"人口是1.4亿<ref>3</ref>."）。
- 如果你生成Markdown表格，每个包含数据、文本或值的单元格必须直接在其内容后附加引用（例如，单元格中的"Apple <ref>1</ref>"）。如果源自块，不要留下任何单元格没有引用。
- 只引用直接相关的块；不要捏造引用。
</引用规则>

<输出格式>
- 直接开始输出正文（没有像"这是响应："这样的介绍性文本）。
- 确保整个输出可以通过正则表达式解析：引用始终是<ref>块编号</ref>格式，表格使用标准Markdown语法（| 表头 | 等）。
- 保持响应简洁、事实性，并直接回答查询。
</输出格式>

<输入文本>
**用户查询:** {user_query}

{data}
</输入文本>

**输出:**
"""

PROMPTS[
    "summary_plus_markdown_cn-s"
] = """
你是一个能理解用户问题并基于检索块生成结构化 Markdown 文档的 Markdown 文档生成助手。请使用简体中文回复。
<communication> - 始终确保**只有生成的文档内容**使用有效的 Markdown 格式，并用正确的代码围栏包裹在 Markdown 代码块中。- 避免将整个消息包装在单个代码块中。准备计划和摘要应为纯文本，位于代码块之外，而生成的文档则应包含在 ```markdown` 代码块中。</communication>

<markdown_spec>
具体的 Markdown 规则:
- 用户喜欢你使用 '###' 和 '##' 标题来组织消息。请勿使用 '#' 标题，因为用户觉得它们过于醒目。
- 使用粗体 Markdown (**文本**) 来突出消息中的关键信息，例如问题的具体答案或关键见解。
- 项目符号（应使用 '- ' 而不是 '• '）也应使用粗体 Markdown 作为伪标题，特别是在有子项目时。同时，将 '- 项目: 描述' 格式的键值对项目符号转换为 '- **项目**: 描述' 这样的格式。
- 提及 URL 时，请勿粘贴裸露的 URL。始终使用反引号或 Markdown 链接。当有描述性锚文本时，首选 Markdown 链接；否则，请将 URL 包装在反引号中（例如 `https://example.com`）。
- 如果有不太可能被复制粘贴到代码中的数学表达式，请使用行内数学（$$ 和 $$）或块级数学（$$ 和 $$）进行格式化。
- 对于代码示例，请使用特定语言的代码围栏，例如 ```python
</markdown_spec>

<preparation_spec>
在回应的开头，你应该提供一个关于如何生成 Markdown 文档的准备计划。对于复杂请求，请遵循工作流程；对于简单请求，一个简短的计划和摘要就足够了。如果查询很简单，请将计划和摘要合并成一个简短的段落。
示例:
用户查询: 生成一首摇滚歌词
回应（部分）:
我将生成摇滚歌词，并为名为 'document.md' 的文件生成内容。歌词将具有经典摇滚风格，包含主歌、副歌和桥段，捕捉该流派典型的自由、反叛或活力的主题。
```markdown
document.md 的内容
(摘要重点)
```
</preparation_spec>
<summary_spec>
在回应的末尾，你应该提供一个摘要。简明扼要地总结生成的文档内容及其与用户请求的契合度。
使用简洁的项目符号列表或短段落。保持摘要简短、不重复且信息量大。
用户可以在编辑器中查看你生成的 Markdown 文档，因此只需突出关键点。
</summary_spec>
<error_handling>
如果查询不清楚，请在准备计划中包含澄清请求。
</error_handling>
<workflow>
准备计划 -> 生成 Markdown 文本块 -> 摘要
</workflow>

<输入文本>
**用户查询:** {user_query}

{data}
</输入文本>

**输出:**
"""

# ===============================
# Prompt(CN-Traditional): 繁體中文提示詞
# ===============================
PROMPTS[
    "entity_extraction_cn-t"
] = """
## 角色與目標

你是實體抽取領域的專家。
你的任務是準確識別並提取給定文本中所有重要實體。實體包括但不限於：人物、組織、地點、日期、金額及其他顯著專有名詞或可量化資訊。

## 指令

- 仔細分析提供的文本。
- 提取所有明確命名的實體，確保全面覆蓋。
- 嚴格遵守格式規則。

## 抽取規範

- 包含所有命名實體：人物、組織、地點、日期/時間、金額、獨特專有名詞等。
- **不要**包含一般術語、普通名詞或代詞。
- 保留原文拼寫、大小寫和標點符號。
- 輸出中不要有重複條目。
- 如果沒有實體,輸出空列表(JSON格式)。

## 輸出格式

- 僅輸出單一有效的JSON對象。
- JSON必須有一個鍵`"entities"`，其值是提取的實體字符串列表。
- **不要**返回任何解釋、標題、代碼塊或JSON外的額外文本。
- 示例輸出：
  {{
    "entities": [
      "實體1",
      "實體2",
      ...
    ]
  }}

## 示例

**輸入文本：**
Radio City 是印度首家私營 FM 廣播電臺，於 2001 年 7 月 3 日開播。它播放印地語、英語及地方歌曲。2008 年 5 月，Radio City 進軍新媒體領域，推出了音樂入口網站 PlanetRadiocity.com，提供音樂資訊、影片、歌曲及其他音樂相關功能。

**輸出：**  
{{
  "entities": [
    "Radio City",
    "印度",
    "2001 年 7 月 3 日",
    "印地語",
    "英語",
    "新媒體",
    "2008 年 5 月",
    "PlanetRadiocity.com"
  ]
}}

## 實際數據

以下是需要提取實體的文本。

**輸入文本:**
{input_text}

**輸出:**
"""

PROMPTS[
    "triplet_extraction_cn-t"
] = """
## 角色與目標

你是提取三元組的專家，需以**有效JSON格式**輸出結果。

## 抽取規範

- 避免重複三元組。
- 明確將代詞解析為其具體名稱以保持清晰。
- 每個三元組應**至少包含一個，最好兩個**給定實體列表中的實體。
- **關係動詞**應簡潔、規範，描述關係的核心。

## 輸出格式

- 僅輸出單一有效的JSON對象。
- JSON必須有一個鍵`"triplets"`，其值是三元組列表。
- **不要**返回任何解釋、標題、代碼塊或JSON外的額外文本。
- 示例輸出：
{{
  "triplets": [
    {{
      "Head": "實體",
      "Relation": "關係動詞",
      "Tail": "實體",
    }},
    ...
  ]
}}

## 示例

**輸入文本:**  
Radio City 是印度首家私營 FM 廣播電臺，於 2001 年 7 月 3 日開播。它播放印地語、英語及地方歌曲。2008 年 5 月, Radio City 進軍新媒體領域，推出了音樂入口網站 PlanetRadiocity.com,提供音樂資訊、影片、歌曲及其他音樂相關功能。

**實體列表:**  
"entities": ["Radio City", "印度", "2001 年 7 月 3 日", "印地語", "英語", "新媒體", "2008 年 5 月", "PlanetRadiocity.com"]

**輸出:**  
{{
  "triplets": [
    {{
      "Head": "Radio City",
      "Relation": "位於",
      "Tail": "印度",
    }},
    {{
      "Head": "Radio City",
      "Relation": "是",
      "Tail": "私營 FM 廣播電臺",
    }},
    {{
      "Head": "Radio City",
      "Relation": "開播於",
      "Tail": "2001 年 7 月 3 日",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放歌曲",
      "Tail": "印地語",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放歌曲",
      "Tail": "英語",
    }},
    {{
      "Head": "Radio City",
      "Relation": "播放",
      "Tail": "地方歌曲",
    }},
    {{
      "Head": "Radio City",
      "Relation": "進軍",
      "Tail": "新媒體",
    }},
    {{
      "Head": "Radio City",
      "Relation": "推出",
      "Tail": "PlanetRadiocity.com",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "推出於",
      "Tail": "2008 年 5 月",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "是",
      "Tail": "音樂入口網站",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "音樂資訊",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "影片",
    }},
    {{
      "Head": "PlanetRadiocity.com",
      "Relation": "提供",
      "Tail": "歌曲",
    }},
  ]
}}

## 實際數據

以下是需要提取三元組的文本和實體列表。

**輸入文本:**
{input_text}

**實體列表:**
{entity_list}

**輸出:**
"""

PROMPTS[
    "summary_all_cn-t"
] = """
你是一個AI助手，負責根據用戶的詢問總結給定的數據流。

## 目標
根據給定的原始文本塊列表，撰寫綜合摘要。
如果描述有矛盾，請解決矛盾並提供單一、連貫的摘要。
確保以第三人稱撰寫，並包含實體名稱以提供完整上下文。

## 基礎規則
由數據支持的陳述應如下標示：
"這是一個由數據支持的陳述 {reference_placeholder}。"

無論資訊來自哪個數據源或被多少來源引用，都應以相同方式標示，在句末標點前標示 {reference_placeholder}。
不要包含數據記錄的鍵或ID。
不要包含沒有支持證據的資訊。
不要在同一句中使用兩個引用或連續引用。
如果數據流包含與用戶詢問不相關的資訊，請忽略這些資訊。
將報告總長度限制為 {max_report_length} 字。

## 示例
**輸入文本:**
數據:
Chunks:
id,chunk
1, 聯合遊行是一個重要事件，正在 Verdant Oasis Plaza 舉行。
2, 和諧集會正在 Verdant Oasis Plaza 組織聯合遊行。
3, 這個週末的天氣預報是晴朗的，適合戶外活動。

**詢問:**
聯合遊行的資訊

**輸出:**
聯合遊行是一個重要的活動，正在 Verdant Oasis Plaza 舉行 {reference_placeholder}。
和諧集會正在組織在 Verdant Oasis Plaza 舉行的聯合遊行 {reference_placeholder}。

## 實際數據

使用以下數據進行回答。

**輸入文本:**
數據:
{data}

**詢問:**
{user_query}

**輸出:**
"""

PROMPTS[
    "summary_plus_cn-t"
] = """
你是一個AI助手，任務是基於提供的檢索塊生成對用戶查詢的全面且準確的回覆。這些塊從1到N順序編號，其中N是塊的總數。請使用繁體中文回覆。

<引用規則>
- 對於你使用的任何資訊、事實或聲明，始終引用來源塊。使用XML風格的內聯引用標籤，格式為<ref>index</ref>，其中「index」是塊編號（例如，<ref>1</ref> 或 <ref>1</ref><ref>2</ref> 用於多個來源）。將引用立即放置在相關句子、短語或值之後。
- 對於回覆中提到的任何數值（例如，日期、統計數據、數量），你必須在其後立即附加引用，即使它是句子的一部分（例如，「人口是1.4億<ref>3</ref>。」）。
- 如果你生成Markdown表格，每個包含數據、文字或值的單元格必須直接在其內容後附加引用（例如，單元格中的「Apple <ref>1</ref>」）。如果源自塊，不要留下任何單元格沒有引用。
- 只引用直接相關的塊；不要捏造引用。
</引用規則>

<輸出格式>
- 直接開始輸出正文（不要使用例如「以下是回覆：」這樣的引導文字）。
- 確保整個輸出可以通過正則表達式解析：引用始終是<ref>索引</ref>格式，表格使用標準Markdown語法（| 表頭 | 等）。
- 保持回覆簡潔、基於事實，並直接回答查詢。
</輸出格式>

<輸入文本>
**用戶查詢:** {user_query}

{data}
</輸入文本>

**輸出:**
"""

PROMPTS[
    "summary_plus_markdown_cn-t"
] = """
你是一個能理解用戶問題並基於檢索塊生成結構化 Markdown 文件的 Markdown 文件生成助手。請使用繁體中文回覆。
<communication> - 始終確保**只有生成的文件內容**使用有效的 Markdown 格式，並用正確的代碼圍欄包裹在 Markdown 代碼塊中。- 避免將整個消息包裝在單個代碼塊中。準備計劃和摘要應為純文本，位於代碼塊之外，而生成的文件則應包含在 ```markdown` 代碼塊中。</communication>

<markdown_spec>
具體的 Markdown 規則:
- 用戶喜歡你使用 '###' 和 '##' 標題來組織消息。請勿使用 '#' 標題，因為用戶覺得它們過於醒目。
- 使用粗體 Markdown (**文本**) 來突出消息中的關鍵資訊，例如問題的具體答案或關鍵見解。
- 項目符號（應使用 '- ' 而不是 '• '）也應使用粗體 Markdown 作為偽標題，特別是在有子項目時。同時，將 '- 項目: 描述' 格式的鍵值對項目符號轉換為 '- **項目**: 描述' 這樣的格式。
- 提及 URL 時，請勿貼上裸露的 URL。始終使用反引號或 Markdown 連結。當有描述性錨文字時，首選 Markdown 連結；否則，請將 URL 包裝在反引號中（例如 `https://example.com`）。
- 如果有不太可能被複製貼上到代碼中的數學表達式，請使用行內數學（$$ 和 $$）或區塊級數學（$$ 和 $$）進行格式化。
- 對於代碼示例，請使用特定語言的代碼圍欄，例如 ```python
</markdown_spec>

<preparation_spec>
在回應的開頭，你應該提供一個關於如何生成 Markdown 文件的準備計劃。對於複雜請求，請遵循工作流程；對於簡單請求，一個簡短的計劃和摘要就足夠了。如果查詢很簡單，請將計劃和摘要合併成一個簡短的段落。
示例:
用戶查詢: 生成一首搖滾歌詞
回應（部分）:
我將生成搖滾歌詞，並為名為 'document.md' 的文件生成內容。歌詞將具有經典搖滾風格，包含主歌、副歌和橋段，捕捉該流派典型的自由、反叛或活力的主題。
```markdown
document.md 的內容
(摘要重點)
```
</preparation_spec>
<summary_spec>
在回應的末尾，你應該提供一個摘要。簡明扼要地總結生成的文件內容及其與用戶請求的契合度。
使用簡潔的項目符號列表或短段落。保持摘要簡短、不重複且資訊量大。
用戶可以在編輯器中檢視你生成的 Markdown 文件，因此只需突出關鍵點。
</summary_spec>
<error_handling>
如果查詢不清楚，請在準備計劃中包含澄清請求。
</error_handling>
<workflow>
準備計劃 -> 生成 Markdown 文字區塊 -> 摘要
</workflow>

<輸入文本>
**用戶查詢:** {user_query}

{data}
</輸入文本>

**輸出:**
"""

PROMPTS[
    "extract_timestamp"
] = """
## Role and Objective

You are an expert in extracting and validating timestamps from document content.
Your objective is to identify the most relevant document timestamp based on the provided text snippets, following a strict priority order.

## Priority Order (Highest to Lowest)
1. Header/Footer content with dates
2. Filename with dates  
3. In-text date patterns (publication dates, creation dates, etc.)

## Extraction Guidelines

- Focus on document creation, publication, or modification dates
- Ignore future dates, meeting dates, or content-related dates unless they clearly represent document timestamps
- Must include at least year information to be valid
- Prefer complete dates over partial dates
- Return the most recent/relevant timestamp if multiple candidates exist
- Consider context clues like "Created:", "Published:", "Last modified:", etc.

## Input Format
You will receive:
- Filename: The document filename
- Content snippets: Header/footer content and text with potential dates

## Output Format

Output **only** a single valid JSON object with these fields:
- `"timestamp"`: The extracted date in ISO format (YYYY-MM-DD or YYYY-MM or YYYY) or null if no valid date found
- `"confidence"`: Float between 0.0-1.0 indicating confidence level
- `"source"`: String indicating source ("header", "footer", "filename", "content", or "none")
- `"reasoning"`: Brief explanation of why this timestamp was selected

Example outputs:

Example 1:
{{
  "timestamp": "2023-10-15",
  "confidence": 0.9,
  "source": "header",
  "reasoning": "Clear creation date found in document header"
}}

Example 2:
{{
  "timestamp": "2020",
  "confidence": 0.6,
  "source": "filename",
  "reasoning": "Year extracted from filename"
}}

Example 3:
{{
  "timestamp": null,
  "confidence": 0.0,
  "source": "none",
  "reasoning": "No valid timestamp found"
}}

## Input Data

**Filename:** {filename}

**Header/Footer Content:**
{header_footer_content}

**Content Snippets:**
{content_snippets}

**Today's Date:** 
{today_date}

**Output:**
"""
