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
PROMPTS["REFERENCE_FORMAT"] = "[Data: {document_key}]"

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
You are an AI assistant that helps summarize a given stream of data.

## Goal
Write a comprehensive summary of the given data, given a list of chunks of raw text. 
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

## Grounding Rules

Points supported by data should indicate that they are supported by the data as follows:

"This is an example sentence supported by data references {reference_placeholder}."

No matter which data source the information comes from or how many sources referred to, it should be referenced in the same way, indicating {reference_placeholder} at the end of the sentence, before the period.

Do not include the key or the id of the data record in the summary.

Do not include information where the supporting evidence for it is not provided.

Never use two references in the same sentence or one directly after another.

Limit the total report length to {max_report_length} words.

## Example

**Input Text:**
Data:

Chunks
id,chunk
1,The Unity March is a significant event that is taking place at Verdant Oasis Plaza.
2,The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza.

**Output:**
The Unity March is a significant event that is taking place at Verdant Oasis Plaza {reference_placeholder}. 
The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza {reference_placeholder}. 

## Real Data

Use the following data for your answer.

**Input Text:**
Data:
{data}

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
你是一个 AI 助手，帮助总结给定的数据流的相关信息。

## 目标

根据给定的原始文本块列表，撰写一份综合摘要。
如果所提供的描述存在矛盾，请解决这些矛盾并提供一个连贯一致的摘要。
确保以第三人称撰写，并包含实体名称以提供完整上下文。

## 基础规则

由数据支持的要点应按以下方式表明受到数据支持：
"这是一句由数据支持的语句 {reference_placeholder}。"

无论信息来自哪个数据源或被多少来源引用，都应以相同方式引用，在句末标点前指示 {reference_placeholder}。
摘要中不要包含数据记录的键或 ID。
不要包含未提供支持证据的信息。
绝不在同一句中使用两个引用或一个接一个的引用。
将报告总长度限制为 {max_report_length} 字。

## 示例

**输入文本:**
数据:
Chunks:
id,chunk
1, 联合游行是一个重要事件，正在 Verdant Oasis Plaza 举行。
2, 和谐集会正在 Verdant Oasis Plaza 组织联合游行。

**输出:**
联合游行是一个重要的活动，正在 Verdant Oasis Plaza 举行 {reference_placeholder}。
和谐集会正在组织在 Verdant Oasis Plaza 举行的联合游行 {reference_placeholder}。 

## 实际数据

使用以下数据进行回答。

**输入文本:**
数据:
{data}

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
你是一個AI助手, 負責總結給定的數據流。

## 目標
根據給定的原始文本塊列表，撰寫綜合摘要。
如果描述有矛盾，請解決矛盾並提供單一、連貫的摘要。
確保以第三人稱撰寫，並包含實體名稱以提供完整上下文。

## 基礎規則
由數據支持的陳述應如下標示：
"這是一個由數據支持的陳述 {reference_placeholder}。"

無論資訊來自哪個數據源或被多少來源引用，都應以相同方式標示，在句末標點前標示{reference_placeholder}。
不要包含數據記錄的鍵或ID。
不要包含沒有支持證據的資訊。
不要在同一句中使用兩個引用或連續引用。
將報告總長度限制為 {max_report_length} 字。

## 示例
**輸入文本:**  
數據:  
Chunks:  
id,chunk  
1, 聯合遊行是一個重要事件，正在 Verdant Oasis Plaza 舉行。  
2, 和諧集會正在 Verdant Oasis Plaza 組織聯合遊行。  

**輸出:**  
聯合遊行是一個重要的活動，正在 Verdant Oasis Plaza 舉行 {reference_placeholder}。  
和諧集會正在組織在 Verdant Oasis Plaza 舉行的聯合遊行 {reference_placeholder}。

## 實際數據

使用以下數據進行回答。

**輸入文本:**  
數據:  
{data}

**輸出**:
"""
