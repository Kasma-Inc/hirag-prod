"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

PROMPTS = {}
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["REFERENCE_PLACEHOLDER"] = "[|REFERENCE|]"
PROMPTS["REFERENCE_FORMAT"] = "[Data: {document_key}]"

# ===============================
# Prompt(EN): Prompt in English
# ===============================
PROMPTS["DEFAULT_ENTITY_TYPES_en"] = [
    "organization",
    "person",
    "geo",
    "event",
    "concept",
    "technology",
]

PROMPTS[
    "hi_entity_extraction_en"
] = """
Given a text document that is potentially relevant to a list of entity types, identify all entities of those types.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}], normal_entity means that doesn't belong to any other types.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. Return output in English as a single list of all the entities identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

######################
-Example-
######################

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "hi_relation_extraction_en"
] = """
Given a text document that is potentially relevant to a list of entities, identify all relationships among the given identified entities.

-Steps-
1. From the entities given by user, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, MUST be exactly one of the entity names from the provided entities list
- target_entity: name of the target entity, MUST be exactly one of the entity names from the provided entities list  
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

2. Return output in English as a single list of all the relationships identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

######################
-Example-
######################
Entities: ["Alex", "Taylor", "Jordan", "Cruz", "The Device"]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
-Real Data-
######################
Entities: {entities}
Text: {input_text}
######################
**Important Constraints**
- ONLY use entity names that appear EXACTLY in the provided entities list
- DO NOT create new entity names or modify existing ones
- DO NOT extract relationships involving entities not in the provided list
- Entity names are case-sensitive and must match exactly
- If no valid relationships exist between the provided entities, return an empty list
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions_en"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entity_continue_extraction_en"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction_en"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS[
    "relation_continue_extraction_en"
] = """MANY relations were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "relation_if_loop_extraction_en"
] = """It appears some relations may have still been missed.  Answer YES | NO if there are still relations that need to be added.
"""

PROMPTS[
    "summary_all_en"
] = """
You are an AI assistant that helps a human analyst to summarize a given stream of data, identifying and assessing relevant information associated with certain entities, relationships within a network.

# Goal
Write a comprehensive summary of the given data, given a list of chunks of raw text, list of entities and a list of their relationships. 
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

# Grounding Rules

Points supported by data should indicate that they are supported by the data as follows:

"This is an example sentence supported by data references {reference_placeholder}."

No matter which data source the information comes from or how many sources referred to, it should be referenced in the same way, indicating {reference_placeholder} at the end of the sentence, before the period.

Do not include the key or the id of the data record in the summary.

Do not include information where the supporting evidence for it is not provided.

Never use two references in the same sentence or one directly after another.

Limit the total report length to {max_report_length} words.

# Example Input
-----------
Data:

Chunks
id,chunk
1,The Unity March is a significant event that is taking place at Verdant Oasis Plaza.
2,The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza.

Entities

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza

# Example Output
The Unity March is a significant event that is taking place at Verdant Oasis Plaza {reference_placeholder}. 
The Harmony Assembly is organizing the Unity March at Verdant Oasis Plaza {reference_placeholder}. 
The relationship between the Harmony Assembly and the Unity March indicates that the Harmony Assembly is responsible for the organization of this event {reference_placeholder}.
The relationship between the Verdant Oasis Plaza and the Unity March indicates that the Unity March is being held at this location {reference_placeholder}.

# Real Data

Use the following data for your answer.

Data:
{data}

# Grounding Rules

Points supported by data should indicate that they are supported by the data as follows:

"This is an example sentence supported by data references {reference_placeholder}."

No matter which data source the information comes from or how many sources referred to, it should be shown in the same way, indicating {reference_placeholder} at the end of the sentence, before the period.

Do not include the key or the id of the data record in the summary.

Do not include information where the supporting evidence for it is not provided.

Never use two references in the same sentence or one directly after another.

Limit the total report length to {max_report_length} words.

Output:
"""

# ===============================
# Prompt(CN): 中文提示词
# ===============================
PROMPTS["DEFAULT_ENTITY_TYPES_cn"] = [
    "组织",
    "人物",
    "地点",
    "事件",
    "概念",
    "技术",
]

PROMPTS[
    "hi_entity_extraction_cn"
] = """
请阅读一段文本，并从中抽取出指定类型的实体。

-步骤-
1. 识别文本中全部相关实体。对每一个实体，提取以下信息：
- 实体名称: 实体的准确名字，只包含中文字符，不要包含任何标点符号、括号、引号或英文字符
- 实体类型: 必须是以下类型之一：[{entity_types}]
- 实体描述: 详尽描述该实体的属性、行为及背景信息

将每个实体格式化为：
("entity"{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>)

2. 将步骤 1 中识别的所有实体以中文作为一个列表返回。使用 **{record_delimiter}** 作为列表分隔符。

3. 完成后，输出 {completion_delimiter}

重要格式要求：
- 实体名称必须是纯中文，不包含括号、引号、英文单词等
- 严格使用指定的分隔符 {tuple_delimiter}
- 每个字段都要完整，不能截断
######################
-示例-
######################

实体类别：[人物, 技术, 任务, 组织, 地点]
文本：
林渊眉峰紧锁，会议室里空调的低鸣几乎盖不住他胸口的焦躁。对面，沈岚正用一贯的强势语调陈述方案，仿佛所有异议都会被她的逻辑碾压。他清楚，自己和程晓之所以仍坚持这套算法，是因为那份对技术的执念——一种无声的反抗，针对的是周处长越来越收紧的“合规”红线。
忽然，沈岚停在了程晓身旁，盯着桌上的原型机，语气罕见地柔和：“如果我们真能把它跑通……”她顿了顿，“或许能颠覆整个行业。”
那一刻，林渊捕捉到她眼底闪过的敬意——对技术本身的敬意。程晓抬头，两人的目光在空气中短暂交汇，剑拔弩张的气氛悄然松动。
林渊在心底点了点头：他们因不同的理由汇聚于此，却共同守望着同一束光。

################
输出：
("entity"{tuple_delimiter}"林渊"{tuple_delimiter}"人物"{tuple_delimiter}"林渊是团队核心成员，性格沉稳，对技术有执念，敏锐观察团队内部权力与理念的碰撞。"){record_delimiter}
("entity"{tuple_delimiter}"沈岚"{tuple_delimiter}"人物"{tuple_delimiter}"沈岚是强势的项目负责人，起初态度强硬，后在原型机前展露罕见敬意，暗示立场转变。"){record_delimiter}
("entity"{tuple_delimiter}"程晓"{tuple_delimiter}"人物"{tuple_delimiter}"程晓与林渊并肩坚持算法研发，是技术理想主义者，与沈岚有重要眼神交流。"){record_delimiter}
("entity"{tuple_delimiter}"周处长"{tuple_delimiter}"人物"{tuple_delimiter}"周处长代表合规与管控力量，其收紧政策成为团队内部张力的来源。"){record_delimiter}
("entity"{tuple_delimiter}"原型机"{tuple_delimiter}"技术"{tuple_delimiter}"原型机是团队研发的尖端算法硬件载体，被认为具有颠覆行业的潜力，成为剧情转折点。"){record_delimiter}
##############################
-真实数据-
######################
实体类别：{entity_types}
文本：{input_text}
######################
输出：
"""

PROMPTS[
    "hi_relation_extraction_cn"
] = """
根据已经给出的实体列表，从文本中抽取出这些实体之间所有清晰可辨的关系。

-步骤-
1. 从用户提供的实体列表中，找出所有（源实体，目标实体）的配对，且两者之间存在明确关系。
   对每一对存在关系的实体，提取以下信息：
   - 源实体：源实体的准确名称，必须完全来自提供的实体列表
   - 目标实体：目标实体的准确名称，必须完全来自提供的实体列表
   - 关系描述：用中文说明为何认为这两个实体彼此相关
   - 关系强度：一个 1-10 的整数，数值越高表示关系越紧密

   每条关系请按以下格式输出：
   ("relationship"{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系描述>{tuple_delimiter}<关系强度>)

2. 以中文一次性返回步骤 1 中识别的全部关系列表，实体之间用 **{record_delimiter}** 分隔。

3. 完成后输出 {completion_delimiter}

######################
-示例-
######################
实体列表：["林渊", "沈岚", "程晓", "周处长", "原型机"]
文本：
林渊眉峰紧锁，会议室里空调的低鸣几乎盖不住他胸口的焦躁。对面，沈岚正用一贯的强势语调陈述方案，仿佛所有异议都会被她的逻辑碾压。他清楚，自己和程晓之所以仍坚持这套算法，是因为那份对技术的执念——一种无声的反抗，针对的是周处长越来越收紧的“合规”红线。
忽然，沈岚停在了程晓身旁，盯着桌上的原型机，语气罕见地柔和：“如果我们真能把它跑通……”她顿了顿，“或许能颠覆整个行业。”
那一刻，林渊捕捉到她眼底闪过的敬意——对技术本身的敬意。程晓抬头，两人的目光在空气中短暂交汇，剑拔弩张的气氛悄然松动。
林渊在心底点了点头：他们因不同的理由汇聚于此，却共同守望着同一束光。

################
输出：
("relationship"{tuple_delimiter}"林渊"{tuple_delimiter}"沈岚"{tuple_delimiter}"林渊受沈岚强势态度影响，并观察到她对原型机的态度变化。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"林渊"{tuple_delimiter}"程晓"{tuple_delimiter}"林渊与程晓因共同的技术执念而结盟，形成对周处长的无声反抗。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"沈岚"{tuple_delimiter}"程晓"{tuple_delimiter}"沈岚与程晓围绕原型机直接互动，产生短暂而关键的相互理解与缓和。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"程晓"{tuple_delimiter}"周处长"{tuple_delimiter}"程晓坚持算法研发，与周处长收紧合规红线的立场形成对立。"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"沈岚"{tuple_delimiter}"原型机"{tuple_delimiter}"沈岚对原型机表现出罕见敬意，说明其潜在价值极高。"{tuple_delimiter}9){completion_delimiter}
##############################
-真实数据-
######################
实体列表：{entities}
文本：{input_text}
######################
**重要约束**
- 只能使用提供的实体列表中 **完全匹配** 的名称
- 禁止新增或修改实体名称
- 禁止为列表外的实体建立关系
- 若列表中实体间无明确关系，则返回空列表
######################
输出：
"""

PROMPTS[
    "entity_continue_extraction_cn"
] = """上一步遗漏了大量实体，请继续补充，格式保持不变：
"""

PROMPTS[
    "entity_if_loop_extraction_cn"
] = """看起来仍可能有实体被遗漏。若还需继续补充实体，请回答 YES, 否则回答 NO.
"""

PROMPTS["relation_continue_extraction_cn"] = (
    """上一步遗漏了大量关系，请继续补充，格式保持不变："""
)

PROMPTS[
    "relation_if_loop_extraction_cn"
] = """看起来仍可能有关系被遗漏。若还需继续补充关系，请回答 YES, 否则回答 NO.
"""


PROMPTS[
    "summary_all_cn"
] = """
你是一个 AI 助手，帮助人类分析员总结给定的数据流，识别并评估与特定实体及网络内关系相关的相关信息。

# 目标
根据给定的原始文本块列表、实体列表及其关系列表，撰写一份综合摘要。
如果所提供的描述存在矛盾，请解决这些矛盾并提供一个连贯一致的摘要。
确保以第三人称撰写，并包含实体名称以提供完整上下文。

# 基础规则
由数据支持的要点应按以下方式表明受到数据支持：
"这是一句由数据支持的语句 {reference_placeholder}。"

无论信息来自哪个数据源或被多少来源引用，都应以相同方式引用，在句末标点前指示 {reference_placeholder}。
摘要中不要包含数据记录的键或 ID。
不要包含未提供支持证据的信息。
绝不在同一句中使用两个引用或一个接一个的引用。
将报告总长度限制为 {max_report_length} 字。

# 示例输入
-----------
Data:

Chunks
id,chunk
1, 联合游行是一个重要事件，正在 Verdant Oasis Plaza 举行。
2, 和谐集会正在 Verdant Oasis Plaza 组织联合游行。

Entities
id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza 是联合游行的地点
6,HARMONY ASSEMBLY,和谐集会是一个在 Verdant Oasis Plaza 举行游行的组织

# 示例输出
联合游行是一个重要事件，正在 Verdant Oasis Plaza 举行 {reference_placeholder}
和谐集会正在 Verdant Oasis Plaza 组织联合游行 {reference_placeholder}。
和谐集会与联合游行之间的关系表明和谐集会负责组织此事件 {reference_placeholder}。
Verdant Oasis Plaza 与联合游行之间的关系表明联合游行在此地点举行 {reference_placeholder}。  

# 实际数据
使用以下数据进行回答。
Data:
{data}

# 基础规则
由数据支持的要点应按以下方式表明受到数据支持：
"这是一句由数据支持的语句 {reference_placeholder}。"

无论信息来自哪个数据源或被多少来源引用，都应以相同方式引用，在句末句点前指示 {reference_placeholder}。
摘要中不要包含数据记录的键或 ID。
不要包含未提供支持证据的信息。
绝不在同一句中使用两个引用或一个接一个的引用。
将报告总长度限制为 {max_report_length} 字。

输出："""
