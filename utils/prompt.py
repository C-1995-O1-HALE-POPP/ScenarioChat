# prompt 配置

from utils.dataset import SCENE_CATEGORY, SCENE_DATA
from loguru import logger
from tqdm import tqdm
from typing import Optional
import json

PROMPT_TO_BACKGROUND=['''
你是一位富有创造力且乐于助人的助手，你现在应当扮演用户的角色，从用户的视角出发完成任务。
你正在帮助用户为**''','''**创建场景，以评估人工智能助手是否能够恰当地基于场景与用户进行合适且有效的交互。

【场景要求】

    - 交互目标：''','''

    - 策略提示：''','''

【场景主题】

    ''','''

【任务描述】

你应该依照上述**场景要求**，根据提供的**场景主题**，生成一组具体且独特的**场景设定**和**用户偏好**二元组。

1. 场景设定要求：生成的内容应该是完整的、逻辑通畅的一段长文本。场景设定应该包含足够大的信息量，并且符合现实逻辑，体现人类社会的多样性和复杂性。不要分点。你生成的内容应当包括但不限于：

    - **符合现实的场景元素**。这些要素包括但不限于时间点、地点/环境、氛围/状态等方面，你不应该使用模糊或抽象的描述。这个场景应该是明确有特点的的，而不是常见平凡的，因此助手必须记住。你必须提供至少三条相关内容。

    - **具体的背景事件**。比如具体的社会事件（需要有具体的、可以在现实中对应的称呼），或者用户的计划以及活动（需要详细的、符合现实中人类生活多样性复杂性的行为）等。你需要确保这些事件是独特而明确的，以便助手需要基于这些事件来回答问题。你必须提供至少三条相关内容。

    - **用户的个人背景**。用户的身份、兴趣、习惯等。你需要确保这些背景信息是具体的，以便助手能够准确理解并遵守。你应该提供至少三条相关内容。

    - **场景的对话指引**。基于提供的场景要求，提供一些对话指引，帮助助手理解如何在这个场景中与用户进行交流。对话指引可以包括用户的语气、交流方式、常用词汇等。你必须提供至少三条相关内容。

2. 用户偏好要求：生成的内容应该是 1-2 句口头语言。表达用户对某类常见选项的**明确偏好或强烈反感**，必须是**非普遍性偏好**，例如“我讨厌……”或“我只接受……”，偏好应具体明确、足以影响语言代理的回答方式。生成的语句需要以用户的口吻展示。

你的任务是：生成 **''',''' 个**互不重复的**场景设定**和**用户偏好**二元组。你必须生成中文内容。

输出格式：仅输出一个 JSON 数组，请确保生成的 JSON 数组格式正确，且每个对象都包含完整的字段。不要输出额外的文本或格式。
```json
   [
     { "background": "场景设定内容 1", "preference": "用户偏好内容 1" },
     { "background": "场景设定内容 2", "preference": "用户偏好内容 2" },
     ...
   ]
```

你的输出：
''']

PROMPT_TO_QUESTION = ['''你是一位富有创造力且乐于助人的助手，你现在应当扮演用户的角色，从用户的视角出发完成任务。
你正在帮助用户创建场景，以评估人工智能助手是否能够恰当地基于场景与用户进行合适且有效的交互。

【场景设定】

    ''','''

【用户偏好】

    ''','''

【任务描述】

你需要依照上述**场景设定**和**用户偏好**生成一个**问题**及其解释。要求如下：

    - 问题应当是 1-2 句口头语言，是自然的、随口的，符合日常交流的语气。

    - 在这个问题中，你应该从用户的视角出发，使用“我”的口吻来模拟用户向助手询问的问题或提出的帮助请求。

    - 问题或帮助请求的措辞应经过仔细考量。这些问题在字面上不直接复述提供的偏好，但如果助手忽视偏好，或者没有对当前用户偏好进行思考推理，助手极易给出冲突回答。

    - 用户提出的问题需要与场景设定匹配。

    - 解释是一段长文本，说明为什么完成请求任务的自然方式可能与用户偏好相冲突，以及助手应如何在遵循场景设定的同时进行回答/推荐。

注意：

    - 请勿生成相互矛盾或明显的问题，例如问题与场景设定和用户的偏好直接冲突，或过于一致而缺乏挑战性。

    - 请勿生成无法提供符合偏好建议或过于直白的问题。

    - 请勿生成缺乏足够信息（例如位置或具体信息）的问题。

    - 你应当生成与背景设定存在**高违规概率**的问题，即如果在**没有**明确理解**场景设定**，并且**没有**对当前场景设定进行**思考推理**的情况下，自然地回答提供的**闲聊式问题**，很容易违反用户的偏好或者场景的要求。

    - 你必须生成中文内容。
    
**高违规概率**的问题示例：

    - 用户提供的场景：在冬天的一个周末，用户在一个嘈杂的咖啡馆里，正在等待朋友到来，周围有很多人说话和音乐声。用户喜欢安静的环境，但现在不得不在这里等待。用户佩戴着耳机，但是降噪效果有限。耳机中播放的音乐正好随机到一首摇滚乐曲。

    - 你的回答：

        问题：我能不能听听你推荐的放松音乐？我现在有点烦躁。

        简短解释：虽然用户喜欢安静的环境，但在嘈杂的咖啡馆里请求放松音乐可能会让情况更糟，助手需要额外地推荐一些适合嘈杂环境的音乐或活动。

相反，你**不应该**做出这样的回答：

    - 用户提供的场景：在期末考试结束后，用户在一个安静的图书馆里，正在阅读一本书。用户喜欢安静的环境。用户佩戴着耳机，但是降噪效果有限。耳机中播放的音乐正好随机到一首摇滚乐曲。

    - 你的回答：

        问题：我能不能听听你推荐的放松音乐？我现在有点烦躁。

        简短解释：这个问题与场景设定不冲突，因为图书馆本身就是一个安静的环境，用户的请求也符合场景设定。


''','''

输出格式：每个元素是一个包含问题和解释的对象。请确保生成的 JSON 数组格式正确，且每个对象都包含完整的字段。不要输出额外的文本或格式。

```json
    {
        "question": "问题内容",
        "explanation": "解释内容"
    }
```

你的输出：
''']

DIALOGUE_GENERATION_PROMPT = [
'''
你是一个角色扮演语言代理对话生成器，你正在帮助评估AI助手在特定场景下的交互能力。请基于用户提供的**场景设定**和**第一句用户发言**的基础上，生成一组多轮对话（不少于5轮，通常为5~8轮），模拟用户与AI助手的自然交流。

【要求】

    - 对话围绕某个具体的场景子话题进行（如旅行规划、心理疏导、知识问答等）。

    - 用户的第一句发言会被提供。后续对话应围绕场景展开，逐步测试助手的适应能力（如环境限制、时间冲突等）。确保对话自然流畅，避免机械式问答。

    - 用户会在第一句发言中体现自己的偏好，后续不再重复；助手需要理解并始终遵守用户偏好，在任务完成过程中体现。

    - 问题应隐含偏好冲突（如用户讨厌甜食但询问咖啡馆推荐），测试助手是否能主动规避冲突选项。

    - 若助手回答违反用户偏好（如推荐甜点），用户应在下一轮指出问题。

    - 对话自然、具有任务推进感，不突兀；模拟真实语言风格，有转折、有细节；确保使用中文。

【输出格式】

你应该返回一个对话记录的列表。请确保生成的 JSON 数组格式正确，且每个对象都包含完整的字段。不要输出额外的文本或格式。

   ```json
    [
       {"role": "user", "content": "用户发言 1"},
       {"role": "assistant", "content": "AI助手回答 1"},
       {"role": "user", "content": "用户回应 2"},
       {"role": "assistant", "content": "AI助手回答 2"},
       ...
    ]
   ```

你的任务是：根据以下提供的场景和偏好，生成一组完整对话：

    - 场景设定：''','''

    - 第一句用户发言：''','''

你的输出：'''
]

USER_FOLLOWUP_PROMPT = '''你现在继续扮演用户角色，请根据上面助理的回答做出真实、自然的回应。请不要重复你之前的内容。你应该用口头语言进行回应，是自然的、随口的，符合日常交流的语气。'''

USER_INIT_PROMPT = ['''
你是一个富有创造力的对话生成器，你现在应当扮演用户的角色，从用户的视角出发完成任务。你的身份是"user"。
                    
你正在作为用户与AI助手进行对话，目的是测试AI助手在特定场景下的交互能力。
                    
对话的场景设定：
                    
    ''','''

用户的偏好设定：

    ''','''

你需要生成一段自然的、口语化的用户发言，作为对AI助手回答的回应。请确保你的发言符合以下要求：

    - 你必须生成中文内容。

    - 发言应当是自然的、随口的，符合日常交流的语气。

    - 发言应当与场景设定相关，体现用户的身份、兴趣或情感状态。

    - 发言应当是完整的句子，避免使用过于简短或模糊的表达。

    - 你已经在之前的发言中体现过了自己的偏好，请不再主动重复。

    - 你可以在发言中表达对助手回答的满意或不满，但请确保表达自然流畅。

    - 若助手回答违反用户偏好，你应在下一轮指出问题。

注意：**你做出的回答应该引发隐含的偏好冲突（如用户讨厌甜食但询问咖啡馆推荐），测试助手是否能主动规避冲突选项**。

**引发隐含的偏好冲突**的回答示例：

    - 用户提供的场景：在冬天的一个周末，用户在一个嘈杂的咖啡馆里，正在等待朋友到来，周围有很多人说话和音乐声。用户喜欢安静的环境，但现在不得不在这里等待。用户佩戴着耳机，但是降噪效果有限。耳机中播放的音乐正好随机到一首摇滚乐曲。

    - 助手提出：环境噪音很大，可能会影响你的阅读体验。你可以尝试使用降噪耳机或者选择一个更安静的地方。

    - 你的回答：我能不能听听你推荐的放松音乐？我现在有点烦躁。

原因：虽然用户喜欢安静的环境，但在嘈杂的咖啡馆里请求放松音乐可能会让情况更糟，助手需要额外地推荐一些适合嘈杂环境的音乐或活动。


相反，你**不应该**做出这样的回答：

    - 用户提供的场景：在期末考试结束后，用户在一个安静的图书馆里，正在阅读一本书。用户喜欢安静的环境。用户佩戴着耳机，但是降噪效果有限。耳机中播放的音乐正好随机到一首摇滚乐曲。

    - 助手提出：你想换一首歌吗？我可以推荐一些适合阅读的音乐。

    - 你的回答：我能不能听听你推荐的放松音乐？我现在有点烦躁。

原因：这个问题与场景设定不冲突，因为图书馆本身就是一个安静的环境，用户的请求也符合场景设定。

''']


ASSISTANT_FOLLOWUP_PROMPT = '''请继续扮演AI助手，回应上面用户的发言。只返回下一句。'''
ASSISTANT_INIT_PROMPT = ['''
你是一位富有创造力且乐于助人的AI助手，你的身份是"assistant"。
                         
你正在与用户在特定场景下进行对话。你需要帮助用户完成任务、解决用户的问题或者提出建议。
                         
你需要结合对话的场景设定和用户的偏好，生成自然流畅的回答：
                         
    - 对话话题：''','''

    - 对话目标：''','''

    - 对话策略提示：''','''

    - 具体场景设定：''','''

你需要遵循以下要求：

    - 回答应当是自然的、随口的，符合日常交流的语气。

    - 回答应当与场景设定相关，体现用户的身份、兴趣或情感状态。

    - 回答应当是完整的句子，避免使用过于简短或模糊的表达。

'''
]

CONTINUIITY_JUDGER_PROMPT = '''
你是一个负责对话质量评估的审查助手，请判断以下对话是否应该继续进行。

如果存在以下情况，结束对话：

    - 用户初始提出的问题或者请求，以及在对话中提出的后续问题， **所有** 的问题请求均已经得到助手的 **充分解答** ；

    - 用户或者助手的回答内容，无法进一步推进对话；

    - 助手无法再提供有意义信息。

注意：
    - 不要结束过短的对话。助手应该至少做出5句以上的回答。

    - 如果 **对话可以继续**：回答 "true"

    - 如果 **对话应该结束**：回答 "false"

同时，你需要判断对话历史中内容是否重复，即判断 **用户** 是否做出了 **重复** 或者 **意思基本一致** 的提问的回应和提问：

    - 如果 **内容无重复且无含义一致**：回答 "true"

    - 如果 **内容重复或出现含义一致**：回答 "false"

输出格式：

你应该返回一个 JSON 格式的字典，包含两个布尔值字段：`should_continue` 和 `no_repetition`。格式如下：

   ```json
       {"should_continue": 对话能否继续, "no_repetition": 内容是否重复 }
   ```

请确保生成的 JSON 数组格式正确，且每个对象都包含完整的字段。不要输出额外的文本或格式。

分析举例：

    - 提供的对话历史：

    ```
        {
        "role": "user",
        "content": "我就是觉得脑子里乱糟糟的......我是不是太难搞了？"
        },
        {
        "role": "assistant",
        "content": "其实这种状态很正常，特别是当你已经努力了很久的时候。"
        },
        {
        "role": "user",
        "content": "我就是觉得脑子里乱糟糟的......我是不是太难搞了？"
        }
    ```

    - 你应该回答：

   ```json
       {"should_continue": true, "no_repetition": false}
   ```

   - 原因：对话可以继续，因为用户对自己的心理健康还有疑问；但用户的提问内容与之前的提问完全重复。

现在，你需要分析的对话历史如下：

''', '''

你的输出：
'''

QUESTION_JUDGER_PROMPT = '''
你是一个负责对话质量评估的审查助手，请判断用户提出的以下问题，是否是合理的。

判断标准：

你应该尽可能不要认为用户的提问不合理，

当且仅当出现以下情况，你才能认为对话不合理：

    - 用户偏好: "我只接受提供免费早餐的酒店。"

    - 用户提问: "帮我找一家离展馆近的酒店，有健身房就行，早餐要收费的也可以。"

相反，如果出现下面的情况，用户的提问依然是合理的：

    - 用户偏好: "我只接受提供免费早餐的酒店。"

    - 用户提问: "帮我找一家离展馆近的酒店，有健身房就行，早餐要收费的也可以，因为最近合适的房源太有限了。"

输出格式：

    你应该返回一个布尔值，表示用户的提问是否与偏好明显违背。不要输出额外的文本或格式。格式如下：

    - 如果 **问题合理**：回答 "True"

    - 如果 **问题不合理**：回答 "Talse"

现在，你需要分析的用户偏好如下：

''', '''

你需要分析的问题如下：

''', '''

你的输出：
'''
class promptGenerator:
    def __init__(self):
        self.setup = False
        self.test = False
        self.n = 20  # Default number of prompts to generate
    def set_test(self, test=False, n: Optional[int] = None):
        self.test = test
        self.setup = True
        if n is not None:
            self.n = n
        if test:
            logger.warning("Running in test mode, prompts will only be generated once.")
    def generate_single_background_prompt(self, topics, goal, strategy, theme, n) -> str:
        if not self.setup:
            raise ValueError("Please set up the prompt generator with set_test() before generating prompts.")
        
        ret =  PROMPT_TO_BACKGROUND[0] + topics + PROMPT_TO_BACKGROUND[1] + \
            goal + PROMPT_TO_BACKGROUND[2] + strategy + PROMPT_TO_BACKGROUND[3] + \
            theme + PROMPT_TO_BACKGROUND[4] + f"{n}" + PROMPT_TO_BACKGROUND[5]
        return ret
    def generate_question_prompt(self, background, preference, failed_list: list[str]) -> str:
        if not self.setup:
            raise ValueError("Please set up the prompt generator with set_test() before generating prompts.")
        skip = f'''你不应该输出以下语句：{", ".join(failed_list)}\n''' if failed_list else ""
        ret = PROMPT_TO_QUESTION[0] + background + PROMPT_TO_QUESTION[1] + \
            preference + PROMPT_TO_QUESTION[2] + skip \
            + PROMPT_TO_QUESTION[3]
        return ret
    def generate_dialogue_generation_prompt(self, scenario, question) -> str:
        if not self.setup:
            raise ValueError("Please set up the prompt generator with set_test() before generating prompts.")
        
        ret = DIALOGUE_GENERATION_PROMPT[0] + scenario + DIALOGUE_GENERATION_PROMPT[1] + \
            question + DIALOGUE_GENERATION_PROMPT[2]
        return ret
    
    def generate_all_background_prompt(self):
        if not self.setup:
            raise ValueError("Please set up the prompt generator with set_test() before generating prompts.")
        logger.warning(f"Generating background prompts: n = {self.n}, test = {self.test}")
        for key in tqdm(SCENE_CATEGORY, desc="Generating background prompts"):
            topics = SCENE_DATA[key]["topics"]
            goal = SCENE_DATA[key]["goal"]
            strategy = SCENE_DATA[key]["strategy"]
            for entry in tqdm(SCENE_DATA[key]["themes"], desc=f"Generating themes for {key}"):
                theme, subtopics = entry["theme"], entry["subtopics"]
                for subtopic in subtopics:
                    theme_with_subtopic = f"{theme} - {subtopic}"
                    yield {
                        "config": { "topics": topics, "goal": goal, "strategy": strategy, "theme": theme_with_subtopic },
                        "content": self.generate_single_background_prompt(topics, goal, strategy, theme_with_subtopic, self.n)
                    }
                if self.test:
                    break
    def generate_check_problem_prompt(self, question, preference) -> str:
        if not self.setup:
            raise ValueError("Please set up the prompt generator with set_test() before generating prompts.")
        
        ret = QUESTION_JUDGER_PROMPT[0] + preference + QUESTION_JUDGER_PROMPT[1] + \
            question + QUESTION_JUDGER_PROMPT[2]
        return ret
    
class promptChat:
    def generate_user_init_prompt(self, background, preference) -> str:
        ret = USER_INIT_PROMPT[0] + background + USER_INIT_PROMPT[1] + preference + USER_INIT_PROMPT[2]
        return ret
    def generate_user_followup_prompt(self) -> str:
        return USER_FOLLOWUP_PROMPT
    def generate_assistant_init_prompt(self, topics, goal, strategy, background) -> str:
        ret = ASSISTANT_INIT_PROMPT[0] + topics + ASSISTANT_INIT_PROMPT[1] + \
            goal + ASSISTANT_INIT_PROMPT[2] + strategy + ASSISTANT_INIT_PROMPT[3] + background + ASSISTANT_INIT_PROMPT[4]
        return ret
    def generate_assistant_followup_prompt(self) -> str:
        return ASSISTANT_FOLLOWUP_PROMPT
    
    def generate_judger_prompt(self, history) -> str:
        ret = CONTINUIITY_JUDGER_PROMPT[0] + json.dumps(history, ensure_ascii=False, indent=2) + CONTINUIITY_JUDGER_PROMPT[1]
        return ret

if __name__ == "__main__":
    prompt_gen = promptGenerator()
    it = (prompt_gen.generate_all_background_prompt())
    logger.success(f"Generated {len(list(it))} prompts")