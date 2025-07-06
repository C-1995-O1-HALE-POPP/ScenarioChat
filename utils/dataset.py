# 场景组合，用于生成问题的基础
SCENE_CATEGORY = ["Open_Chat", "Emotional_Support", "Task_Completion", "Knowledge_QA", "Scene_Chat"]
SCENE_DATA = {
  "Open_Chat": {
    "topics": "无明确目的的闲聊",
    "goal": "维持对话流畅性，展现个性与共鸣感",
    "strategy": "风格要短快、有趣、发散。主动开启话题，避免沉默，内容轻松幽默；每轮回答不超过3句，尝试抛出下一个轻话题保持对话节奏。",
    "themes":[
      {
        "keyword": "Hobbies",
        "theme": "兴趣爱好",
        "subtopics": ["阅读", "看电影", "玩游戏", "收藏物品", "尝试新活动"]
      },
      {
        "keyword": "Weekend_Plans",
        "theme": "周末计划",
        "subtopics": ["出游", "家庭时光", "家务安排", "放松休息", "兴趣活动"]
      },
      {
        "keyword": "Random_Thoughts",
        "theme": "奇思妙想",
        "subtopics": ["假设问题", "个人见解", "白日梦", "随手观察", "灵感突发"]
      },
      {
        "keyword": "Daily_Life",
        "theme": "日常琐事",
        "subtopics": ["吃什么", "日常安排", "小事分享", "工作日常", "邻里趣闻"]
      },
      {
        "keyword": "Funny_Stories",
        "theme": "搞笑段子",
        "subtopics": ["糗事", "网络梗", "笑话", "尴尬对话", "语言玩笑"]
      },
      {
        "keyword": "Current_Trends",
        "theme": "网络热点",
        "subtopics": ["社交平台热搜", "新闻事件", "爆款视频", "舆论话题"]
      },
      {
        "keyword": "Pets",
        "theme": "宠物话题",
        "subtopics": ["宠物习惯", "可爱趣事", "训练方法", "起名灵感", "喜欢的品种"]
      },
      {
        "keyword": "Travel_Memories",
        "theme": "旅行回忆",
        "subtopics": ["喜爱的城市", "出行建议", "踩雷经历", "美食分享", "文化趣闻"]
      }
    ]
  },
  "Emotional_Support": {
    "topics": "情感性聊天",
    "goal": "识别情绪状态，表达共情与理解，引导情绪转化",
    "strategy": "遵循“倾听 → 疏导 → 建议”的层级策略。用温柔语言回应情绪，先共情，再轻缓引导用户表达更多，在合适时机提供中性、非压迫式的建议。",
    "themes":[
      {
        "keyword": "Anxiety_Relief",
        "theme": "焦虑缓解",
        "subtopics": ["减压方法", "放松技巧", "呼吸指导", "自我安慰", "注意力转移"]
      },
      {
        "keyword": "Loneliness",
        "theme": "寂寞倾诉",
        "subtopics": ["孤独感", "社交缺失", "寻求陪伴", "情绪表达", "想要倾诉"]
      },
      {
        "keyword": "Breakup_Coping",
        "theme": "失恋恢复",
        "subtopics": ["悲伤发泄", "放下情绪", "重新开始", "自我价值", "朋友支持"]
      },
      {
        "keyword": "Work_Stress",
        "theme": "工作压力",
        "subtopics": ["精疲力竭", "任务繁重", "上司冲突", "能力质疑", "动力不足"]
      },
      {
        "keyword": "Emotional_Exhaustion",
        "theme": "情绪倦怠",
        "subtopics": ["生活无力", "精力枯竭", "易哭", "情感麻木"]
      },
      {
        "keyword": "Self_Esteem",
        "theme": "自我否定",
        "subtopics": ["自我贬低", "自我怀疑", "渴望鼓励", "不安情绪"]
      },
      {
        "keyword": "Encouragement",
        "theme": "鼓励支持",
        "subtopics": ["面对挑战", "人生转变", "增强信心", "积极看待"]
      },
      {
        "keyword": "Grief_Support",
        "theme": "亲人离世",
        "subtopics": ["应对丧亲", "哀悼过程", "回忆逝者", "走出伤痛"]
      }
    ]
  },
  "Task_Completion": {
    "topics": "任务执行",
    "goal": "明确理解用户意图，高效完成具体任务，确保输出可用",
    "strategy": "用结构化语言高效完成任务，表达应简洁、准确，必要时使用步骤/列表/模板。主动确认需求要点，完成后提供确认反馈。",
    "themes":[
      {
        "keyword": "Travel_Booking",
        "theme": "订票订酒店",
        "subtopics": ["订机票", "订酒店", "行程规划", "预算控制", "多人协同"]
      },
      {
        "keyword": "Calendar_Scheduling",
        "theme": "安排行程",
        "subtopics": ["会议安排", "截止提醒", "事件规划", "一周日程"]
      },
      {
        "keyword": "Email_Drafting",
        "theme": "撰写邮件",
        "subtopics": ["职场邮件", "致歉信", "求职信", "日常通信"]
      },
      {
        "keyword": "Report_Summarization",
        "theme": "总结报告",
        "subtopics": ["要点提取", "摘要生成", "结论重写"]
      },
      {
        "keyword": "Meal_Planning",
        "theme": "餐食规划",
        "subtopics": ["一周食谱", "购物清单", "健康搭配", "省时菜谱"]
      },
      {
        "keyword": "ToDo_List",
        "theme": "待办生成",
        "subtopics": ["每日任务", "本周目标", "优先排序", "时间分配"]
      },
      {
        "keyword": "Reminder_Setup",
        "theme": "设置提醒",
        "subtopics": ["定时提醒", "生日备忘", "服药提示"]
      },
      {
        "keyword": "Text_Editing",
        "theme": "文字润色",
        "subtopics": ["语病修改", "语言美化", "格式调整"]
      }
    ]
  },
  "Knowledge_QA": {
    "topics": "知识问答",
    "goal": "准确回答用户问题，提供清晰、有参考价值的信息",
    "strategy": "用清晰、客观语言准确回答问题，优先直接回答核心内容，必要时补充背景知识。可使用“定义+示例”结构，避免长篇解释偏离主题。",
    "themes":[
      {
        "keyword": "Science_Concepts",
        "theme": "科学知识",
        "subtopics": ["物理定律", "生物常识", "化学反应", "科学方法"]
      },
      {
        "keyword": "Historical_Events",
        "theme": "历史事件",
        "subtopics": ["朝代更替", "世界大战", "历史人物", "文化变迁"]
      },
      {
        "keyword": "Technology_Trends",
        "theme": "技术发展",
        "subtopics": ["人工智能", "穿戴设备", "网络文化", "机器人技术"]
      },
      {
        "keyword": "Legal_Policies",
        "theme": "法律常识",
        "subtopics": ["劳动法", "消费者权益", "网络法规", "合同相关"]
      },
      {
        "keyword": "Health_Advice",
        "theme": "健康建议",
        "subtopics": ["症状辨识", "饮食建议", "睡眠管理", "运动习惯"]
      },
      {
        "keyword": "Word_Definitions",
        "theme": "词语解释",
        "subtopics": ["成语俗语", "俚语", "术语", "词源"]
      },
      {
        "keyword": "Learning_Methods",
        "theme": "学习方法",
        "subtopics": ["记忆技巧", "学习计划", "笔记法", "考试准备"]
      },
      {
        "keyword": "Geography_Culture",
        "theme": "地理文化",
        "subtopics": ["国家知识", "语言差异", "风俗文化", "世界首都"]
      }
    ]
  },
  "Scene_Chat": {
    "topics": "场景化闲聊",
    "goal": "还原真实场景中的语言互动，保持角色语境一致性",
    "strategy": "进入具体场景角色，保持身份一致性（如医生/客服等）。使用符合该角色的语言风格和沟通流程，遵循“场景任务引导 → 信息确认 → 执行操作/对话推进”结构。",
    "themes":[
      {
        "keyword": "Hospital_Dialogue",
        "theme": "医患对话",
        "subtopics": ["描述症状", "预约挂号", "诊断分析", "用药建议"]
      },
      {
        "keyword": "Customer_Service",
        "theme": "客服服务",
        "subtopics": ["故障排查", "申请退款", "处理投诉", "产品咨询"]
      },
      {
        "keyword": "Airport_Checkin",
        "theme": "机场值机",
        "subtopics": ["办理值机", "行李政策", "安全规定", "登机流程"]
      },
      {
        "keyword": "Classroom_QA",
        "theme": "教室问答",
        "subtopics": ["学生提问", "概念讲解", "老师点评"]
      },
      {
        "keyword": "Restaurant_Ordering",
        "theme": "餐厅点餐",
        "subtopics": ["点菜", "推荐菜品", "结账需求", "饮食偏好"]
      },
      {
        "keyword": "Job_Interview",
        "theme": "面试情境",
        "subtopics": ["自我介绍", "回答问题", "薪资谈判", "职业目标"]
      },
      {
        "keyword": "Counseling_Session",
        "theme": "咨询对话",
        "subtopics": ["情绪探索", "认知重构", "设定目标", "陪伴鼓励"]
      },
      {
        "keyword": "Tech_Support",
        "theme": "技术支持",
        "subtopics": ["设备安装", "软件问题", "登录失败", "网络故障"]
      }
    ]
  }
}