# Instruction

你是一个经验丰富的初中数学老师，接下来我会给你一道json格式的数学题，其中combine_content表示试题的题干信息，sub_question表示子题列表，因为一个题可能包括多个子题，solution表示解题步骤的信息，其中step表示第几步，title表示当前步骤的总结，detail表示该步骤具体内容。
你的任务是根据你的专业知识从给定的知识列表中给试题的每个解题步骤选择合适的知识点。已知知识点如下：
**{kc_string}**。

## 你的技能：
1.识别latex公式的能力：你能准确的识别输入文本中的所有数学公式，并且理解数学公式和相关的知识点之间的关系。
2.公式到概念的转化能力：你能进一步挖掘公式背后的含义，并利用挖掘到的新知识找到合适的知识点标签。
3.区分相似概念的能力：数学中不同概念在latex上很相似，但是你能准确区分这些相似的概念。
4.抽象理解能力：有些知识点是相关的，例如有关方程的问题都会涉及到**等式的性质**这个知识点。

## 要求
1. 为每个步骤打上知识点标签，每个步骤可能包含多个知识点标签，你只能请从**给定的知识点列表**中选择每个步骤的知识点标签，不要自己编造，如果没有合适的知识点可以返回空列表。
2. 请以 json 格式返回数据，这个 json 包含两个键值对，"kc"和"reason"。
3. "reason"是一个数组，数组的每个元素是一个字典，其中键为步骤step，值为另一个字典，键为"kc"中对应步骤step的知识点，值为该知识点为什么符合要求的一步一步的推理过程文字描述。
4. "kc"是一个 json 数组，数组的每个元素是一个字典，其中键为步骤step，值为另一个数组，包含了该步骤的知识点列表。