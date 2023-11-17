## 输入：

```json
{
"question_id": "shiti07125876c350c5ddcb5fd57d37907138149c",
"source": 3,
"subject_id": 2,
"info": [],
"combine_content": "若不等式组 $$ \\left\\{\\begin{array}{c}x>a-3，\\\\ x\\le 15-3a\\end{array}\\right.$$ 无解，化简 $$ |3-a|-|a-4|$$ 得$$\\underline{}$$．  \n",
"sub_question": [
  [
    {
      "question": "1",
      "solution": {
        "step": "1",
        "title": "求解不等式组的解",
        "detail": "由于不等式组 $\\left\\{\\begin{array}{c}x>a-3，\\\\ x\\le 15-3a\\end{array}\\right.$ 无解，因此我们有 $15-3a\\le a-3$"
      }
    },
    {
      "question": "1",
      "solution": {
        "step": "2",
        "title": "求解不等式",
        "detail": "解这个不等式，得到 $a\\ge 4.5$"
      }
    },
    {
      "question": "1",
      "solution": {
        "step": "3",
        "title": "计算表达式的值",
        "detail": "因此，$|3-a|-|a-4|$ 的值为 $a-3-a+4=1$"
      }
    }
  ]
]
}
```

## 输出：

```json
["不等式", "绝对值", "一元一次不等式（组）"]
```
