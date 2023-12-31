## 输入：

```json
{
    "question_id":"shiti08143a80c8f7a4760786e10b308cb5bd4f4d",
    "source":3,
    "subject_id":2,
    "info":[

    ],
    "combine_content":"下列说法正确的个数有（  ）\n①2是8的立方根； ②±4是64的立方根； ③无限小数都是无理数； ④带根号的数都是无理数．\nA．1个    B．2个    C．3个    D．4个\n",
    "sub_question":[
        [
            {
                "question":"1",
                "solution":{
                    "step":"1",
                    "title":"判断每个选项的正确性",
                    "detail":"①2是8的立方根，正确；②±4是64的立方根，错误，因为±4是16的平方根，而±4的立方是±64；③无限小数都是无理数，错误，因为有的无限小数是循环小数，是有理数；④带根号的数都是无理数，错误，因为有的带根号的数可以化简为有理数，例如$\\sqrt{4}=2$。"
                }
            },
            {
                "question":"1",
                "solution":{
                    "step":"2",
                    "title":"统计正确选项的个数",
                    "detail":"因此，正确的选项只有1个，即①。"
                }
            }
        ]
    ]
}
```

## 输出：

```json
{
    "kc":[
        {
            "1":[
                "开立方",
                "立方根的定义",
                "无理数的定义",
                "有理数的定义"
            ]
        },
        {
            "2":[

            ]
        }
    ],
    "reason":[
        {
            "1":{
                "立方根的定义":"首先，我们需要知道什么是立方根。如果一个数a的三次幂等于另一个数b，那么我们就说a是b的立方根。例如在选项①中，“2是8的立方根”是正确的，因为2^3=8。",
                "开立方":"然后在选项②中，“±4是64的立方根”这一说法就错误了, 因为64开立方等于4，不包含-4。",
                "无理数的定义":"无理数指不能表示为两个整数比值形式（即分子/分母）或者小数部分既非有限也非循环周期性质持续不断地数字序列。在选项③中，“无限小数都是无理数”，这种说法错误，因为虽然所有无理数字都可以写作非终止、非循环小数组合形式，在某些情况下（如0.33333...），但并不意味着所有非终止、非循环小数组合形式必须都属于无理数字类别(例如1/3=0.33333... 是一个有限循环小数组合)。",
                "有理数的定义":"有理数指可以表示成两整数之比或者可以表示成小于某给定正整数量级精度内被确定完全准确地数字序列。在选项④中，“带根号的数都是无理数”，也同样错误。虽然许多带平方式标记符号√或³√等复杂算术运算符号组合构成表达式常常出现在代表各种各样类型数据类别区间范围内包含大量数量级精度高超微差异化特性具备独特性质优良特色难以用常规方法求得准确值而只能近似估计处理结果处境困局问题，但并不意味着所有带有这些符号的数都是无理数。例如，根号下4等于2，2是一个有理数。"
            }
        },
        {
            "2":{

            }
        }
    ]
}
```