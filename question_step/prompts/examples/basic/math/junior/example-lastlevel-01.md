## 输入：

```json
{
    "question_id": "2b4b253803a742e6b830e1ced5963107",
    "source": 1,
    "subject_id": 2,
    "info": "[]",
    "combine_content": "如图，$$CD//AB$$，$$CB$$平分$$\\angle ACD$$，$$CF$$平分$$\\angle ACG$$，$$\\angle BAC=40{}^\\circ $$，$$\\angle 1=\\angle 2$$，则下列结论：①$$CB\\bot CF$$；②$$\\angle 1=70{}^\\circ $$；③$$\\angle ACE=2\\angle 4$$；④$$\\angle 3=2\\angle 4$$，其中正确的是（   ）．\n<img alt="" data="https://taltools-cdn.speiyou.com/ggb/tiku/data/ggb_classic_/4b750d02-562f-4f3d-8476-d8fde0fbfc0e.ggb" height="123" src="https://taltools-cdn.speiyou.com/ggb/tiku/img/ggb_classic_/4b750d02-562f-4f3d-8476-d8fde0fbfc0e.svg" width="302" />\nA.①②③\n\nB.①②④\n\nC.②③④\n\nD.①②③④\n\n",
    "sub_question": [
        [
            {
                "question": "1",
                "solution": {
                    "step": "1",
                    "title": "利用角平分线性质",
                    "detail": "因为$BC$平分$\\angle ACD$，$CF$平分$\\angle ACG$，所以$\\angle ACB=\\frac{1}{2}\\angle ACD$，$\\angle ACF=\\frac{1}{2}\\angle ACG$。",
                },
            },
            {
                "question": "1",
                "solution": {
                    "step": "2",
                    "title": "计算角度和",
                    "detail": "因此，$\\angle ACB+\\angle ACF=\\frac{1}{2}\\angle ACD+\\frac{1}{2}\\angle ACG=\\frac{1}{2}(\\angle ACD+\\angle ACG)=\\frac{1}{2}\\times 180{}^\\circ =90{}^\\circ$。所以，$CB\\bot CF$，结论①正确。",
                },
            },
            {
                "question": "1",
                "solution": {
                    "step": "3",
                    "title": "利用平行线性质",
                    "detail": "因为$CD//AB$，$\\angle BAC=40{}^\\circ$，所以$\\angle ACG=\\angle BAC=40{}^\\circ$，所以$\\angle ACF=\\angle 4=20{}^\\circ$。",
                },
            },
            {
                "question": "1",
                "solution": {
                    "step": "4",
                    "title": "计算角度",
                    "detail": "所以，$\\angle 2=\\angle BCD=180{}^\\circ -\\angle BCF-\\angle 4=180{}^\\circ -90{}^\\circ -20{}^\\circ =70{}^\\circ$，所以$\\angle 1=\\angle 2=70{}^\\circ$，结论②正确。",
                },
            },
            {
                "question": "1",
                "solution": {
                    "step": "5",
                    "title": "计算角度",
                    "detail": "因为$\\angle ECG=\\angle 1=70{}^\\circ$，所以$\\angle ACE=\\angle ECG-2\\angle 4=70{}^\\circ -40{}^\\circ =30{}^\\circ$，结论③不正确。",
                },
            },
            {
                "question": "1",
                "solution": {
                    "step": "6",
                    "title": "计算角度",
                    "detail": "因为$\\angle ECF=\\angle ACE+\\angle ACF=30{}^\\circ +20{}^\\circ =50{}^\\circ$，所以$\\angle 3=\\angle BCF-\\angle ECF=90{}^\\circ -50{}^\\circ =40{}^\\circ =2\\angle 4$，结论④正确。",
                },
            },
            {
                "question": "1",
                "solution": {"step": "7", "title": "选择正确选项", "detail": "综上，所以选择B选项"},
            },
        ]
    ],
}
```

## 输出：

```json
[
  {
    "1": ["角分线的性质"]
  },
  {
    "2": ["邻补角的定义与性质", "角的和差", "垂直的定义"]
  },
  {
    "3": ["平行线的性质", "内错角的定义","角分线的性质"]
  },
  {
    "4": ["平行线的性质","邻补角的定义与性质", "角的和差","内错角的定义"]
  },
  {
    "5": ["角的和差", "平行线的性质", "内错角的定义"]
  },
  {
    "6": ["角的和差", "余角的定义与性质"]
  },
  {
    "7": []
  }
]
```