# 10-414/714: Deep Learning Systems (2022) [ Tianqi Chen、Zico Kolter] 


## 相关链接

- [homepage](https://dlsyscourse.org/)
- [github](https://github.com/dlsyscourse)

关键注解请参考内部文档 [[docs](./docs/)]

---

## content
- [X] hw0
- [X] hw1
- [X] hw2 （ <adam的测试> 和 <最后的模型训练> 有点小问题。）
- [X] hw3
- [X] hw4 (还存在问题)
- public_notebooks
---

## 一些问题
hw4
- [ ] bug1：测试需要补充一些（lstm 、rnn 的 input_fea_num, 瞎写都没有错误），实现过程中错写导致的问题无法通过测试暴露出来。
- [ ] 最后几个测试的模型训练需要重写一遍，bug较多。
- [ ] 优化版本的cuda matmul 在 !python3 -m pytest -l -v -k "nn_conv_forward" 会挂一个case
- [ ] conv 实现再看看，前向过程简化为二维矩阵乘，后向过程还是一个卷积操作。pad、strides等操作极大增加公式复杂度。

整体
- [ ] ndarray的矩阵乘，无法直接对多维度情况扩展 operator(@)
        需要更新ops里 matmul的操作（不整了，一堆bug, 代码复杂度还上去了）
- [ ] ops.summation 不支持 keepdims
- [ ] ops.max、ops.logsumexp 要求支持多轴操作就很恶心。（设计丑陋，容易出bug）
---
## 课程总结：
1、作业设计打磨的不够充分，有时不能完全理解实现逻辑。

2、出bug后测试环节极度痛苦。因为代码前后的依赖关系，前文通过测试的代码也大概率存在问题。

3、走完整体流程收获也不少。对pytorch的autograd框架有了更深入的理解。常用的module、layer、optimizer也手动的实现了一遍。

4、public_notebooks里的专题值得多看几遍。介绍的一些技巧也很巧妙
- 内存中矩阵元素的strides寻址
- 卷积操作的矩阵乘实现
- 底层cuda(gpu)、c++(cpu)端的代码实现与优化

5、特别建议: 不要使用ipynb来debug，重载代码的问题，不确定性太多（痛苦）。

## 致谢
天奇和Zico Kolter

fbsh的 [[代码](https://github.com/fbsh/cmu-10714)] 实现很有参考价值




