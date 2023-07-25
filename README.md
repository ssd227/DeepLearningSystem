# 10-414/714: Deep Learning Systems (2022) [ Tianqi Chen、Zico Kolter] 


## 相关链接

- [homepage](https://dlsyscourse.org/)
- [github](https://github.com/dlsyscourse)

关键注解请参考内部文档 [[docs](./docs/)]

---

## content
- [X] hw0
- [X] hw1
- [X] hw2 (logsumexp待整体处理) 
- [X] hw3 (cuda 矩阵乘法加速待完成)
- [X] hw4
- public_notebooks
---

## 一些问题
hw2
- [ ] logsumexp 数值稳定版本的ops，前向后向推导。为什么非得搞这个算子，使得又要手动推导公式。麻烦。
- [ ] 为了实现logsumexp, 还需要实现tensor的 max、argmax ops来方便计算。后续可以拓展到pooling操作。
- [ ] hw2里 <adam的测试> 和 <最后的模型训练> 有点小问题。logsumexp算子待完善

hw4
- [ ] bug1：测试需要补充一些（lstm 、rnn 的 input_fea_num, 瞎写都没有错误），实现过程中错写导致的问题无法通过测试暴露出来。
---


## todo hold

hw4
- [x] Bug：ndarray的矩阵乘，无法直接对多维度情况扩展 operator(@)
        需要更新ops里 matmul的操作（不整了，一堆bug, 代码复杂度还上去了）

---
## 课程总结：
1、作业设计打磨的不够充分，有时不能完全理解实现逻辑。

2、出bug后测试环节极度痛苦。因为代码前后的依赖关系，前文通过测试的代码也大概率存在问题。

3、走完整体流程收获也不少。对pytorch的autograd框架有了更深入的理解。常用的module、layer、optimizer也手动的实现了一遍。

4、public_notebooks里的专题值得多看几遍。介绍的一些技巧也很巧妙
- 内存中矩阵元素的strides寻址
- 卷积操作的矩阵乘实现
- 底层cuda(gpu)、c++(cpu)端的代码实现与优化


## 致谢
天奇和Zico Kolter

fbsh的 [[代码](https://github.com/fbsh/cmu-10714)] 实现很有参考价值




