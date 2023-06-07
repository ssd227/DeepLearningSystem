# 10-414/714: Deep Learning Systems (2022) [ Tianqi Chen、Zico Kolter] 


## 相关链接

- [homepage](https://dlsyscourse.org/)
- [github](https://github.com/dlsyscourse)

关键注解请参考内部文档 [[docs](./docs/)]

---

## content
- [x] hw0
- [x] hw1
- [x] hw2 (logsumexp待整体处理) 
- [x] hw3 (cuda 矩阵乘法加速待完成)
- [ ] hw4
- public_notebooks
---

## todo list

hw2
- [ ] logsumexp 数值稳定版本的ops，前向后向推导。为什么非得搞这个算子，使得又要手动推导公式。麻烦。
- [ ] 为了实现logsumexp, 还需要实现tensor的 max、argmax ops来方便计算。后续可以拓展到pooling操作。
- [ ] hw2里adam的测试和最后的模型训练有点小问题。logsumexp算子待完善 

hw3
- [x] some warnings in pow(cpu-ops、gpu-ops)
测试里rand A, 存在负数不能开根号，nan warning（已修正）

- [ ] cuda 矩阵乘代码重写，就代码高并发不行。


hw4
- [ ] bug1：测试需要补充一些（lstm 、rnn 的 input_fea_num, 瞎写都没有错误），实现过程中错写导致的问题无法通过测试暴露出来。
---


## todo hold

hw4
- [x] Bug2：ndarray的矩阵乘，无法直接对多维度情况扩展 op[@]
        需要更新ops里 matmul的操作（不整了，一堆bug, 代码复杂度还上去了）

---
## 课程总结：
1、作业设计打磨的不够充分，写hw的过程中有时不能完全理解需要实现的逻辑。

2、出bug后的测试环节也极度痛苦，因为代码前后的依赖关系，有时前文中的code即时通过了测试，实现的代码也是有问题的。

3、整体作业流程走完了之后，收获也不少。对pytorch类型的autograd框架有了更深入的理解。常用的module、layer、optimizer也手动的实现了一遍。

4、尤其是public_notebooks里的专题notebook值得多看几遍原理。介绍的一些技巧也很巧妙
- 内存中矩阵元素的strides寻址
- 卷积操作的矩阵乘实现
- 底层cuda(gpu)、c++(cpu)端的代码实现与优化


## 致谢
fbsh的 [[代码](https://github.com/fbsh/cmu-10714)] 实现很有参考价值

再次感谢天奇和Zico Kolter


