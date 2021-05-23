# SFRL
**目前框架处于开发中，还有大量的工作需要做**
a simple framework for Reinforcement learning  
这是一个不依赖任何第三方库的纯c强化学习框架，功能比较简单，但可以清晰的展现一个深度学习框架从头开始是怎么搭建并运行起来的
# 起因
这个项目的起因是因为tf做并行化是一件比较麻烦的事情，要每个进程维护一个独立的tf上下文，并且本身python做并行化就很别扭  
爹有妈有不如自己有啊，为啥我不自己写一个框架呢？深度学习理论上的东西大家都很熟悉了但是真正落到工程实践上又是怎么样的呢？ 
正巧，之前阅读过darknet torch 和 caffe1的代码，对架构和细节差不多有个大概的把握 
那么好，为了弄清楚所有的细节问题，挑战一下自己，这里只用c语言，并且不依赖任何第三方库，所有需要的轮子都自己造一遍并且所有的流程和操作都会写详细的注释说明  
这样，框架写完，除了可以自己方便用之外，还可以巩固之前所有的理论知识，并提高工程能力
# 相关框架
深度学习的大框架，tf、pytorch、mxnet等就不说啦，专注强化学习的有deepmind的pixel和github上开源的天赐，前者不是狭义的框架，只是一个c++库，后者是python实现  
# 目前的进度
完善训练流程中。。。
