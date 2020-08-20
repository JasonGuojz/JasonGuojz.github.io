## Event Detection with Neural Networks: A Rigorous Empirical Evaluation  

1. 文章仅包含了对 GRU输出 word 的 hidden state向量 ht 的具体改进，其余的设置需要看 JRNN及改进模型的文章[Joint Event Extraction via Recurrent Neural Networks] [Joint Extraction of Events and Entities within a Document Context]
2. Event detection is often framed as a multi-class classification problem
3. 本文借助DAG-GRU建模语法信息**syntactic  information**(当前大多数神经网络模型都忽略了文本中的syntactic relationships)与时间结构temporal  structure 结合在一起。单纯的GRU不能建模远距离的关系（两个token间远距离存在一条边）,某种程度上是一种图结构。本文借助Attention解决某个token上存在多条边的问题
4. 另一个贡献是“**对系统性能对模型初始化的敏感性进行了实证研究**”
5. word embedding 方法 用的 EMLo

### DAG-GRU model

The standard GRU model works as follows:

![image-20200309164528918](E:\Users\Desktop\暑研文献\Event Detection with Neural Networks A Rigorous Empirical Evaluation\image\image-20200309164528918.png)

**contribution:** **incorporates syntactic information through dependency parse relationships**,using **attention** to **combine syntactic and temporal information.** 

Think of the standard GRU as a graph structure. The standard GRU edges are included as *(t, t -1*, *temporal)*.



can't understand this part....

<img src="E:\Users\Desktop\暑研文献\Event Detection with Neural Networks A Rigorous Empirical Evaluation\image\image-20200309202056357.png" alt="image-20200309202056357" style="zoom:80%;" />

dependency relationship will produce a  dependency tree according to the rules.



By adding dependency relationship to GRU, the model may have problem in implementing back-propagation as it requires a directed acyclic graph(DAG).  

**Solution** split it into two parts

<img src="E:\Users\Desktop\暑研文献\Event Detection with Neural Networks A Rigorous Empirical Evaluation\image\image-20200309203031882.png" alt="image-20200309203031882" style="zoom:80%;" /> 

**Attention mechanism:**

Used to combine the multiple hidden vectors.  Traditional GRU is much like the LSTM, choosing to memorize ht or forget ht-1, as the author says, only the temporal edges. For event detection, involve the  dependency edge can add syntactic relationships of the text into the model. Then using attention to compute the weight according to all the edges link to the node.



## 神经网络的事件检测：严格的经验评估——Event Detection with Neural Networks: A Rigorous Empirical Evaluation

### 专有名词

1. **event detection**：包括识别表示事件的“触发”词 trigger 并将事件分类为精炼类型——事件检测是推断有关事件的更多语义信息inferring more semantic information about the event的必要的第一步，包括提取事件的参数arguments以及识别不同事件之间的时间和因果关系。
2. **ACE2005**用于精确定义precise definition任务和数据以进行评估for the purposes of evaluation.  它由599个文档组成，分为529training、30development和40testing——这个分割是事实上的评估标准————但是测试集很小并且仅包含newswire documents，然而在ACE2005里包含multiple domains。————These two factors lead to a significant difference between the training and testing event type distribution训练集和测试集的事件类型分布差异——虽然已经有跨域 across domains比较方法的工作（Nguyen和Grishman，2015年）——**variations  in  the training/test split including all the domains has not been studied.** ***(同样也是使用数据集训练的神经网络的局限性)***
3. DAG——directed acyclic graph有向无环图

### 摘要

检测事件events detection并将其分类为预定义predefined  types的类型是从自然语言文本natural  language  texts中提取知识 knowledge  extraction的重要步骤。**尽管神经网络模型通常领先于最新技术，但尚未严格研究不同体系结构different  architectures之间的性能差异**  

本文中提出**GRU-based mode** 通过注意机制attention mechanism将句法信息syntactic  information与时间结构temporal  structure 结合在一起。

我们通过在ACE2005数据集的不同随机初始化和训练-验证-测试拆分下的经验评估表明，它与其他神经网络体系结构具有竞争力。

### 1   简介

1. **问题：**

   a.     当前大多数神经网络模型都忽略了文本中的句法关系——syntactic relationships。

   b.   **event detection任务的其中一个挑战是the  size  and  sparsity  of  this  dataset。**

2. **提出的模型：**a new **DAG-GRU** architecture 通过具有依存解析关系dependency parse relationships 的文本的双向读取bidirectional reading of the text来捕获上下文和语法信息。通过注意力机制，通用化了GRU模型以**在图上**运行。

3. 我们评估模型准确率的敏感性，通过随机性研究，改变训练和测试集的划分

4. 鉴于与神经网络模型使用的和其他数据集相比训练集的数据量十分有限，并且许多高性能方法之间的裕度狭窄，因此需要考虑这些方法的不同初始化得到的不同效果。**在本文中，我们对系统性能对模型初始化的敏感性进行了实证研究。**

5. 所有方法的性能对随机模型初始化的敏感性比预期的要高。重要的是，基于标准训练-验证-测试拆分的性能的不同方法的排名有时与基于多个拆分的平均值的排名不同，这表明社区应远离单个拆分评估。

### 2   Related work

跟踪了Nguyen,  Kyunghyun的工作

<img src="Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200812162212731.png" alt="image-20200812162212731" style="zoom:80%;" />

在[Jointly Extracting Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16222/16157) 中给出一下的结构，通过增加一个 控制依存关系的门信号，引入依存关系，对输入到当前节点的有关隐层向量 $h_i$ 进行加权平均 

![image-20200812215240862](Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200812215240862.png)

### 3   DAG GRU Models

1. **event detection**通常被建模成multi-class classification problem——**所以event detection的目标**是预测测试文档中每个单词的事件标签event label ，NIL if the word is not an event，一个句子是n个word $x_1,\ ...x_n$ 的一个序列，每个word由一个k长度的向量表示

2. <img src="Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200812161254941.png" alt="image-20200812161254941" style="zoom:80%;" />

3. 标准的GRU model是<img src="Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200709224035732.png" alt="image-20200709224035732" style="zoom:67%;" />GRU模型通过将word $x_t$ 的表示representation与前一个的hidden vector $h_{t-1}$ 结合起来，生成当前的word的hidden vector $h_{t}$ 。这样，$h_{t}$概括了the word and its prior context。

4. 我们提出的DAG-GRU模型通过依存解析关系dependency  parse relationships 将句法信息syntactic  information  纳入其中，其实质类似于（Nguyenand Grishman，2018）和（Qian等人，2018————改进处在用attention机制把syntactic  information和temporal information结合之后与前一个$h_t$ 组合成新的hidden vector然后应用在标准GRU模型。

5. 在标准依存关系表达中加入了parent-child orientation  

   <img src="Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200812115313571.png" alt="image-20200812115313571" style="zoom:80%;" />

6. Each  relationship  表示成一条edge，$(t,t^\prime,e)$，即在index $t^\prime$ and $t^\prime$ 的word间有一条edge，类型是e——作为对比，标准的GRU的edge格式为 $(t,t-1,temporal)$——edge $e$  包含了dependency relationship 和 temporal 时间表达

7. 因为边(dependency relationship)可能存在在任意两个word间，所以生成的图可能包含cycle。但是，而BPTT需要有向无环图（DAG）——因此将由时间temporal和依存边dependency  parse relationships **E**  组成的句子图被分为两个DAG：“正向” DAG $G_f$ 包含的edge $(t,t^\prime,e)$ 只有 $t^\prime< t$ ；以及一个“反向” DAG $t^\prime> t$ ————$t,t^\prime$ 间的依存关系包含 parent-child orientation e.g., nsubj-parent or nsubj-child for an subj(subject) relation。

8. 注意力机制attention mechanism用于组合多个隐藏向量multiple hidden vectors————每个word都要经过system生成一个 $h_{a}$ 通过collecting and transforming所有先前的hidden  vectors 指向node t(当前这个word)，同一个hidden vertor有不同的edge type $e$ 就要多重复放入一次组成一个 input matrix，经过非线性的变换来形成矩阵 $D_t$ ——$D_t$ 和一个权重矩阵生成 $\alpha$ gives the attention, 分布加权在边缘的importance——————在标准的RNN模型下，the subject “members”将是遥不可及的，但是DAG-GRU模型可以通过依赖边缘和注意力集中在这一重要连接上via dependency edges and attention.

   <img src="Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200710000624298.png" alt="image-20200710000624298" style="zoom: 67%;" />

**DAG体现在哪**？ 类似于![image-20200812231236378](Event%20Detection%20with%20Neural%20Networks%20A%20Rigorous%20Empirical%20Evaluation/image-20200812231236378.png)

这个模型，分成正反两个方向

**对GRU的应用不同在哪**？经典的GRU输入间只有时序关系，现在在时序关系中加上句法关系

用 attention 结合 syntactic information(这里即dependency information) 和 temporal information，



