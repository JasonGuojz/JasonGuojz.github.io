## Biomedical Event Extraction Based on Knowledge-driven Tree-LSTM

### 专有名词

1. **Tree-Structured LSTM模型**：[参考1](https://zhuanlan.zhihu.com/p/76671510)、[参考2](https://zhuanlan.zhihu.com/p/46651378)、[参考3](https://zhuanlan.zhihu.com/p/26261371)
2.  recursive matching criterion ——Overview of genia event taskin bionlp shared task 2011. Jin-Dong Kim

### Abstract

**生物医学领域**的**事件提取 Event  extraction** 比一般新闻领域的事件提取更具挑战性，因为它需要**更广泛的领域特定知识 domain-specific  knowledge  的获取和对复杂上下文的更深入的理解**。为了更好地**编码上下文信息和外部背景知识**，我们提出了一种**新颖的知识库（KB）-驱动的树结构长短期记忆网络（Tree-LSTM）框架** novel  **knowledge  base**  (KB)-**driven**  tree-structured  long  short-term  memory  networks(**Tree-LSTM**)  framework,  ，整合了两种新类型的功能：

1. 依赖结构 dependency structures 以捕获广泛的上下文  wide contexts ；
2. 通过实体链接 entity linking 从外部本体获得实体属性（类型和类别描述 types  and  category  descriptions ）。(远程监督)

我们在  BioNLP shared task 和 Genia数据集 上评估了与我们的方法，并获得了最新的结果。此外，定量和定性 quantitative and qualitative 研究都证明了 Tree-LSTM 的发展以及生物医学事件提取的外部知识表示。

### 1  Introduction

**生物医学事件通常是指状态变化 change of status**，尤其是蛋白质或基因状态变化。事件提取的目的是从生物医学文本中识别 triggers 及其 arguments，然后将事件类型分配给每个trigger，并将角色分配给每个参数。

图1所示的句子

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200723111259746.png" alt="image-20200723111259746" style="zoom:80%;" />

(这句生物方面的句子 trigger有两个type： gene expression和positive regulation，***Tax*** 是 **gene expression**类型trigger 的 **Theme** argument，而在生物领域 gene expression既可以是trigger type也可以是positive regulation trigger type 的argument)

它包含了一个**基因表达 gene expression** 和一个 正向调节**positive regulation** 事件提及，两者都由 ***transduced*** 触发。***Tax*** 是 **gene expression** 的 **Theme** argument。一个事件也可以作为另一个事件的 argument ， 导致嵌套结构——举例：the **gene expression** event **triggered** by ***transduced*** is also a **Theme argument** of the **positive regulation** event。

一个事件也可以用作另一个事件的参数，从而导致嵌套结构。例如，由转导触发的基因表达事件也是正调控事件的表象，如图1所示。

早期的事件提取研究依赖于**核方法**，例如支持向量机（SVM），这些方法使用手工特征。最近**基于分布表示 distributional   representation   based** 的方法（Rao等人，2017;Bj̈orne和Salakoski，2018）探索了仅需要分布式语义特征distributed  semantic  features 的深度神经网络。但是，与**一般新闻领域中的事件提取不同，生物医学事件提取需要广泛获取领域特定的知识和对复杂上下文的深刻理解**——$\color{red}{[}$  *例如，在 BioNLP shared task 2011的 Genia 事件提取中（Kim等，2011），大约80％的实体提及是基因，蛋白质和疾病的缩写 abbreviations of genes, proteins and diseases，而**超过36％的事件 triggers 和 参数 arguments 之间有10个以上的单词，距离很长。*** $\color{red}{]}$

为了有效地从**广泛的上下文中**获取**指示性信息indicative information**，我们首先采用基于 **Tree-LSTM 网络**。**依存句法树 Dependency tree structure** 可以**连接语义相关的概念semantically related concepts**，从而显着**缩短trigger 与its arguments之间的距离**。例如，在下面的句子“ ...，*which* **binds** *to the enhancer A located in the promoter of the mouse MHC class I gene* **H-2Kb**,，……”中，在确定 **binds** 的触发类型时，我们需要仔细选择其 **上下文词contextual words**，例如  **H-2Kb** ，表示 是 **binds**  的对象。但是，**binds** 和 **H-2Kb** 中间隔了16个words，**这对链式的LSTM来说很难捕捉到这种长距离的依赖关系**，而在**依存树结构中，它们的距离显着缩短到7**。  

此外，为了更好地捕获 特定领域的知识 domain-specific knowledge，我们还建议**利用 leverage 外部知识库 external knowledge bases（KBs）**来**获取所有生物医学实体biomedical entities的属性 properties**。KB的属性对于我们的模型更明确地学习**模式**非常有益。以图1中的 ***Tax*** 实体为例，它是一种通常参与**基因转录正调控positive regulation of transcription**的生物学过程的**蛋白质**，**根据 基因本体论referred to Gene Ontology**（Ashburner等，2000）——**外部知识**。此功能描述为确定***transduced*** as ***positive regulation***提供了关键线索。  (转导和转录)

因此，为了从外部知识库中获取此类知识，

1. 对于每个实体 entity，我们首先从其属性中学习 在 KB 层面上的 嵌入a KB concept embedding
2. 然后通过 **a gate function** 自动将 **embedding**  **并入** 其 **Tree-LSTM** 的 **hidden state** 

### 2  KB-driven Tree-LSTM for Event Extraction

#### 2.2  Constructing KB Concept Embedding

对于**生物医学事件提取**，我们主要将 ***Gene Ontology*** 作为外部 ***KB***，它提供了**所有物种中每个基因和基因产物属性的详细描述**。它包含两种类型的信息：

（1）**the gene ontology（GO）**定义了：

1. 所有基因功能 the gene functions，
2. 这些基因功能之间的关系，relations  between  these  gene  functions
3. 用于描述基因功能的概念 aspects used  to  describe  the  gene  function，包括分子功能 molecular  function, ，细胞成分 cellular  component  和生物过程 biological process.。

（2）**基因产物注释(GO  Anno)**   提供所有与实体相关的属性 entity  related  attributes，比如：实体全名   full entity name； 实体类型  entity type ； 与之相关的基因功能  相关gene functions 

 ***Gene Ontology*** 能提供的信息示例：

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200723202324522.png" alt="image-20200723202324522"    />

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200723202245300.png" alt="image-20200723202245300" style="zoom:80%;"   />

**为了利用 *Gene Ontology*  的信息**：

1.  apply  **QuickGO  API** ：将**每个 entity mention** 链接到 **Gene Ontology** link each entity mention to the Gene Ontology 并检索所有知识库 KB 注释。对于每个实体，我们仔细选择 **两种** 类型的**对事件提取任务有利**的**属性**： the **entity type** (e.g.,**protein** for **tax**) ；**the gene ontology function** it is related to (e.g.,***positive regulation of transcription*** for **tax**)。 **实体类型 entity type 可以促进显式模式学习 argument role 的 labeling任务**，例如基因表达事件模式。**基因本体功能 gene ontology function** 可以提供**隐含的线索**来确定 **trigger 类型**
2.  **KB Concept Embedding： **我们分配了一个**词嵌入词 word  embedding**，它在 **PubMed 和 PMCtexts** 上进行了**预训练**（Moen和Ananiadou，2013），以**表示每种实体类型 entity  type**。对于通常是一个**长短语 long phrase,的每个 gene  ontology  function**，我们使用 **state-of-the-art 的 sentence embedding approach**（Conneauet等人，2017）自动学习向量表示。然后，我们将这两种类型的 KB 属性表示形式连接起来，作为最终的 KB Concept Embedding。

#### 2.3  Event Trigger Extraction

KB concept embeddings 作为  domain-specific knowledge 放入  Tree-LSTM

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724102218772.png" alt="image-20200724102218772" style="zoom:80%;" />

A 中展示了 Tree-LSTM 的一个 unit， 为了得到 input token $x_j$ 的 隐状态 hidden state $h_j$ ，这个 unit  通过深度优先遍历 计算包含了它的所有孩子节点的 hidden state $h_{i-1}, h_{i-2}$。

给了上图中的句子：

1. perform  the  **dependency  parsing**  with  the  **Stanford  dependency parser** ，obtain  a  dependency tree structure

2. 对于树结构中的 节点 j ，$C(j)$ 是节点 j 的所有 子节点集合， $\mu_k$ 是节点 k 的KB concept embedding，$h_k$ 是 节点k 的 hidden state。 当 节点 k not a biomedical entity 设置 $\mu_k$ 为0，$\color{red}{\tilde\mu_j  = \sum_{k\in C(j)}\mu_k \  \ \ \ \ \ \ \ \tilde h_j  = \sum_{k\in C(j)}h_k}$   (和向量的感觉)

3. 我们将 **KB concept embedding 嵌入到 Tree-LSTM 的输入，遗忘和输出门**中  to select useful KB information implicitly,：

   <img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724115046407.png" alt="image-20200724115046407" style="zoom:80%;" />

   *这里嵌入，相当于是原sentense中和 node j 有依赖关系的一个子node的 embedding代表的是node j在外部资料库中的资料，相当于原文句子中加上了一个括号，括号中对这个word做了解释*

4. 此外，**引入knowledge specific output gate $g_j$ 显示地将 KB concept embedding 加到每个node  的 hidden state中**，与Maet al(2018)不同，他仅考虑每个节点本身的knowledge concept embedding，我们**使用整个子树(推测是自身加子节点？)的knowledge concept embedding的总和来代替**：

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724141846765.png" alt="image-20200724141846765" style="zoom:80%;" />

5. 前面算了在隐式添加KB知识后的各种门的参数，现在得到 Cell 的 state 的值 以及 添加显式 KB 知识的 输出 $h_j$

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724142105412.png" alt="image-20200724142105412" style="zoom:80%;" />

以上所有 $W$ 参数都是需要学习的:

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724142934532.png" alt="image-20200724142934532" style="zoom:80%;" />



#### 2.4  Event Argument Role Labeling 

在检测到所有 candidate triggers 之后，我们**进一步为每个 triggers 提取 arguments** 。 **Genia event extraction shared task 提供了所有实体提及 entity mentions 的注释**。因此，对于每个trigger，我们**使用在同一句子中出现的所有*实体提及作为 trigger 的候选参数***，然后**分配一个  argument role or None** 。与trigger提取 trigger  extraction, 不同，我们**在依赖关系树结构中使用最短依赖路径shortest  dependency path（SDP）代替表面上下文 surface contexts** ，以更好地捕获 trigger 和每个参数 arguments 之间的依赖关系。

以下图的句子为例，给定一个 trigger : ***transcription***和一个候选参数 argument:  ***OBF-1***

我们首先执行 **dependency parsing: ** ，**用 Dijkstra 算法** 得到 transcription 和 OBF-1 间的的**最短依赖路径**，**transcription → of → gene → OBF-1**。我们使用与第2.3节中介绍的相同的  ***KB-driven Tree-LSTM*** 将**每个节点编码为 hidden state 表示**。我们使用根节点的隐藏状态 h0 作为整个依赖路径的整体矢量表示。最后，我们**将 h0 和 hidden state of the trigger and argument 的串联结果 输入 另一个 softmax 中，predict the argument role**。我们也通过 最小化负对数似然损失来优化模型。

![image-20200724104152082](Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724104152082.png)

### 3  Experiment

#### 3.1  Task Description

**The Genia Event Extraction task**  是 **BioNLP  Shared  Task  series** 中一个主任务

任务介绍：

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724145319796.png" alt="image-20200724145319796" style="zoom:80%;" />

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724145328270.png" alt="image-20200724145319796" style="zoom:80%;" /><img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724145338765.png" alt="image-20200724145319796" style="zoom:80%;" />

#### 3.2  Experimental Setup

我们在Genia 2011数据集上应用了 KB-driven  Tree-LSTM  模型。Genia数据集中的实体是手动注释的，并作为输入的一部分给出。

我们使用Genia任务组织者提供的官方在线工具 在测试集上评估结果。 根据先前的研究（Bj̈orneandSalakoski，2011; Venugopal等人，2014; Raoet等人，2017; Bj̈orne和Salakoski，2018），**we report scores obtained by the approximate span**（允许触发跨度 trigger spans  与黄金跨度 gold spans 有一个word 的 不同）。由于我们只关注于匹配的核心参数 arguments ，因此我们使用递归匹配标准  recursive matching criterion 评估，不需要为其他事件引用的事件匹配其他参数（Kim等，2011 Overview of genia event taskin bionlp shared task 2011.）。

**word  embedding  pretrained  on PubMed  and  PMC  texts**

#### 3.3  Results and Error Analysis

* 事先对比 **comparison：**  only using Tree-LSTM 和 a standard BiLSTM model 对比

  结果： Tree-LSTM  outperforms  the  BiLSTM  baseline  which indicates the power of **Tree-LSTM in dealing with long-distance dependency structure** **in biomedical literature**.  

* **incorporating external KB information: ** 2.12% F-score gain  comparing  to  Tree-LSTM

* **event  extraction**  results  from  the  **BioNLP  shared  task  using the  same  corpus**.

  <img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724151252001.png" alt="image-20200724151252001" style="zoom:80%;" />

我们注意到我们的方法在 **Simple** event  types 上获得了高分，但在 **Binding event** 和 **Regulation event types**上获得了相对低的分。我们分析结果后发现**Binding event** 有 **multiple arguments**

![image-20200724151508868](Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724151508868.png)

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724151742589.png" alt="image-20200724151742589" style="zoom:80%;" />

<img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724151753726.png" alt="image-20200724151753726" style="zoom:80%;" /><img src="Biomedical%20Event%20Extraction%20Based%20on%20Knowledge-driven%20Tree-LSTM/image-20200724151815746.png" alt="image-20200724151815746" style="zoom:80%;" />

#### 3.4  Effect of KB concepts

·

### 4  Related Work

### 5  Conclusions and Future Work