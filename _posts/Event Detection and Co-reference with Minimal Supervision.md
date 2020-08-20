## Event detection and co-reference with minimal supervision

1. weakly supervised algorithm——i唯一监督信息是提供对于event type 的先验知识，限定了type 的feature vector
2. Event detection: constructing a **feature vector** using several instances of the type**,** and **measures similarity** with the **event vector(embedding)** of a text;
3. 解决co-reference problem: measures similarity between two event vector
4. similarity都是在语义层面上，两个vector间的相似度度量用的都是余弦相似性
5. Freebase
6. event表示：SRL 以谓词为中心(认为一般trigger为谓词)
7. 对事件向量的表示做修改，更强调谓词的作用，具体使用的现有的embedding方法看具体的效果，没有特别指定
8. 逆着用 event type classification过程——已知type的vector，用于Event Mention Detection
9. 共指co-reference：比较event vector 的相似度

### Abstract

**Uses a weakly supervised algorithm to tackle Event detection and co-reference problem**. 

In the article, both **event detection and co-reference problem** both are considered **similarity detection problems.** Using several instances of each type as event vectors of that type, calculating the similarity between the **new event vector** and **event vector of each type**, judging **whether the event belongs to that type based on similarity.**

### Introduction

<img src="E:\Users\Desktop\暑研文献\Event Detection and Co-reference with Minimal Supervision\image\image-20200308193457898.png" alt="image-20200308193457898" style="zoom:150%;" />

<center>the MSEP framwork. Event example is the only source of suppervison </center>  

* **Event detection**:   identifying whether an event in context is semantically related to a set of **events of a specific type.**
* **Co-reference problem**:   whether two event mentions are semantically similar enough to indicate that the author intends to refer to the same thing.

As for Event detection, more like **constructing a feature vector using several instances of the type**, and measures similarity with the event vector of a text; 

As for co-reference problem, **measures similarity between two event vector.**

* **How to represent an event**:  using SRL, different embedding  methods(ESA, BC, W2V, DEP) to convert event components  structured vector representation.
* **How to measure similarity**:  calculating the cosine distance.

####  Structured Vector Representation

 Event triggers are mostly predicates of sentences or clauses. Predicates roughly corresponds to event types. 

 Map SRL arguments to : action, agent*sub*, agent*obj* , location and time. 

![image-20200308195646493](E:\Users\Desktop\暑研文献\Event Detection and Co-reference with Minimal Supervision\image\image-20200308195646493.png)

#### Event Mention Detection

We define the ***event type representation*** as the **numerical average** of all vector representations corresponding to example events under that type.

similarity measurement: 

![image-20200308195931031](E:\Users\Desktop\暑研文献\Event Detection and Co-reference with Minimal Supervision\image\image-20200308195931031.png)

![image-20200308195955412](E:\Users\Desktop\暑研文献\Event Detection and Co-reference with Minimal Supervision\image\image-20200308195955412.png)

#### Event co-reference

We compare agent*sub* and agent*obj* . If none of the types for the aligned event arguments match, this pair is determined to be in conflflict. If the event argument is missing, we deem it compatible with any type.

we generate a "conflict set". 

For event not in "conflict set",

![image-20200308201247562](E:\Users\Desktop\暑研文献\Event Detection and Co-reference with Minimal Supervision\image\image-20200308201247562.png)

[reference——jue wang的blog]([https://juewang.me/posts/%5B2018.1.21%5DEvent-detection-and-co-referentce/](https://juewang.me/posts/[2018.1.21]Event-detection-and-co-referentce/))





### Introduction

#### 自然语言理解的一个重要方面涉及到对***事件及其之间的关系进行识别和分类***

#### **解决从少量的数据中学习复杂的模型的过拟合情况**

developing an event detection and co-reference system with minimal supervision,  in  the  form  of  **a  few  event  examples**. 

We view these tasks as **semantic similarity** problems between **event mentions** or **event mentions  and  an  ontology** **of  types**我们将这些任务视为事件提及或事件与类型本体之间的语义相似性问题

有助于使用大量域外数据，

模型利用文本的结构our semantic relatedness function exploits the structure of the text by making use of a semantic-role-labeling based representation of an event.

我们的语义相关性功能利用事件的基于语义角色标记的表示来利用文本的结构。

1. 理解事件还需要理解它们之间的关系Understanding events also necessitates  understanding  relations  among  them。关系有助于确定两个文本片段是否表示同一事件
2. the frame-based structure of events基于帧的事件结构必须解决多个耦合问题(multiple coupled problems)，这些问题很难单独研究。
3. 可能是更核心的问题：whether current set of events’ definitions is adequate 
4. 该领域当前的评估方法侧重于事件的有限领域 limited  domain of event
5. Consequently, this allows researchers to train supervised systems that are tailored to these sets of events and that overfit the small domain covered in the annotated data, rather than address the realistic problem of understanding events in text训练的模型是适合于带注释的数据所覆盖的小范围的监督系统
6. 根本上说，事件检测是关于识别上下文中的事件是否与特定类型的事件集有语义上的关联；事件共同引用是关于两个事件提及在语义上是否足够相似，以表明作者打算引用相同的内容。如果将事件检测和共引用表示为语义相关性问题，则可以对其进行扩展以处理更多类型，并且有可能跨领域进行概括。此外，这样做有助于我们使用大量数据，这些数据不属于现有带注释的事件集合，甚至不属于同一域。
7. 主要的问题是 a. how to represent events。 b.  how to model event similarity。 这些都很难，因为events have structure

#### 1   我们的框架：

1. which essentially requires no labeled data；Event examples 是唯一的监督来源，用于产生 Example vectors

2. 在实践中，为了将事件映射到事件本体(map an event mention to an event ontology)（作为与用户通信的方式），对于用户想要提取的每种类型，我们只需要用纯文本形式的几个事件示例即可。event  type 用几个example作为definition

3. 与标准无监督方法相比，我们的方法所做的假设要少得多less assumption，后者通常需要实例的集合和实例之间的近似相似性才能最终学习模型。

4. 给定事件类型定义，我们可以classify a single event into  a  provided  ontology 然后判断两者是否are co-referent

5. 从这个意义上讲，我们的方法类似于所谓的无数据分类dataless classification

6. 我们的方法基于两个关键思想：

   a.  首先，为了表示事件结构event structures，我们使用通用的名义general purpose nominal和语言语义角色标记verbial semantic role labeling representation（SRL）表示。这使我们能够开发事件的结构化表示。

   b. 我们将事件组件embed event component在保持结构的同时嵌入到多个语义空间中，在上下文，主题和句法层面上。 multiple semantic space at a contextual, topical, and syntactic levels.

   这些语义表示是从大量文本中得出的，其方式完全独立于手头的任务，并且用于表示事件提及和事件类型，来用于分类事件

   1). 这些语义空间的组合以及事件的结构化矢量表示，使我们能够直接确定候选事件的提及是否为有效事件。如果是，则是哪种类型。

   2). **且**在具有相同表示形式的情况下，我们可以评估事件相似性，并确定两个事件提及是否是共同引用，所以这个模型can also adapt to new domains without any training

7. A few event examples就是MSEP所需要的全部监督信息了。这些示例甚至一劳永逸地确定了需要确定的几个决策阈值，并用于我们评估的所有测试案例。

#### 专有词汇

1. event mentions
2. semantic-role-labeling based representation of an event
3. event  co-reference on **benchmark data sets**
4. transfer across domains 跨域转移
5. event co-reference  problem——determining  whether  two  snippets  of text represent the same event or not
6. dataless classification(Chang et al.,  2008;  Song and Roth,2014).
7. 通用的名义general purpose nominal
8. 语言语义角色标记verbial semantic role labeling representation（SRL）
9. ESA vector
10. seed-based event trigger labeling technique employed in Bronstein et al. (2015)
11. the  mention-pair  model  in  entity  co-reference (Ng and Cardie, 2002; Bengtson and Roth,2008;  Stoyanov   et  al.,  2010),
12. the  Illinois  Wikification  (Chengand  Roth,  2013)  tool 
13. Explicit  Semantic  Analysis(ESA), Brown Cluster (BC), Word2Vec (W2V) and Dependency-Based Word Embedding (DEP) 
14.  TF-IDF vector
15. event detection 即 **trigger  identification**

​    

### 2   The MSEP System

#### 2.1 Structured Vector Representation

1. 事件结构和句子结构存在相似性，事件触发器 Event  triggers主要是句子或从句 clauses的谓词Predicates。谓词可以消除歧义sense disambiguated

2. 事件自变量Event arguments主要是实体提及entity mentions或时间/空间自变量temporal/spatial argument。它们充当事件中的特定角色，类似于为谓词分配角色标签的SRL参数

3.  Illinois SRL **pre-process** the text。 good coverage of SRL predicates and arguments on event triggers and arguments

4. 评估事件触发器和事件参数event  triggers  and  event  arguments,的SRL覆盖率。   对于事件触发器，我们仅关注recall，因为我们希望事件提及检测模块vent mention detection能够过滤掉大多数非触发谓词。即使我们仅获得近似的事件参数，将其归类为五个抽象角色也比确定事件触发器的确切角色标签更容易，更可靠。

5. 我们确定了五个最重要的抽象事件语义成分：![image-20200707110825519](Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200707110825519.png)为了将SRL参数映射到这些事件参数，执行以下过程：

   a.  set predicates as actions,and preserve SRL negations for action;

   b.   set SRL subject as$agent_{sub}$

   c.   set SRL object and indirect object as $agent_{obj}$

   d.  set SRL spatial argument as event  location如果没有此类SRL标签，则将扫描该动作所属的句/句中的任何NER位置标签 NER location label。我们会根据NER信息设置位置set the location according to NER information如果它存在

   e.   We set the SRL temporal argument as event time.  如果没有这种 SRL  label, 则use the Illinois Temporal Expression Extractor  在事件的主句或从句中找到temporal argument

   f.  允许$agent_{sub}、agent_{obj}$、location or time missing， 但是action一定有

   **在2.3中我们将每个事件分量event  component转换为其相应的向量vector  representation表示形式**

6. 我们将action所在的主句或从句作为上下文信息，append its corresponding vector to the event representation。   对missing 的event arguments, we set the corresponding  vector  to  be  “NIL”  (we  set  each  position as “NaN”）
7. 我们还通过连接更多的文本片段来增强事件向量表示，以增强动作和其他自变量之间的交互![image-20200707115835192](Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200707115835192.png)Here, “+” means that we first put text fragments together and then convert the combined text fragment into an ESA vector
8. 本质上，我们将事件结构展宽以保留事件参数的对齐方式，以便结构化信息可以反映在向量空间中。

#### 2.2    Event Mention Detection

1. 我们将针对每个事件类型标签下描述的事件示例使用ACE注释准则turn to ACE annotation guidelines for event examples described under each event type label 即在ACE的guidelines event types下找event examples
2. 则event  type的表示可以用the numerical average of all vector representations corresponding to example events under that type.
3. 我们使用事件候选者event candidate与事件类型event type表示之间的相似性来确定候选者是否属于事件类型![image-20200707184516462](Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200707184516462.png)请注意，可能缺少事件参数 event  argument（NIL）。在这种情况下，我们使用该特定组件的所有非NIL相似度分数的平均值作为贡献分数

4. 这些平均贡献分数独立于语料库，可以提前进行预先计算。我们使用临界值确定事件不属于任何事件类型，因此可以将其消除。此阈值仅通过调整事件示例集来设置，该事件示例与语料库无关

#### 2.3    Event Co-reference

1. 在应用共同引用模型之前，我们首先使用外部知识库来识别冲突事件，**the  Illinois  Wikification tool ** link  event  arguments  to Wikipedia pages
2. 使用Wikipedia ID，我们将事件参数event arguments映射到Freebase条目。我们将顶层Freebase类型视为事件参数类型event argument type。一个事件参数可以包含多个wiki的实体entities，从而导致多个Wikipedia页面以及一组Freebase类型。我们还使用NER标签扩充了参数类型集argument type set：PER（人）和ORG（组织）。如果我们检测到这样的命名实体，则添加NER标签之一
3. conflict:      we check event arguments $agent_{sub}、agent_{obj}$ respectively.如果对齐的事件参数的类型都不匹配，则确定 this pair该对存在冲突。 如果缺少事件参数，我们认为它与任何类型兼容。。通过检查是否有conflict，得到一个$Set_{conflict}$，里面的元素间不会有共指链接 co-reference links
4. left-linking greedy algorithm————for performing  event  co-reference  inference：：![image-20200707223550299](Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200707223550299.png)
5. 当$S(e_p,e_{k+1})$ event-pair similarity大于一个阈值时，make  co-reference  links。这个阈值是tuned only on event examples ahead of time。这里的$e_{k+1}$相当于共指聚类的聚类中心，当$e_{k+1}$和$e\in\{e_1,e_2,e_3,...,e_k\}$所有的event的相似度都低于阈值时，将$e_{k+1}$作为一个新的聚类中心的

### 3.   Vector Representations

测试了包括 Explicit  Semantic  Analysis(ESA), Brown Cluster (BC), Word2Vec (W2V) and Dependency-Based Word Embedding (DEP)  分别将文本转换为矢量。然后，我们将事件的所有组成部分串联在一起，以形成结构化的向量表示形式。

#### 3.1 Explicit  Semantic  Analysis ESA显式语义分析

使用Wikipedia作为外部知识库为给定的文本片段生成概念generate  concepts for a given fragment of text （Gabrilovich和Markovitch，2009年）。ESA首先将给定的文本片段表示为TF-IDF向量，然后对每个单词使用一个反向索引来搜索Wikipedia语料库。因此，文本片段表示是对应于其单词的概念向量的加权组合。我们使用与Chang等人（2008）中相同的设置来过滤掉少于100个单词的页面和少于5个超链接的页面。为了在ESA表示的有效性和成本之间取得平衡，我们使用权重最高的200个概念。因此，我们将每个文本片段转换为数以百万计的非常稀疏的矢量（但我们仅存储200个非零值）

#### 3.2  Brown Cluster  

support abstraction in NLP tasks，测量单词的分布相似度measuring  words’  distributional  similarities。 This method generates a hierarchical tree of word clusters by evaluating the word co-occurrence based on a  n-gram  mode

**use the implementation by Song and Roth (2014) ** generated over the latest Wikipedia dump转储.使用长度为4,6和10的路径前缀的组合作为我们的BC表示。Thus, we convert each word to a vector of $2^4+2^6+2^{10}=1104$dimensions

#### 3.3  Word2Vec

 the skip-gram tool by Mikolov etal. (2013) over the latest Wikipedia dump, resulting in word vectors of dimensionality 200.

#### 3.4  Dependency-Based Embedding

DEP is the generalization of the skip-gram model with negative sampling to include arbitrary contexts.  In particular, it deals with dependency-based contexts, and produces markedly different embeddings. DEP exhibits more functional similarity than the original skip-gram embeddings (Levy and Goldberg,  2014).  We directly use the released 300-dimension word embeddings

请注意，对于ESA，这是直接的文本向量转换。但是对于BC，W2V和DEP，我们首先从文本中remove stop words，然后对所有剩余的词向量进行元素平均，以生成文本碎片的结果向量表示。

### 4  Experiments

#### 4.1   Datasets

**ACE** ACE-2005英语语料库（NIST，2005）包含细粒度的事件注释event  annotations，包括事件触发器，参数，实体和时间戳记注释event trigger, argument, entity, and time-stamp annotations。我们从新闻报道中选择40个文档进行事件检测评估，其余用于培训（与Chen等人（2015年）相同）。我们对事件共同引用进行了十折交叉验证

**TAC-KBP**   TAC-KBP-2015语料库带有事件块event  nuggets ，分为38种类型，并且事件之间具有共同引用关系.我们使用官方TAC提供的训练/测试数据拆分

#### 4.2  Compared Systems

#### 4.3  Evaluation Metrics

对于事件检测，我们使用标准精度，召回率和F1指标。对于事件共指event  co-reference，我们使用标准F1指标比较所有系统。  ![image-20200708000900237](Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200708000900237.png)我们使用这四个指标的平均得分（AVERAGE）作为主要比较指标。

#### 4.4 result

在事件检测上，SSED  achieves  state-of-the-art  performance。。。SEED和MSEP都是使用SRL谓词作为输入，因此可以使用更好的SRL模块进一步改进。

#### 4.5  Results for Event Co-reference



关于如何进行Semantic Role Labeling，先引用juewang博客中的内容[引用]([https://juewang.me/posts/%5B2018.1.21%5DEvent-detection-and-co-referentce/](https://juewang.me/posts/[2018.1.21]Event-detection-and-co-referentce/))

<img src="Event%20Detection%20and%20Co-reference%20with%20Minimal%20Supervision/image-20200810113906464.png" alt="image-20200810113906464" style="zoom:80%;" />

