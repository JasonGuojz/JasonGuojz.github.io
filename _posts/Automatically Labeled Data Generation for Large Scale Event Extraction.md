## Automatically Labeled Data Generation for Large Scale Event Extraction——自动标记数据的生成用于大规模事件提取

1. 原本远程监督的三元组表示的是两个entity和两者间的relation，因此可以用三元组在大型语料库中提取包含这两个实体的句子，作为关系提取的训练数据。本文将远程监督用于事件提取。将argument 和 event 看作relation extraction中的两个"entity"，则可以将event extraction看作是多种或复杂multiple  or  complicated的关系数据relational  data，也就可以像DS for RE一样使用event  instance和argument来自动生成训练数据以进行argument identification(**role**)
2. 因为根据ACE2005，一个event instance是表示为a trigger word，所以文中也将event detction转换成 trigger detection问题
3. 做了两个关于 event instance、argument和 trigger(文中直接用verb) 的假设
4. For example, compared with arguments like time, location and so on, spouses are key arguments in a marriage event. We call these arguments as key arguments。用**Freebase**来 figure out **key arguments** of an event使用key arguments来标记事件并找出触发词trigger words
5. 因为一个event type 的所有argument被包含在一个sentence中的情况很少，因此选出一个key argument认为句子中如果出现这个key argument则这个句子很可能在表达这个key argument代表的event type，因此可以在这个sentence中提取trigger。设计了 KR(key rate)找key argument即一个argument在这个event type 的所有instance中出现的频率乘上与其他event type不相关的程度。用 key argument 来 label sentence，在label过的sentence中找trigger，设计了TR(trigger rate)来找 trigger (类似于 TF-IDF中的思想)
6. **FrameNet**——被用来 noisy trigger words and expand more triggers。因为只用verb作为trigger，则像marriage这种名词trigger就不包含，因此引入 linguistic KG FrameNet。计算framebass中event type下所有words 的word embedding的均值和framnet中的frame下的LUs的word embedding，比较两个向量的相似度来作为map的依据。如果map到framenet的frame后不包含verb trigger则作为noise去掉，如果words中的nounswith high confidence inthe mapped frame to expand trigger lexicon。
7. SDS 软远程监督：如果文档中的句子包含一个event type 的所有key argument和一个trigger，则用这个句子生成训练数据。
8. event extraction过程：生成训练数据后，做监督训练，使用的是《Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks》中的网络模型提取词汇级和句子级的特征，两个任务：对event进行分类，以及句子中包含的word做 argument role分类
9. 因为远程监督的noise影响，为了减少错误标签的影响，使用Multi-instance Learning训练，目标函数的定义也是以instance bag



### 专有名词

1. **KBP——Knowledge Base Population** KBP 公开任务的研究目标，是让机器可以自动从自然书写的非结构化文本中抽取实体，以及实体之间的关系。

   <img src="Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200813101927421.png" alt="image-20200813101927421" style="zoom:80%;" />

2. **Distant Supervision远程监督（DS）**：our automatically labeled data  face  a  noise  problem,  which  is  a  intrinsic problem  of  using  DS  to  construct  training  data。

3. DS for **标记 Relation Extraction（RE）的training data**

4. **知识库——knowledge bases **  

5. Dynamic Multi-pooling Convolutional Neural Network-s  **(DMCNNs)**： is the best reported CNN-based model for event extraction (Chen et al., 2015) by using human-annotated training data. 

6. **Multi-instance Learning (MIL)** 多实例学习

   [知乎参考1](https://zhuanlan.zhihu.com/p/40812750)

   [南大周志华教授 miVLAD and miFV,](http://www.lamda.nju.edu.cn/CH.Data.ashx?AspxAutoDetectCookieSupport=1#code)

**[Event Data](https://github.com/acl2017submission/event-data)**



### 摘要

​	用于诸如ACE之类任务的事件提取的现代模型都是基于监督学习的，该学习是从少量**手工标记**small hand-labeled的数据中进行的。**手工标记**small hand-labeled的数据————expensive，low  coverage of event types， limited in size——所以难以提取large scale of events  for **KBP**

​	因此，为了**解决数据标记问题**，我们提议通过使用world knowledge and linguistic knowledge语言知识 automatically label training data，which can detect key arguments and trigger words for each event type and employ  them  to  label  events  in  texts  auto matically.

​	而且我们的自动标记数据可以与人工标记数据结合在一起，然后提高从这些数据中学到的模型的性能

### 1  简介

1. 事件提取：检测和typing events，extracting  arguments  with  different  **roles**(Argument  Identification)

   ![image-20200713204833668](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200713204833668.png)

2. 到目前为止，大多数方法使用详尽的注释数据——manually labeling training data cost high、预定义事件类型的覆盖率低。

3. 因此，对于提取大规模事件，尤其是在开放域场景open domain scenarios，如何自动高效地生成足够的训练数据是一个重要的问题。

4. **本文旨在自动为EE生成训练数据，其中包括labeling triggers，event types，arguments及其role(Argument  Identification)。**

5. 事实证明，**Distant Supervision远程监督（DS）**的最近被证明可有效地**标记 Relation Extraction（RE）的training data**，**Relation Extraction（RE）**的目的是**predict 两个命名实体的语义关系semantics relation** between  pairs  of  entities。$(entity_1,relation,entity_2)$——————**DS for RE假设如果两个实体在已知的知识库中有关系have  a  relationship  in a known knowledge base，则所有提及这两个实体的句子将以某种方式表达这种关系。**

6. 当用DS for RE来做EE时，遇到以下的挑战

    **Triggers are not given out in existing knowledge bases**:EE aims to detect an event instance of a specific type and extract their arguments and roles, formulated as$ (event\ instance,event\ type;role1,argument_1;role_2,argument_2;...;role_n,argument_n)$——**这可以看作是多种或复杂multiple  or  complicated的关系数据relational  data**。在以an  example  of spouse of relation between Barack Obama and Michelle Obama, 

   ![image-20200713214520119](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200713214520119.png)

   中，**DS  for  RE**  uses  **two  entities**  to  automatically  **label  training  data**。像左边的图，中间的是event  instance，周围的矩形是arguments of the event instance，边指的是 the role of the argument——————看起来可以像DS for RE一样使用event  instance和argument来自动生成训练数据以进行argument identification(**role**)。————但是event instance是在text中隐式地提及，只是 a virtual node in existing knowledge bases————in  Freebase，前面提到的婚姻事件实例表示为$m.02nqglv$，所以不能直接拿event  instance  and  an  argumen来label  back in  sentences——————在ACE中，一个event instance是表示为a trigger word，最清楚地代表句子中事件发生的main word————受ACE启发，可以用trigger word来表示event instance，像$married$作为event instance  $people.marriage$  的trigger——但是，现有知识库 existing knowledgebases中未给出触发器

7. 为了解决trigger  missing  problem，我们需要在使用distant supervision来自动标记event argument之前**先去找trigger words**————根据上面**DS for RE**的假设，

   **下面做了很多关于语言学上的假设：**

   a.   我们自然地假设**a  sentence  contains all arguments of an event in the knowledge base tend to express that event, and the verbs occur in these sentences tend to evoke引发 this type of events**

   b.   However,**arguments for a specific event instance are usually mentioned in multiple sentences.** ——所以仅仅用知识库中所有的arguments来标注句子只能生成很少的training data——只有很少的event instances能在一个sentence中找到所有的argument![image-20200714161509083](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714161509083.png)

8.  因此我们提出用**world knowledge(Freebase)**和**linguistic knowledge (FrameNet)**来自动**generate labeled data** for **large scale EE**。 

   **a.**  首先，我们提出了一种使用Freebase为每个事件类型优先化参数prioritize arguments并选择key或代表参数(representative arguments)（请参阅第3.1节中的详细信息）

   **b.**  其次，我们仅使用key arguments来标记事件并找出触发词trigger words

   **c.**  第三，外部语言知识资源external linguistic knowledge  resource——FrameNet——被用来filter **noisy trigger words** and **expand more triggers**

   **d.**  之后，我们为EE提出了一个**软远程监督(SDS)**，以自动标记训练数据————SDS假设了**Freebase**中任何包含all key arguments和一个 corresponding trigger word的句子都可能以某种方式表达该事件，该句子中出现的arguments可能在该事件中play the corresponding roles 

   **e.**   我们通过手动和自动评估方法来评估自动标记的培训数据的质量

   **f.**    we employ a **CNN-based  EE  approach**  with  **multi-instance  learning**  在自动标记的data上作为baseline

#### 做出的三大贡献

1. the first work to automatically label data for large scale EE via world  knowledge  and  linguistic  knowledge
2. 用Freebase来 figure out **key arguments** of an event，并使用它们来自动检测events和相应的corresponding trigger words。并且用FrameNet来过滤noisy triggers并expand more triggers.
3. 大规模自动标记数据的质量有保证，可以扩充传统的人工注释数据，从而可以显着改善the extraction performance

### 2   背景

1. 使用Freebase作为包含event  instance的world  knowledge，以及使用FrameNet作为包含trigger information的linguistic knowledge。————Wikipedia中文章用作要标记的非结构化文本，
2. **Freebase**——是semantic  knowledge  base——利用复合值类型compound  value  type， CVTs，来将多个值合并为一个值——$people.marriage$  是CVT值中的一个type——这个type有很多的instance，比如 the marriage of $Barack\ Obama$ and $Michelle\ Obama$ is numbered as $m.02nqglv$ .  $Spouse,from,to\ and \ location\  of  \ ceremony$ 是 $people.marriage$ CVT type的**roles**
3. <img src="Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714173434756.png" alt="image-20200714173434756" style="zoom:50%;" /><img src="Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714173530661.png" alt="image-20200714173530661" style="zoom:50%;" />

### 3     Method of Generating Training Data

**Architecture**![image-20200714174457629](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714174457629.png)

#### 3.1  Key Argument Detection

**总结：**prioritizes 每个event type 的arguments，为each type of event选择key arguments——**如何通过FreeBase为每个event type找到key arguments**

1. **point：**arguments of a type of event play different roles——在区分不同类型的events时key arguments能作为线索——use **Key  Rate(KR)**  to estimate  the  **importance  of  an argument** to a type of event，由两个因素：角色显著性(Role Saliency)和事件相关性(Event Relevance)

2. **角色显著性(Role Saliency)**: 什么是显著性Saliency：If we tend to **use an argument** to **distinguish** one **event instance** from other instances of a given **event type**, this argument will play **a salient role** in the given event type.

   定义：![image-20200714194641930](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714194641930.png)

   其中$RS_{ij}$是第 i 个argument对第 j 个event type的role saliency值。$Count(A_i,ET_j)$ 是$arguement_i$ 在Freebase的$eventType_j$的所有实例中出现的次数，$Count(ET_j)$是Freebase中$eventType_j$的所有实例个数

   (候选的argument集如何选出的？)

3. **事件相关性(Event Relevance)**：反映一个argument 能被用来区分不同event type的能力

   定义：![image-20200714195554100](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714195554100.png)

   $Sum(ET)$是在Freebase中所有event types的个数和，$Count(ETC_i)$是所有包含第 i 个argument的event type的个数。

4.  **Key  Rate(KR)**的计算是

   ![image-20200714200858515](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714200858515.png)

5. 同**KR**值排序，选择前K个arguments作为key arguments。

#### 3.2  Trigger Word Detection

**总结：**使用key arguments来label可以初步表达事件的句子，然后detect  triggers

1. **通过key  arguments对在Wikipedia中可能表达event的句子进行label。**

   使用**Standford CoreNLP tool**将原始的Wikipedia文本转换为句子序列并附加 NLP annotation (POS tag, NER tag)，然后我们把包含一个Freebase中event实例所有arguments的句子选择出来，这个句子我i们认为是能够表达corresponding events. 最后，我们用这些labeled句子来detect triggers

2. 在句子中，一个verb更可能意味着一个event的出现(AES中，60%的event的trigger是verbs)——所以直观地说，在labeled sentences中一个verb出现的次数多于另外的verb，则这个verb更可能是trigger；但是像$is$这些verb作为trigger 的probability 要小。————这里用到$Trigger Candidate Frequency(TCF)$和$TriggerEvent Type Frequency(TETF)$

   ![image-20200714202823550](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714202823550.png)![image-20200714202832708](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714202832708.png)![image-20200714202846272](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714202846272.png)

   其中，$TR_{ij}$是第 i 个verb对第 j 个event type的trigger rate，$Count(V_i,ETS_j)$ 是表达第 j 个event type并且包含第 i 个verb的句子的个数，$Count(ETS_j)$是表达第 j 个event type的句子个数，$Count(ETI_i)$是number of event types, which have  the  labeled  sentences  containing i-th  verb————最后，我们选取有高TR值的verbs作为event type的trigger words

#### 3.3 Trigger Word Filtering and Expansion

**总结：**使用**linguistic resource——FrameNet to filter noisy verbal triggers and expand nominal triggers**

通过上面**3.2**步得到初始的trigger lexicon，但是这个 lexicon is noisy 并且只包含动词的 triggers，名词 triggers是缺失的，像marriage等。——为什么不用**TR值**来找？，因为一个句子中名词的数量通常大于动词。

1. 因为**word embedding **成功捕捉word的语义，使用word embedding来把Freebase中的event映射到 FrameNet中——将 $i-th$ Freebase event type中的所有词的word embedding取平均记为$e_i$ ， 记 k-th lexical units of j-th frame为$e_{j,k}$，选择具有最高$similarity(e_i,e_{k,j})$ 的frame为映射的frame![image-20200714204542697](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714204542697.png)
2. we filter the verb, which is in initial verbal trigger word lexicon and not in the mapping frame. And we use nouns with high confidence in the mapped frame to expand trigger lexicon

#### 3.4  Automatically labeled data generation

**总结：**uses a **SDS** 来生成训练数据

### 4  Method of Event Extraction

event  extraction被分为两个阶段：第一步：**Event  Classification**： predict  whether  the  **key  argument**  candidates  participate  in  a  Freebase  event.——如果是，则进行第二步**argument classification**：assign arguments  to  the  event  and  identify  their  corresponding  roles为事件分配参数并确定其相应角色

用**Distant Supervision远程监督（DS）**存在的问题：自动标注的训练数据存在噪声————为了减轻**wrong label problem**，采用**Multi-instance Learning (MIL)** 

用的模型是two similar Dynamic Multi-pooling Convolutional Neural Networks with Multi-instance Learning (DMCNNs-MIL)

以下阐述Multi-instance Learning (MIL) 用在**argument classification**的方法：

1. 定义在 DM-CNNs  训练的参数$\theta$

2. 假设有T bags${M_1,M_2,...,M_T}$，the $i-th$ bag包含$q_i$ 个实例instance $M_i={m_i^1,m_i^2,...,m_i^{q_i}}$

3. **Multi-instance Learning (MIL)**的目的是predict the labels of the unseen **bags**

4. **什么是bag**——we take **sentences** containing the same **argument candidate** and **triggers** with a **same event type** as a bag and all **sentences** in a bag  are  considered  independently. 我们采用包含相同自变量候选词和具有与bag相同的事件类型的触发器的句子，并且bag中的所有实例均认为是相互独立。

5. 输入实例$m_i^j$，参数$\theta$ 的网络的输出向量$O$ ，其中向量第r的元素指的是 argument role r对应的值，这里求极大似然值$p(r|m_i^j,\theta)$ ，这个值由softmax operation over all **argument role types**

   ![image-20200714230406990](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714230406990.png)

6. **区分 bags**，目标函数using cross-entropy at the bag leve：

   ![image-20200714231727305](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714231727305.png)

   ![image-20200714231750908](Automatically%20Labeled%20Data%20Generation%20for%20Large%20Scale%20Event%20Extraction/image-20200714231750908.png)

### 5    实验

在21个选择的Freebase中的events上自动生成大量的标记数据，选择最大的5个event。

通过grid search将两个超参（我们的自动数据标签中的key arguments的数量和TR的值）分别设置为2和0.8

1. 先手动评估自动标记的数据。
2. 基于ACE corpus对标记的数据进行自动评估
3. 展示了DMCNNs-MIL在自动标记数据上的性能。

以及讨论了key arguments数量对结果的影响