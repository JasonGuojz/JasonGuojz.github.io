## Open Domain Event Extraction from Twitter

1. Open Domain
2. 根据**[Named entity recognition in tweets: An experimental study.]**中的模型 twitter文本有较多OOV(用Brown Cluster对lexical variations产生的OOV进行聚类来解决OOV)，manually annotated tweets as in-domain data 用于train POS system(特殊的词性标注集)，CRF(包含Brown Cluster的feature函数以及以及POS dictionary，spelling and contextual feature)，来完成POS tag任务。 named entity segmentation和classification任务的具体细节见文章
3. event phrase extraction——即EE任务中的对event mention的提取，用 linear chain CRF model作为序列标注任务的模型，识别trigger(这里的CRF用到的特征函数包括 contextual, dictionary, and orthographic features, and also include features based on our Twitter-tuned POS tagger [Named entity recognition in tweets: An experimental study], and dictionaries of event terms gathered from WordNet by Sauri et al.[Evita: a robust event recognizer for QA systems.])
4. temporal expression提取，即event argument ——TempEx
5. event type classification——因为推文open domain性质：首先，目前还不清楚哪些类别适合Twitter(?)。其次，使用事件类型标记tweet需要大量的手工工作。第三，重要类别(和实体)的集合很可能会随着时间的推移而改变，或者在特定的用户群体中改变。最后，许多重要的类别是相对少见的，所以即使一个大的带注释的数据集可能只包含这些类别的几个例子，使分类困难。——无监督方法，生成模型，使用主题模型中的LDA方法，该模型推断出一组适当的事件类型来匹配我们的数据，并通过利用大量的未标记数据将事件分类为不同的类型。
6. 推断——LDA方法的inference，gibbs sampling

### code

https://github.com/aritter/twitter_nlp

### 专有名词

1. recent work on NLP in noisy text

2. sequence labeling model

3. semi-supervised approaches 

4. latent variable models

5. POS tagged pos标记

6. NLP tool 命名实体分割器和旨在处理经编辑文本（例如新闻文章）的部分语音标记器

7. TEMPORAL EXPRESSIONS 解决时间表达式

8. Event indicator phrase 

9. LinkLDA ![image-20200705193641839](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200705193641839.png)提及事件短语的类型分布的信息可以在提及的内容中共享，而歧义性也可以自然保留。该方法基于生成的概率模型，对数据执行许多不同的概率查询很简单。例如，在对Ag-Aggregate事件进行分类时，这很有用。

10. 为了进行推断(inference)，我们使用折叠(collapsed)的Gibbs采样: 其中依次对每个隐藏变量进行采样，并对参数进行积分。

11. 折叠(collapsed)的Gibbs采样(sampling)

12. 流方法(a streaming approach)进行推理以预测

13. Gibbs markov链

14. symmetric Dirichlet distribution对称狄利克雷分布

15. Bayesian Inference techniques 贝叶斯推理技术

16. 流推论技术 streaming inference techniques

17. $G^2$ log likelihood ratio statistic，对于text分析比卡方更有效

    Fisher’s Exact test   ——— produce P values 但对于大数据量，计算Fisher’s Exact test statistic困难。。。。在我们的环境中，G2测试足够好用，因为计算实体和日期之间的关联比使用成对的实体（或单词）时产生的稀疏列联表(sparse contingency table)要少。

    ![image-20200705205639212](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200705205639212.png)

18. 远程监督 distant -supervision

19.  Locally Sensitive Hash Functions 

### challenge：

1. 大多数tweets是琐事
2. 没有足够信息将其归类到一个category的event
3. tweets的格式不是正式的书写方式，传统的NLP不适合

### Opportunities:

1. NLP in noisy text
2. 找到事件的总体表示，为**事件分类**提供额外的上下文
3. **识别那些和唯一时间相关的重要事件，而不是在一段时间内均匀发生的**
4. 如何找到重要事件的集合
5. 监督和半监督需要manually annotate。提出一种自动找type，然后filter和annotate with label的方法，用找到的类别对extracted event进行分类。
6. 主要的优点是使用了大量无标签的数据

### Representation

extracts a 4-tuple representation of events which includes **a named entity, event phrase, unambiguous calendar date, and event type** 

![image-20200308135214464](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200308135214464-1596944096144.png)

选择这种表达方式是为了与Twitter中重要事件的典型表达方式紧密匹配，目的是找到Twitter中重要event并以这种格式记录。

### Model 框图

![image-20200809113907373](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809113907373.png)

open domain下做 命名实体提取NER、事件提取EE和时间表达式解析Temporal Resolution。对提取的Event进行分类，根据significance进行排序。

分类依据：

significance排序依据:  根据每个date中出现的次数决定the strengthen association between each named entity and date

### NLP tool

1. named entity segmenters 
2.  PoS tagger

因为推文的形式不规范，没有可靠的特征描述，所以 ENR tool 是在 **in-domain Twitter data**重新训练的

### EXTRACTING EVENT MENTIONS

1. 手动 annotate a corpus of tweets 用于 train **sequence models** to extract event

2. 对推文用现成的 PoS tagger (chain CRF, which is beneficial for extracting multi-word event phrase)打上PoS tag

3. Event phrases can consist of many different 词性(event mentions?) 

   ![image-20200809152426429](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809152426429.png)

4. 手动注释的细节：

   ![image-20200809152646060](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809152646060.png)

5. CRF方法，定义feature function的细节：

   ![image-20200809154338160](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809154338160.png)

### EXTRACTING AND RESOLVING TEM-PORAL EXPRESSIONS

时间表达的抽取和解析

use **TempEx**

直接用在推文中出现的描述date 的text上准确率也很高——直接使用的TempEx，pipelined system

 input a reference date, some text, and parts of speech (from our Twitter-trained POS tagger) and marks temporal expressions with unambiguous calendar references.(**designed for use on newswire text, future work adapting temporal extraction to Twitter** )

### CLASSIFICATION OF EVENT TYPES

将事件类型 event type 作为 latent variables $z$ ，推断一组适当的事件类型以匹配我们的数据，并通过利用大量未标记的数据将事件分类为各种类型  (用的是 LinkLDA ，主题模型中的一种，主题为隐变量，为文档预测主题 )

![image-20200809165055337](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809165055337.png)

unsupervised approaches that will automatically induce event types which match the data.

Each type corresponds to a distribution over named entities n involved in specific instances of the type, in addition to a distribution over dates d on which events of the type occur. 

每个type有一个该类型特定实例的命名实体的分布，和该类型事件发生的日期 的 分布。

### Inference

![image-20200809165151047](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809165151047.png)

![image-20200809165207521](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200809165207521.png)

**pipelined system**

### Algorithm

#### 算法细节：

1. 这些推文经过POS标记，然后提取命名的实体和事件短语，解析时间表达式，并将提取的事件分类。最后，我们根据它们共同出现的推文的数量来测量每个命名实体与日期之间的关联强度，以便确定事件是否重要。

2. NLP tool：我们利用named entity segmenters 和PoS tagger， ENR tool是根据先前工作[46]中介绍的域内Twitter数据进行训练的。我们还针对域内带注释的数据开发了一个事件标记器

3. 我们使用命名实体tagger对先前工作中提出的域内Twitter数据进行了培训[46] .3对tweet进行培训可以大大提高对命名实体进行分段的性能。 域内训练的重要性

5. event phrase提供了重要的上下文信息，提供的信息也有助于categorizing event into types

7. 我们使用条件随机场（conditional random filed)进行学习learning 和推理inference，将事件触发的识别问题视为序列标记任务sequence labeling task[24

8. 线性链CRF(linear chain CRF)对相邻单词(adjacent words)的预测标签(predicted label)之间的相关性(dependencies)进行建模，这有助于提取多单词事件短语(extract multi-word event phrase)

9. 我们使用上下文(contextual)，字典(dictionary)和正字法(orthographic)特征，还包括基于Twitter调整的POS标记器[46]的功能，以及Sauri等人[50]从WordNet收集的事件项的词典。

10. 除了提取事件和相关的命名实体之外，我们还需要在事件发生时进行提取。

11. 解决时间表达式：resolve temporal expression。 使用TempEx——输入参考日期，一些文本和词性（来自我们的Twitter训练的POS标记器），并使用明确的日历参考标记时间表达。 a reference date, some text, and parts of speech(from our Twitter-trained POS tagger)TempEx在推文上的高精度可以用以下事实来解释，即某些时间表达相对明确 TempEx’s high precisionon Tweets can be explained by the fact that some temporal expressions are relatively unambiguous.尽管通过处理嘈杂的时间表达似乎有改善Twitter上时间提取（例如，“明天”一词的50多个拼写变化列表，请参见Ritter等人[46]），但我们保留了适应时间的提取到Twitter作为潜在的未来工作（for example see Ritter et. al. [46] for a list of over50 spelling variations on the word “tomorrow”）

12. 为了将提取的事件分类为不同的类型，我们提出了一种基于潜在变量模型(atent variable models )的方法，该方法可以推断出适当的事件类型集以匹配我们的数据，并通过利用大量未标记的数据将事件分类为各种类型。 模型的来由 by recent work on modeling selectional preferences [47, 39, 22, 52, 48], and unsupervised information extraction [4, 55, 7].

13. **Supervised or semi-supervised classification of event categories存在一些问题：**

    **a. priori unclear不清楚哪个类别适用于Twitter。**

    **b. 用事件类型注释推文需要大量的人工。**

    **c. 重要的类别（和实体）可能会随着时间的推移而变化，或者随着关注的用户人群而变化。 important categories (and entities) is likely to shift over time**

    **d. 即使是大型带注释的数据集也可能仅包含这些重要类别的少量示例making classification difficult.**

12. 事件指示器短语event indicator phrase ，都被建模为类型的混合a mixture of types![image-20200705172439009](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200705172439009.png)每种type对应于该种type的特定实例中涉及到的命名实例的分布

13. 在我们的模型中包括日历日期，具有鼓励在同一日期发生的事件被分配为相同类型的作用。有助于指导推理，因为对同一事件的不同引用也应具有相同的类型.

16. 为了估计给定事件在类型上的分布，在充分 burn in(扔掉前几个迭代) 后从Gibbs markov链中获取相应的隐藏变量的样本。使用流方法(a streaming approach)进行推理以预测新数据
15. 用于注释自动发现的事件类型的完整标签列表以及每种类型的覆盖范围。请注意，将标签分配给类型仅需要完成一次，并为任意数量的事件实例生成标签。

#### 评价

1. 为了评估模型对重大事件进行分类的能力，收集， we gathered 65 million extracted events of the form 。We then ran **Gibbs Sampling** with 100 types for 1,000 iterations of burn-in, keeping the hidden variable assignments found in the last sample.

2. 可以使用流推论技术轻松地使用相同的类型集对新事件实例进行分类[efficient methods for **topic model inference** on streaming document collections.]。未来工作的一个有趣方向是自动标记和对自动发现的事件类型进行一致性评估，类似于主题模型的最新工作[38，25]。

3. 为了评估模型对**汇总事件**进行分类的能力(classify aggregate events,).我们将出现20倍或20倍以上数据的所有（实体，日期）对组合在一起，然后使用模型发现的**事件类型**对**关联性最高的500**（请参阅第7节）进行注释

4. 为了帮助证明利用大量未标记数据进行事件分类的好处.compare against a supervised **Maximum Entropy baseline** 最大熵模型基线  which makes use of the 500 annotated events using 10-fold crossv alidation

5. For features, we treat the set of event phrases that co-occur with each (entity, date) pair as a **bag-of-words**, and also include the associated entity. Because many event categories are infrequent, there are often few or no training examples for a category, leading to low performance

6. 仅使用频率来确定哪些事件是重大事件是不够的，因为许多推文都涉及用户日常生活中的常见事件。为了从Twitter提取一般兴趣的重大事件，我们需要某种方法来**衡量实体与日期之间的关联强度**。

7. 然后，我们将提取的三元组添加到第6节中所述的用于推断事件类型的数据集中，并对Gibbs采样执行了50次迭代，以预测新数据上的事件类型，同时使原始数据中的隐藏变量保持不变。。使用的是 streaming inference

8. 按照第7节中的描述对提取的事件进行排名，并从排名最高的100,500和1,000中随机抽取50个事件。我们用4个单独的标准注释了事件。

9. 注释事件 annotate the event

   标准：![image-20200705210901858](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200705210901858.png)

### Ranking Event

重要的事件可以被区分为那些与一个独特的日期有强烈关联的事件，而不是在日历中均匀地分布。为了从Twitter中提取一般意义上的重要事件，我们需要某种方法来衡量一个实体和一个日期之间的关联强度。

To **extract significant events of general interests** from Twitter, we thus need some way to **measure the strength of association between an entity and a date.**

用 G 检验 (异常值检验)

![image-20200308154402927](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200308154402927.png)

![image-20200308154656117](Open%20Domain%20Event%20Extraction%20from%20Twitter/image-20200308154656117.png)

#### 实验

1. 我们从每个100M的文本中提取了**事件短语**和时态表达式之外的命名实体。提取包括event phrases、temporal expressions 和named entities 
2.  predictions (“entity + date + event + type”)

#### 相关工作

1. 研究有两个与之相关的关键环节：从Twitter提取特定类型的事件，以及从新闻中提取开放域事件。



推文具有高度重复性，因此我们有动力集中精力提取事件的总体表示形式，从而为诸如事件分类之类的任务提供额外的上下文，并通过利用信息的重复性来过滤掉平凡的事件