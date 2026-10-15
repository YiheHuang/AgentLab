# RAG 入门

RAG（Retrieval-Augmented Generation，检索增强生成）是在回答前，检索相关外部资料，把检索结果加入模型上下文的技术。检索结果可以作为 Tool Observation 返回给 Agent。

Chunking 是把长文档拆成适当大小的文本块。Top-K Retrieval 从所有文本块中选出最相关的 K 块。TF-IDF 根据词项在文档中的频率和稀有程度计算权重，余弦相似度衡量查询与文档块的匹配程度。

本课使用的是词法检索，不能保证匹配语义相近但用词不同的表达。后续可替换为 Embedding 检索或混合检索。
