"""
SimpleRAG —— 最小可行 RAG 实现，用于理解检索增强生成的核心流程。

完整管线：
  Document → Chunks → Embeddings → VectorStore → 用户查询 → 检索 → LLM 生成

设计理念：
  - 零外部依赖（只用 numpy + 标准库）
  - 每步都可独立调试，清楚看到数据形态
  - 与项目现有体系无缝集成：可作为 Tool 注册，供 ReActAgent 调用

用法：
  rag = SimpleRAG(chunk_size=300)
  rag.load_text("knowledge.txt")
  result = rag.query("远程办公的优缺点有哪些？")
"""

import json
import os
import re
import hashlib
from typing import Any


# ═══════════════════ 第1步：文档加载 ═══════════════════

class DocumentLoader:
    """从文件或字符串加载文档。

    支持：纯文本、目录批量加载、JSONL 格式。
    """

    @staticmethod
    def from_file(filepath: str) -> str:
        """读取单个文本文件，返回全部内容。"""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def from_directory(dirpath: str) -> list[dict[str, str]]:
        """批量读取目录下所有 .txt 文件，返回 [{filename, content}, ...]。"""
        docs = []
        for fname in sorted(os.listdir(dirpath)):
            if fname.endswith((".txt", ".md")):
                with open(os.path.join(dirpath, fname), "r", encoding="utf-8") as f:
                    docs.append({"source": fname, "content": f.read()})
        return docs

    @staticmethod
    def from_jsonl(filepath: str) -> list[dict[str, str]]:
        """读取 JSONL 文件（每行一条 JSON），content 字段为正文。"""
        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                docs.append({"source": obj.get("source", ""), "content": obj["content"]})
        return docs


# ═══════════════════ 第2步：文本分块 ═══════════════════

class TextChunker:
    """将长文本切分为语义相对完整的小块。

    为什么分块？
      - LLM 上下文窗口有限，不能一次塞入整本书
      - 小块语义更聚焦，检索精度更高
      - 只需要把「相关」的块传给 LLM，而不是全部内容
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        参数：
            chunk_size: 每块最大字数
            overlap:   区块间重叠字数（避免关键信息刚好落在断点处）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """按句子边界分块，优先在句号处断开。"""
        # 先按句子分割
        sentences = re.split(r"(?<=[。！？\.\!\?])\s*", text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current.strip())
                # 新块带 overlap：保留上一块尾部
                current = current[-self.overlap:] + sent if self.overlap > 0 else sent

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def chunk_documents(self, docs: list[dict[str, str]]) -> list[dict[str, Any]]:
        """批量分块多篇文档，每块保留来源信息。

        返回：
            [{"source": "file1.txt", "chunk_id": 0, "content": "..."}, ...]
        """
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "content": chunk,
                })
        return all_chunks


# ═══════════════════ 第3步：嵌入向量 ═══════════════════

class TFIDFEmbedder:
    """轻量级 TF-IDF 向量化：每个文档块变成一个固定维度的稀疏向量。

    完整方案会调用 Embedding API（如 text-embedding-3-small），
    但教育目的下用 TF-IDF 可以清楚看到向量化的每一步。

    核心公式：
      TF(i, d)  = 词 i 在文档 d 中出现的次数
      IDF(i)    = log(总文档数 / 包含词 i 的文档数)
      TF-IDF(i, d) = TF(i, d) × IDF(i)
    """

    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}  # word → index
        self.idf: list[float] = []            # 每个词的 IDF 值

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：中文按字+词组，英文按空格。"""
        # 中文字符直接作为 token
        tokens = []
        # 提取中文字符
        chinese = re.findall(r"[\u4e00-\u9fff]+", text)
        for word in chinese:
            # 中文按单字切分
            tokens.extend(list(word))
        # 提取英文单词
        english = re.findall(r"[a-zA-Z]+", text)
        tokens.extend([w.lower() for w in english])
        return tokens

    def fit(self, documents: list[str]):
        """在所有文档上构建词表，计算 IDF。"""
        doc_count = len(documents)
        word_doc_count: dict[str, int] = {}
        word_freq: dict[str, int] = {}

        # 统计每个词出现的文档数
        for doc in documents:
            tokens = set(self._tokenize(doc))  # 同一文档中同词只计1次（IDF）
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
            for token in tokens:
                word_doc_count[token] = word_doc_count.get(token, 0) + 1

        # 按词频排序，取 top-N 作为词表
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: i for i, (word, _) in enumerate(sorted_words[:self.max_features])}

        # 计算 IDF: log(总文档数 / 出现该词的文档数)
        import math
        self.idf = [0.0] * len(self.vocabulary)
        for word, idx in self.vocabulary.items():
            doc_freq = word_doc_count.get(word, 1)
            self.idf[idx] = math.log((doc_count + 1) / (doc_freq + 1)) + 1

    def encode(self, text: str) -> list[float]:
        """将单个文本转为 TF-IDF 向量。"""
        tokens = self._tokenize(text)
        vector = [0.0] * len(self.vocabulary)

        # 计算 TF
        token_count: dict[str, int] = {}
        for t in tokens:
            token_count[t] = token_count.get(t, 0) + 1

        # TF × IDF
        for token, count in token_count.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens) if tokens else 0
                vector[idx] = tf * self.idf[idx]

        return vector

    def encode_batch(self, documents: list[str]) -> list[list[float]]:
        """批量编码，返回二维向量列表。"""
        return [self.encode(doc) for doc in documents]


# ═══════════════════ 第4步：向量检索 ═══════════════════

class VectorStore:
    """向量存储与相似度检索。

    核心操作：
      1. 将所有 chunk 向量化后存储
      2. 用户查询同样向量化
      3. 用余弦相似度找到最相关的 Top-K 个 chunk

    余弦相似度公式：
      cos(a, b) = a·b / (|a| × |b|)
      值域 [-1, 1]，越接近 1 越相似
    """

    def __init__(self, embedder: TFIDFEmbedder):
        self.embedder = embedder
        self.chunks: list[dict[str, Any]] = []          # 块元信息
        self.vectors: list[list[float]] = []             # 对应的向量

    def add_chunks(self, chunks: list[dict[str, Any]]):
        """将文本块向量化后存储。"""
        texts = [c["content"] for c in chunks]
        vectors = self.embedder.encode_batch(texts)
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """余弦相似度检索 Top-K 最相关 chunk。

        返回：
            [{"source": "...", "chunk_id": 0, "content": "...", "score": 0.85}, ...]
        """
        import math

        query_vec = self.embedder.encode(query)

        # 计算 query 与每个 chunk 的余弦相似度
        scores = []
        for i, chunk_vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(query_vec, chunk_vec))
            norm_a = math.sqrt(sum(a * a for a in query_vec))
            norm_b = math.sqrt(sum(b * b for b in chunk_vec))
            cos_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            scores.append((i, cos_sim))

        # 得分降序排列，取 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(scores))

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                **self.chunks[idx],
                "score": round(score, 4),
            })
        return results


# ═══════════════════ 第5步：RAG 管线 ═══════════════════

class SimpleRAG:
    """完整的 RAG 管线 —— 加载文档 → 分块 → 建索引 → 查询。

    与 LLM 的交互方式：
      1. search(query)   → 返回相关 chunk（供外部 Tool 调用，ReActAgent 使用）
      2. query(question) → 完整流程：检索 + 组装 prompt（需传入 LLM 客户端）

    用法：
      rag = SimpleRAG(chunk_size=300, top_k=3)
      rag.load_from_file("knowledge.txt")        # 加载文档，自动分块+建索引
      results = rag.search("远程办公的优缺点")    # 纯检索，返回 chunk
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 3):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = TFIDFEmbedder()
        self.vector_store = VectorStore(self.embedder)
        self.top_k = top_k
        self._indexed = False

    # ---------- 数据加载 ----------

    def load_from_file(self, filepath: str):
        """从单个文件加载文本，分块，建索引。"""
        text = DocumentLoader.from_file(filepath)
        docs = [{"source": os.path.basename(filepath), "content": text}]
        chunks = self.chunker.chunk_documents(docs)
        self._fit_and_index(chunks)

    def load_from_directory(self, dirpath: str):
        """从目录批量加载 .txt/.md 文件。"""
        docs = DocumentLoader.from_directory(dirpath)
        chunks = self.chunker.chunk_documents(docs)
        self._fit_and_index(chunks)

    def load_from_jsonl(self, filepath: str):
        """从 JSONL 文件加载结构化文档。"""
        docs = DocumentLoader.from_jsonl(filepath)
        chunks = self.chunker.chunk_documents(docs)
        self._fit_and_index(chunks)

    def load_from_texts(self, texts: list[str], source: str = "inline"):
        """直接从字符串列表加载（无需文件）。"""
        docs = [{"source": f"{source}_{i}", "content": t} for i, t in enumerate(texts)]
        chunks = self.chunker.chunk_documents(docs)
        self._fit_and_index(chunks)

    def _fit_and_index(self, chunks: list[dict[str, Any]]):
        """在所有块上训练 TF-IDF 词表，然后存入向量库。"""
        texts = [c["content"] for c in chunks]
        self.embedder.fit(texts)
        self.vector_store.add_chunks(chunks)
        self._indexed = True

    # ---------- 检索 ----------

    def search(self, query: str) -> list[dict[str, Any]]:
        """检索 Top-K 相关文档块。

        返回格式：
        [
          {
            "source": "knowledge.txt",
            "chunk_id": 3,
            "content": "远程办公可以减少企业30%的办公空间支出...",
            "score": 0.82
          },
          ...
        ]
        """
        if not self._indexed:
            return []
        return self.vector_store.search(query, top_k=self.top_k)

    def retrieve_context(self, query: str) -> str:
        """检索后格式化为可直接注入 prompt 的上下文文本。"""
        results = self.search(query)
        if not results:
            return ""

        parts = ["\n--- 检索到的相关资料 ---"]
        for r in results:
            parts.append(f"[来源: {r['source']} | 相关度: {r['score']}]")
            parts.append(r["content"])
            parts.append("")
        return "\n".join(parts)

    # ---------- 统计 ----------

    def stats(self) -> dict:
        """返回索引统计信息。"""
        return {
            "chunk_count": len(self.vector_store.chunks),
            "vocab_size": len(self.embedder.vocabulary),
            "top_k": self.top_k,
            "chunk_size": self.chunker.chunk_size,
            "indexed": self._indexed,
        }
