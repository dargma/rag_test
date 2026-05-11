"""
SBERT (sentence-transformers) embedding model — for fair comparison with HippoRAG2/RAPTOR.
Wraps sentence_transformers.SentenceTransformer for safe loading
(BGE-style AutoModel loading triggers CUDA assert with SBERT vocabs).
"""
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class SBERTEmbeddingModel(BaseEmbeddingModel):
    """sentence-transformers based embedder. Compatible with mpnet/MiniLM/etc."""

    def __init__(self, global_config: Optional[BaseConfig] = None,
                 embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name

        self._init_embedding_config()

        logger.info(f"[SBERT] loading {self.embedding_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        # Match BGEEmbeddingModel API
        self.tokenizer = self.embedding_model.tokenizer
        # alias `encode` to internal _encode (BGE wraps with cache; we skip cache for simplicity)
        self.encode = self._encode

    def _init_embedding_config(self) -> None:
        # Provide minimal config compatible with BaseEmbeddingModel
        from .base import EmbeddingConfig
        cfg = EmbeddingConfig()
        cfg.embedding_model_name = self.embedding_model_name
        cfg.model_init_params = {}
        cfg.encode_params = {
            "max_length": 512,
            "instruction": "",
            "norm": True,
        }
        self.embedding_config = cfg

    def _encode(self, prompts: List[str], **kwargs) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        instruction = kwargs.get("instruction", "")
        if instruction:
            prompts = [instruction + text for text in prompts]
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                prompts,
                batch_size=kwargs.get("batch_size", 64),
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=kwargs.get("normalize", True),
            )
        return embeddings

    def batch_encode(self, texts, **kwargs):
        # Match BGEEmbeddingModel API: return NumPy array
        if isinstance(texts, str):
            texts = [texts]
        batch_size = kwargs.get("batch_size",
                                self.global_config.embedding_batch_size if self.global_config else 64)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            emb = self._encode(chunk, **kwargs)
            all_embs.append(emb.cpu())
        results = torch.cat(all_embs, dim=0)
        # Convert to NumPy (matches BGEEmbeddingModel)
        return results.numpy()
