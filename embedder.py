from __future__ import annotations

import os


class Embedder:
    def __init__(self, model_name: str = "local", hf_endpoint: str = ""):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        if model_name == "local":
            return

        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_name)

    def _load_model(self, name: str):
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name).to(self._device).eval()

    def _mean_pool(self, token_embeds, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeds * mask).sum(1) / mask.sum(1)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return DefaultEmbeddingFunction()(texts)

        import torch
        import torch.nn.functional as F
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        emb = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._model is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return len(DefaultEmbeddingFunction()([""])[0])
        return self._model.config.hidden_size
