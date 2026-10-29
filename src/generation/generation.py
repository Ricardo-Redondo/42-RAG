from __future__ import annotations
from typing import TYPE_CHECKING

from src.models import MinimalAnswer, MinimalSearchResults, MinimalSource, \
    StudentSearchResults, StudentSearchResultsAndAnswer
from src.config import Config
from src.errors import RAGError
from dataclasses import dataclass, field
from tqdm import tqdm
import re
import time
import sys

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NO_SOURCES = "No relevant sources were found to answer this question."
NOT_IN_SOURCES = "The provided sources do not contain the answer."

SYSTEM_PROMPT = (
    "You are a technical assistant for the vLLM codebase. "
    "Answer the question using ONLY the numbered sources given by the user.\n"
    "Rules:\n"
    "1. Answer directly in 2 to 4 sentences that make sense on their own: "
    "name the subject of the question in your answer.\n"
    "2. After each fact, cite the number of the source it comes from in "
    "square brackets, like [1].\n"
    "3. Only state what the sources say. Never invent names, values, "
    "commands or code.\n"
    "4. Do not describe or list the sources, do not repeat the question.\n"
    f"5. Only if none of the sources answers the question, reply with this "
    f"sentence alone: \"{NOT_IN_SOURCES}\""
)

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Generator:
    config: Config
    tokenizer: PreTrainedTokenizerBase | None = field(default=None, init=False)
    llm: PreTrainedModel | None = field(default=None, init=False)
    files: dict[str, str] = field(default_factory=dict, init=False)

    def load(self) -> None:
        """Import transformers and load Qwen the first time (lazy)."""
        if self.tokenizer is None or self.llm is None:
            now = time.perf_counter()
            print(f"Loading {self.config.llm}...", file=sys.stderr)
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm)
                # float32: measured ~1.7x faster than bfloat16 on CPU
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.config.llm, dtype=torch.float32)
            except (OSError, ValueError) as e:
                raise RAGError(
                    f"cannot load model '{self.config.llm}': {e}") from e
            self.llm.eval()
            elapsed = time.perf_counter() - now
            print(f"Loaded in {elapsed:.1f}s", file=sys.stderr)

    def answer(
            self, search_results: StudentSearchResults
            ) -> StudentSearchResultsAndAnswer:
        """Generate one answer per search result (batch in, batch out)."""
        if any(r.retrieved_sources for r in search_results.search_results):
            self.load()
        answers: list[MinimalAnswer] = []
        for result in tqdm(search_results.search_results, desc="Answering",
                           file=sys.stderr,
                           disable=len(search_results.search_results) < 2):
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=self.generate(result),
            ))
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k)

    def generate(self, result: MinimalSearchResults) -> str:
        """Run Qwen on one question and its sources, return the answer."""
        if not result.retrieved_sources:
            return NO_SOURCES
        if self.tokenizer is None or self.llm is None:
            raise RuntimeError("model not loaded")
        import torch

        sources = self.select_context(result.retrieved_sources)
        prompt = self.tokenizer.apply_chat_template(
            self.build_messages(result.question, sources),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # PyTorch Tensor ex: {"input_ids": tensor([[151644, 8948, …]]),
        # "attention_mask": tensor([[1, 1, …]])}
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # size of prompt (remembering where prompt ends)
        prompt_length = inputs["input_ids"].shape[1]

        # inference mode = no training -> faster, less RAM
        with torch.inference_mode():
            output = self.llm.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                do_sample=False)

        # only the answer tokens (after prompt)
        new_tokens = output[0][prompt_length:]
        answer: str = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True)
        return self.add_citations(answer.strip(), sources)

    def select_context(
            self, retrieved: list[MinimalSource]
            ) -> list[tuple[MinimalSource, str]]:
        """
        Pick the sources (and their text) that go into the prompt.

        Best sources first, at most max_context_sources, and at most
        max_context_tokens in total: the last source is cut to fit.
        """
        if self.tokenizer is None:
            raise RuntimeError("model not loaded")

        budget = self.config.max_context_tokens
        selected: list[tuple[MinimalSource, str]] = []

        for source in retrieved[:self.config.max_context_sources]:
            ids = self.tokenizer(self.source_text(source))["input_ids"]
            if not ids:
                continue
            if len(ids) > budget:
                if budget < 50:     # too little room left to be useful
                    break
                ids = ids[:budget]
            selected.append((source, self.tokenizer.decode(ids)))
            budget -= len(ids)
            if budget <= 0:
                break
        return selected

    @staticmethod
    def build_messages(
            question: str,
            sources: list[tuple[MinimalSource, str]]
            ) -> list[dict[str, str]]:
        """Build the chat messages: rules + numbered sources + question."""
        context = "\n\n".join(
            f"[{i}] File: {source.file_path}\n{text}"
            for i, (source, text) in enumerate(sources, start=1)
        )
        user = f"Sources:\n\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def add_citations(
            answer: str, sources: list[tuple[MinimalSource, str]]) -> str:
        """
        Append the file paths behind the [n] markers the model used.

        A 0.6B model often forgets to cite: then every source it was given
        is listed, since the answer was generated from them.
        """
        # the model sometimes answers AND adds the refusal sentence:
        # keep the answer, drop the stray refusal
        answer = answer.replace(NOT_IN_SOURCES, "").strip()
        if not answer:
            return NOT_IN_SOURCES
        cited = sorted({int(n) for n in CITATION.findall(answer)
                        if 1 <= int(n) <= len(sources)})
        numbers = cited or list(range(1, len(sources) + 1))
        # one line per file: [1][2] path
        files: dict[str, list[int]] = {}
        for n in numbers:
            files.setdefault(sources[n - 1][0].file_path, []).append(n)
        lines = ["".join(f"[{n}]" for n in ns) + f" {path}"
                 for path, ns in files.items()]
        return answer + "\n\nSources:\n" + "\n".join(lines)

    def source_text(self, source: MinimalSource) -> str:
        """Read a source's text back from the corpus file (cached)."""
        if source.file_path not in self.files:
            try:
                with open(source.file_path, encoding="utf-8",
                          newline="") as f:
                    self.files[source.file_path] = f.read()
            except (OSError, UnicodeDecodeError):
                self.files[source.file_path] = ""
        content = self.files[source.file_path]
        return content[source.first_character_index:
                       source.last_character_index]
