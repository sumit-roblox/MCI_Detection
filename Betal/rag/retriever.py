"""
RAG Retriever Module.

Bridges the Medical Knowledge Graph with the LangChain retrieval interface.
At inference time this module:

  1. Accepts a patient query (free text or patient_id).
  2. Queries the Neo4j KG for relevant medical context.
  3. Returns a list of LangChain ``Document`` objects ready for any
     LangChain chain (e.g., RetrievalQA).

If LangChain is not installed the retriever still works in standalone mode,
returning plain KGQueryResult objects.

Usage
-----
    kg = MedicalKnowledgeGraph(...)
    kg.connect()

    retriever = MedicalGraphRetriever(kg)
    docs = retriever.retrieve("What biomarkers are associated with MCI?")

    # Or use as a LangChain retriever:
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_langchain_retriever())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rag.knowledge_graph import KGQueryResult, MedicalKnowledgeGraph

logger = logging.getLogger(__name__)

# Optional LangChain import — graceful degradation
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    logger.info("LangChain not installed. Standalone retrieval mode active.")


@dataclass
class RetrievedContext:
    """Plain-Python fallback when LangChain is not available."""
    query: str
    context_text: str
    entities: list[dict[str, Any]] = field(default_factory=list)


class MedicalGraphRetriever:
    """
    Retriever that queries the Medical Knowledge Graph and returns context
    suitable for augmenting multimodal inference or an LLM chain.
    """

    def __init__(self, kg_client: MedicalKnowledgeGraph, max_results: int = 5) -> None:
        """
        Args:
            kg_client   : Connected MedicalKnowledgeGraph instance.
            max_results : Maximum number of related entities to include in context.
        """
        self.kg = kg_client
        self.max_results = max_results

    # ── Primary retrieval interface ──────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievedContext | list["Document"]:
        """
        Retrieve medical context from the Knowledge Graph for a given query.

        If LangChain is installed, returns a list of ``Document`` objects.
        Otherwise returns a ``RetrievedContext`` dataclass.

        Parameters
        ----------
        query : Free-text medical question, symptom, or biomarker name.
        """
        # Extract the key concept to query the graph (simple keyword extraction)
        concept = self._extract_concept(query)
        kg_result: KGQueryResult = self.kg.query_concept(concept)

        if _LANGCHAIN_AVAILABLE:
            return self._to_langchain_documents(kg_result, query)
        else:
            return RetrievedContext(
                query=query,
                context_text=kg_result.context_text,
                entities=kg_result.related_entities[: self.max_results],
            )

    def retrieve_for_patient(self, patient_id: str) -> RetrievedContext | list["Document"]:
        """
        Retrieve patient-specific context from the Knowledge Graph.

        Parameters
        ----------
        patient_id : Unique patient identifier stored in the Neo4j Patient node.
        """
        patient_ctx = self.kg.query_patient_context(patient_id)

        if patient_ctx is None:
            context_text = f"No patient-specific graph context found for ID: {patient_id}"
        else:
            diagnoses = ", ".join(patient_ctx.get("diagnoses", []) or ["unknown"])
            biomarkers = ", ".join(patient_ctx.get("biomarkers", []) or ["none recorded"])
            symptoms = ", ".join(patient_ctx.get("symptoms", []) or ["none recorded"])
            mmse = patient_ctx.get("mmse_score", "N/A")

            context_text = (
                f"Patient {patient_id}: MMSE Score={mmse}. "
                f"Diagnoses: {diagnoses}. "
                f"Biomarkers: {biomarkers}. "
                f"Symptoms: {symptoms}."
            )

        result = RetrievedContext(query=patient_id, context_text=context_text)

        if _LANGCHAIN_AVAILABLE:
            return [Document(page_content=context_text, metadata={"source": "neo4j", "patient_id": patient_id})]

        return result

    # ── LangChain compatibility ──────────────────────────────────────────────

    def as_langchain_retriever(self) -> "LangChainKGRetriever | None":
        """
        Return a LangChain-compatible BaseRetriever wrapping this class.
        Returns None if LangChain is not installed.
        """
        if not _LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not installed — cannot create LangChain retriever.")
            return None

        return LangChainKGRetriever(graph_retriever=self)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_concept(query: str) -> str:
        """
        Lightweight concept extraction: picks the first noun-like token from
        the query.  Replace with a proper NER model for production.
        """
        keywords = [
            "MCI", "Alzheimer", "Amyloid", "Tau", "APOE", "hippocampal",
            "memory", "biomarker", "cognitive", "dementia",
        ]
        query_lower = query.lower()
        for kw in keywords:
            if kw.lower() in query_lower:
                return kw
        # Fallback: use the whole query (KG does a CONTAINS match)
        return query.strip()

    def _to_langchain_documents(
        self, kg_result: KGQueryResult, original_query: str
    ) -> list["Document"]:
        """Convert a KGQueryResult into a list of LangChain Document objects."""
        docs = []

        # Primary context document
        docs.append(
            Document(
                page_content=kg_result.context_text,
                metadata={
                    "source": "neo4j_knowledge_graph",
                    "concept": kg_result.concept,
                    "query": original_query,
                },
            )
        )

        # One Document per related entity (up to max_results)
        for entity in kg_result.related_entities[: self.max_results]:
            entity_text = (
                f"{entity.get('entity', '')} [{entity.get('entity_type', '')}] "
                f"is related to {entity.get('related_entity', '')} "
                f"via [{entity.get('relationship', '')}]."
            )
            docs.append(
                Document(
                    page_content=entity_text,
                    metadata={
                        "source": "neo4j_knowledge_graph",
                        "entity_type": entity.get("entity_type", ""),
                    },
                )
            )

        return docs


# ── LangChain BaseRetriever subclass ────────────────────────────────────────

if _LANGCHAIN_AVAILABLE:

    class LangChainKGRetriever(BaseRetriever):
        """
        LangChain-compatible retriever that wraps MedicalGraphRetriever.

        Plug directly into any LangChain chain:
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=lc_retriever)
        """

        # LangChain Pydantic models require class-level field declarations
        graph_retriever: MedicalGraphRetriever

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: "CallbackManagerForRetrieverRun",
        ) -> list["Document"]:
            return self.graph_retriever.retrieve(query)
