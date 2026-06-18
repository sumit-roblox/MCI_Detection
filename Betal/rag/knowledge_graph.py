"""
Medical Knowledge Graph Module.

Connects to a Neo4j database containing a domain-specific schema for
Alzheimer's Disease and Mild Cognitive Impairment research.

Neo4j Schema (Cypher notation)
-------------------------------

Nodes:
  (:Patient   {patient_id, age, sex, mmse_score})
  (:Diagnosis {name, icd_code})          e.g. "Mild Cognitive Impairment", "J90.3"
  (:Biomarker {name, modality, unit})    e.g. "Amyloid-beta 42", "CSF", "pg/mL"
  (:Gene      {symbol, ensembl_id})      e.g. "APOE", "ENSG00000130203"
  (:Drug      {name, mechanism})         e.g. "Donepezil", "AChE inhibitor"
  (:Symptom   {name, category})          e.g. "Memory loss", "cognitive"

Relationships:
  (:Patient)-[:HAS_DIAGNOSIS]->(:Diagnosis)
  (:Patient)-[:EXHIBITS]->(:Symptom)
  (:Patient)-[:HAS_BIOMARKER {value}]->(:Biomarker)
  (:Diagnosis)-[:ASSOCIATED_WITH]->(:Biomarker)
  (:Diagnosis)-[:ASSOCIATED_WITH]->(:Gene)
  (:Diagnosis)-[:TREATED_BY]->(:Drug)

Usage
-----
    kg = MedicalKnowledgeGraph(uri=..., user=..., password=...)
    kg.connect()
    results = kg.query_concept("MCI")
    kg.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class KGQueryResult:
    """Structured result from a Knowledge Graph query."""
    concept: str
    related_entities: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""


class MedicalKnowledgeGraph:
    """
    Neo4j-backed medical knowledge graph for Alzheimer's / MCI domain.

    Gracefully degrades to offline mode if the Neo4j driver is not installed
    or the database is unreachable, allowing the rest of the pipeline to run
    without a live graph connection.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    # ── Connection management ────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish connection to Neo4j. Falls back silently if unavailable."""
        try:
            from neo4j import GraphDatabase  # Lazy import — optional dependency

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            logger.warning("neo4j Python driver not installed. Running in offline mode.")
        except Exception as exc:
            logger.warning(f"Could not connect to Neo4j ({exc}). Running in offline mode.")

    def close(self) -> None:
        """Close the Neo4j driver session."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> MedicalKnowledgeGraph:
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Schema initialisation ────────────────────────────────────────────────

    def initialize_schema(self) -> None:
        """
        Create indexes and seed the graph with core MCI / Alzheimer's concepts.
        Only needs to run once on a fresh database.
        """
        if self._driver is None:
            logger.warning("No active Neo4j connection. Schema init skipped.")
            return

        cypher_statements = [
            # Indexes for fast lookup
            "CREATE INDEX patient_id IF NOT EXISTS FOR (p:Patient) ON (p.patient_id)",
            "CREATE INDEX diagnosis_name IF NOT EXISTS FOR (d:Diagnosis) ON (d.name)",
            "CREATE INDEX biomarker_name IF NOT EXISTS FOR (b:Biomarker) ON (b.name)",

            # Seed diagnoses
            """MERGE (d:Diagnosis {icd_code: 'G31.84'})
               SET d.name = 'Mild Cognitive Impairment' """,
            """MERGE (d:Diagnosis {icd_code: 'G30.9'})
               SET d.name = "Alzheimer's Disease" """,

            # Seed biomarkers
            """MERGE (b:Biomarker {name: 'Amyloid-beta 42'})
               SET b.modality = 'CSF', b.unit = 'pg/mL' """,
            """MERGE (b:Biomarker {name: 'Total Tau'})
               SET b.modality = 'CSF', b.unit = 'pg/mL' """,
            """MERGE (b:Biomarker {name: 'Phosphorylated Tau 181'})
               SET b.modality = 'blood', b.unit = 'pg/mL' """,
            """MERGE (b:Biomarker {name: 'Hippocampal Volume'})
               SET b.modality = 'MRI', b.unit = 'mm3' """,

            # Seed genes
            """MERGE (g:Gene {symbol: 'APOE'})
               SET g.ensembl_id = 'ENSG00000130203' """,
            """MERGE (g:Gene {symbol: 'CLU'})
               SET g.ensembl_id = 'ENSG00000120885' """,

            # Link MCI → biomarkers
            """MATCH (d:Diagnosis {icd_code: 'G31.84'}), (b:Biomarker {name: 'Amyloid-beta 42'})
               MERGE (d)-[:ASSOCIATED_WITH]->(b)""",
            """MATCH (d:Diagnosis {icd_code: 'G31.84'}), (b:Biomarker {name: 'Hippocampal Volume'})
               MERGE (d)-[:ASSOCIATED_WITH]->(b)""",

            # Link AD → genes
            """MATCH (d:Diagnosis {icd_code: 'G30.9'}), (g:Gene {symbol: 'APOE'})
               MERGE (d)-[:ASSOCIATED_WITH]->(g)""",
        ]

        with self._driver.session(database=self.database) as session:
            for stmt in cypher_statements:
                session.run(stmt)
        logger.info("Neo4j schema initialised with MCI/Alzheimer's seed data.")

    # ── Query methods ────────────────────────────────────────────────────────

    def query_concept(self, concept: str) -> KGQueryResult:
        """
        Query the graph for entities related to a medical concept.

        Parameters
        ----------
        concept : A string like "MCI", "Amyloid-beta 42", or "APOE".

        Returns
        -------
        KGQueryResult with a list of related entities and a text summary.
        """
        if self._driver is None:
            return self._offline_fallback(concept)

        cypher = """
            MATCH (n)
            WHERE n.name CONTAINS $concept OR n.symbol CONTAINS $concept
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN
                n.name AS entity,
                labels(n)[0] AS entity_type,
                type(r) AS relationship,
                m.name AS related_entity,
                labels(m)[0] AS related_type
            LIMIT 25
        """

        related: list[dict[str, Any]] = []
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, concept=concept)
            for record in result:
                related.append(dict(record))

        context_text = self._build_context_text(concept, related)
        return KGQueryResult(concept=concept, related_entities=related, context_text=context_text)

    def query_patient_context(self, patient_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve all diagnosis, biomarker, and symptom relationships for a patient.
        Used to build RAG context at inference time.
        """
        if self._driver is None:
            return None

        cypher = """
            MATCH (p:Patient {patient_id: $pid})
            OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
            OPTIONAL MATCH (p)-[:HAS_BIOMARKER]->(b:Biomarker)
            OPTIONAL MATCH (p)-[:EXHIBITS]->(s:Symptom)
            RETURN
                p.patient_id  AS patient_id,
                p.mmse_score  AS mmse_score,
                collect(DISTINCT d.name) AS diagnoses,
                collect(DISTINCT b.name) AS biomarkers,
                collect(DISTINCT s.name) AS symptoms
        """

        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, pid=patient_id).single()
            return dict(result) if result else None

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context_text(concept: str, records: list[dict]) -> str:
        """Convert raw query records into a human-readable context string for RAG."""
        if not records:
            return f"No graph information found for '{concept}'."

        lines = [f"Knowledge graph context for '{concept}':"]
        seen: set[str] = set()

        for rec in records:
            entity = rec.get("entity", "")
            rel = rec.get("relationship", "")
            related = rec.get("related_entity", "")

            if rel and related:
                line = f"  • {entity} [{rec.get('entity_type','')}] —[{rel}]→ {related} [{rec.get('related_type','')}]"
            else:
                line = f"  • {entity} [{rec.get('entity_type','')}]"

            if line not in seen:
                seen.add(line)
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _offline_fallback(concept: str) -> KGQueryResult:
        """Return a static hardcoded context when no DB connection is available."""
        static_contexts: dict[str, str] = {
            "MCI": (
                "Mild Cognitive Impairment (MCI) is characterised by a decline in cognitive "
                "function beyond normal aging. Key biomarkers include reduced Amyloid-beta 42, "
                "elevated Total Tau in CSF, and reduced hippocampal volume on MRI. "
                "APOE-ε4 allele is the strongest genetic risk factor."
            ),
            "Amyloid": (
                "Amyloid-beta 42 is a peptide fragment. Low CSF levels (<192 pg/mL) indicate "
                "Alzheimer's pathology. Associated with MCI-to-AD progression."
            ),
            "APOE": (
                "APOE (Apolipoprotein E) gene, specifically the ε4 allele, is the primary "
                "genetic risk factor for late-onset Alzheimer's Disease."
            ),
        }

        for key, text in static_contexts.items():
            if key.lower() in concept.lower():
                return KGQueryResult(
                    concept=concept,
                    related_entities=[],
                    context_text=text,
                )

        return KGQueryResult(
            concept=concept,
            related_entities=[],
            context_text=f"[Offline] No static context available for '{concept}'.",
        )
