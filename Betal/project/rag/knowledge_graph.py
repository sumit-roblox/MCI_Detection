"""
Knowledge Graph Module (Starter).
Placeholder for connecting to a Neo4j schema for medical knowledge.
"""

class MedicalKnowledgeGraph:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        
    def connect(self):
        """Establish connection to Neo4j database."""
        pass
        
    def query_concept(self, concept: str):
        """Query the graph for a specific medical concept (e.g., MCI symptoms)."""
        return []
