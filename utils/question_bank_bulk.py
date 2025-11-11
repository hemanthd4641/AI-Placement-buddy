"""
Bulk interview question bank for scaling beyond the base module.
Additions follow the same schema as utils/question_bank.py and can be extended freely.
"""
from typing import List, Dict, Any

# Schema per item:
# {
#   "question": str,
#   "answer": str,
#   "tags": List[str],
#   "category": "technical"|"hr",
#   "difficulty": "easy"|"medium"|"hard",
#   "skill": str,
# }

BULK_INTERVIEW_QUESTIONS: List[Dict[str, Any]] = [
    # Additional DSA
    {
        "question": "KMP vs Rabin-Karp vs Naive string matching: trade-offs?",
        "answer": (
            "Naive: O(nm) worst. KMP: O(n+m) using prefix function to avoid re-checks. Rabin-Karp: expected O(n+m) with rolling hash; worst O(nm) on collisions; good for multi-pattern search."
        ),
        "tags": ["dsa", "strings", "kmp", "rabin-karp"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },
    {
        "question": "Union-Find (Disjoint Set Union) operations and applications.",
        "answer": (
            "Operations: find with path compression, union by rank/size; near O(1) amortized. Applications: cycle detection, Kruskal MST, connected components, percolation."
        ),
        "tags": ["dsa", "union-find", "graphs"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },
    {
        "question": "Topological sort: when applicable and how it's implemented.",
        "answer": (
            "Applicable to DAGs to order tasks by dependencies. Implement via DFS postorder stack or Kahn's algorithm with in-degree queue; detects cycles if not all nodes processed."
        ),
        "tags": ["graphs", "topological-sort", "dag"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },

    # Python
    {
        "question": "Explain Python data model dunder methods (\"data model\") and why they matter.",
        "answer": (
            "Dunder methods (__len__, __iter__, __getitem__, __eq__, __hash__, __repr__, etc.) enable Pythonic behavior: containers, iteration, operator overloading, hashing, printing; integrate with ecosystem."
        ),
        "tags": ["python", "dunder", "datamodel"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "python",
    },
    {
        "question": "When to use dataclasses vs namedtuple vs attrs?",
        "answer": (
            "dataclasses: mutable records with defaults, type hints, easy to extend. namedtuple: lightweight immutable tuples. attrs: powerful validation, converters, performance options."
        ),
        "tags": ["python", "dataclasses", "namedtuple"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "python",
    },

    # JavaScript/Frontend
    {
        "question": "Virtual DOM and reconciliation basics.",
        "answer": (
            "Virtual DOM diffing computes minimal updates; reconciliation uses keys to match elements; improves perceived performance by batching and minimizing real DOM ops."
        ),
        "tags": ["react", "virtual-dom", "frontend"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "javascript",
    },
    {
        "question": "CORS and preflight requests explained.",
        "answer": (
            "CORS controls cross-origin resource sharing via headers (Origin, Access-Control-Allow-*). Preflight (OPTIONS) checks methods/headers; server must respond with allowed origins and methods."
        ),
        "tags": ["web", "cors", "http"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "javascript",
    },

    # SQL/DB
    {
        "question": "Sharding vs partitioning vs replication.",
        "answer": (
            "Partitioning splits a table for manageability/perf; sharding distributes data across nodes for scale; replication copies data for availability and reads; often combined."
        ),
        "tags": ["db", "sharding", "partitioning", "replication"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dbms",
    },
    {
        "question": "MVCC (Multi-Version Concurrency Control) in databases.",
        "answer": (
            "Readers see a snapshot without blocking writers; writers create new versions; GC cleans old versions. Reduces lock contention; used by Postgres, InnoDB."
        ),
        "tags": ["db", "mvcc", "concurrency"],
        "category": "technical",
        "difficulty": "hard",
        "skill": "dbms",
    },

    # System Design
    {
        "question": "CDN basics and cache invalidation strategies.",
        "answer": (
            "CDNs cache static/edge-compute content near users. Invalidation via TTLs, purge APIs, versioned assets. Consider cache hierarchy and consistency needs."
        ),
        "tags": ["cdn", "caching", "system-design"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "system-design",
    },
    {
        "question": "Idempotency in APIs and implementation techniques.",
        "answer": (
            "Idempotent methods yield same result on repeats. Use idempotency keys, dedupe tables, natural keys, or state checks to safely retry operations."
        ),
        "tags": ["api", "idempotency", "reliability"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },

    # DevOps/SRE
    {
        "question": "Circuit breaker and bulkhead patterns.",
        "answer": (
            "Circuit breaker trips on failures to stop cascading issues; half-open probes recovery. Bulkheads isolate resources per component to limit blast radius."
        ),
        "tags": ["resilience", "circuit-breaker", "bulkhead"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "devops",
    },
    {
        "question": "Git strategies: Gitflow vs trunk-based development.",
        "answer": (
            "Gitflow uses long-lived branches and release cycles; trunk-based favors small, frequent merges to main with feature flags; better CI/CD velocity."
        ),
        "tags": ["git", "workflow", "ci/cd"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "devops",
    },

    # ML/NLP
    {
        "question": "Word embeddings vs contextual embeddings.",
        "answer": (
            "Static embeddings (Word2Vec, GloVe) assign one vector per word. Contextual (BERT) depend on surrounding words, capturing polysemy and richer semantics."
        ),
        "tags": ["nlp", "embeddings", "bert"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "ml",
    },
    {
        "question": "Feature scaling techniques and why they matter.",
        "answer": (
            "Normalization (min-max), standardization (z-score), robust scaling; help gradient-based methods converge and distance-based methods perform properly."
        ),
        "tags": ["ml", "preprocessing", "scaling"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "ml",
    },

    # HR (Behavioral)
    {
        "question": "Tell me about a time you improved a process.",
        "answer": (
            "STAR: identify bottleneck, measure baseline, propose change, implement, measure improvement (e.g., 30% faster), and lessons learned."
        ),
        "tags": ["hr", "process", "improvement"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
    {
        "question": "How do you handle ambiguous requirements?",
        "answer": (
            "Clarify with stakeholders, define acceptance criteria, create prototypes/spikes, iterate, document decisions, and manage risk with timeboxed experiments."
        ),
        "tags": ["hr", "ambiguity", "requirements"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
]

__all__ = ["BULK_INTERVIEW_QUESTIONS"]
