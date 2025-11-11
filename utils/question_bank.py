"""
Curated interview question bank for technical and HR topics.
These are authoritative Q&A items stored in the vector DB as knowledge for RAG.
"""
from typing import List, Dict, Any

# Each item structure:
# {
#   "question": str,
#   "answer": str,
#   "tags": List[str],          # skills/keywords
#   "category": "technical"|"hr",
#   "difficulty": "easy"|"medium"|"hard",
#   "skill": str,               # primary skill area
# }

TECHNICAL_QUESTIONS: List[Dict[str, Any]] = [
    # Data Structures & Algorithms
    {
        "question": "What is the time complexity of different array operations (access, search, insert, delete)?",
        "answer": (
            "Access: O(1) due to direct indexing. "
            "Search: O(n) for unsorted, O(log n) for sorted with binary search. "
            "Insert/Delete at end: amortized O(1) (dynamic arrays); at arbitrary index: O(n) due to shifting."
        ),
        "tags": ["dsa", "arrays", "time-complexity"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },
    {
        "question": "Explain the difference between a stack and a queue with real-world examples.",
        "answer": (
            "Stack: LIFO (Last-In, First-Out), e.g., browser back stack or function call stack. "
            "Queue: FIFO (First-In, First-Out), e.g., print queue or task scheduling."
        ),
        "tags": ["dsa", "stack", "queue"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },
    {
        "question": "What is a hash table? How do collisions occur and how are they handled?",
        "answer": (
            "A hash table stores key→value pairs using a hash function to index buckets. Collisions occur when two keys map to the same bucket. "
            "Common strategies: chaining (linked lists or trees per bucket) and open addressing (linear/quadratic probing, double hashing)."
        ),
        "tags": ["hashing", "hash-table", "collision-handling"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },
    {
        "question": "Binary search tree vs balanced BST (AVL/Red-Black): differences and trade-offs?",
        "answer": (
            "Unbalanced BST can degrade to O(n) operations in the worst case. Balanced trees (AVL, Red-Black) maintain height O(log n), guaranteeing O(log n) search/insert/delete. "
            "AVL is stricter (faster lookups), Red-Black is looser (faster inserts/deletes)."
        ),
        "tags": ["bst", "avl", "red-black", "trees"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },
    {
        "question": "What is Big-O, Big-Θ, and Big-Ω notation?",
        "answer": (
            "Big-O gives an asymptotic upper bound (worst-case). Big-Ω gives a lower bound (best-case). Big-Θ gives a tight bound (both upper and lower)."
        ),
        "tags": ["complexity", "big-o", "asymptotics"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },

    # Python
    {
        "question": "What is the Global Interpreter Lock (GIL) in Python, and when does it matter?",
        "answer": (
            "The GIL allows only one thread to execute Python bytecode at a time in CPython. It affects CPU-bound multi-threaded code; I/O-bound threads are fine. "
            "Workarounds: multiprocessing, native extensions (NumPy), or alternative interpreters."
        ),
        "tags": ["python", "gil", "concurrency"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "python",
    },
    {
        "question": "Explain decorators and provide a simple example.",
        "answer": (
            "Decorators wrap a function to modify behavior without changing its source. "
            "Example: @timed to measure execution time by recording start/end and printing duration."
        ),
        "tags": ["python", "decorators", "functions"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "python",
    },
    {
        "question": "What is the difference between an iterator and a generator in Python?",
        "answer": (
            "An iterator implements __iter__ and __next__. A generator is a special iterator created with yield; it auto-implements the iterator protocol and is memory-efficient."
        ),
        "tags": ["python", "iterator", "generator"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "python",
    },

    # JavaScript
    {
        "question": "var vs let vs const in JavaScript?",
        "answer": (
            "var has function scope and is hoisted; let/const have block scope. const prevents reassignment of the binding (object contents can still change)."
        ),
        "tags": ["javascript", "scope", "hoisting"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "javascript",
    },
    {
        "question": "Explain closures and a practical use case.",
        "answer": (
            "A closure captures variables from the lexical scope even after the outer function returns. Use cases: data privacy, function factories, memoization."
        ),
        "tags": ["javascript", "closures", "functional"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "javascript",
    },
    {
        "question": "How does the event loop work in JavaScript?",
        "answer": (
            "The event loop pulls tasks from the callback/microtask queues to the call stack when it’s empty. Promises (microtasks) run before timers (macrotasks)."
        ),
        "tags": ["javascript", "event-loop", "async"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "javascript",
    },

    # Java
    {
        "question": "HashMap vs Hashtable vs ConcurrentHashMap in Java?",
        "answer": (
            "HashMap is non-synchronized; Hashtable is legacy and synchronized; ConcurrentHashMap supports concurrent access with segmenting/locking."
        ),
        "tags": ["java", "collections", "concurrency"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "java",
    },
    {
        "question": "Explain equals() and hashCode() contract.",
        "answer": (
            "Equal objects must have the same hashCode. If equals is overridden, hashCode must be consistent with equals to maintain correct behavior in hash-based collections."
        ),
        "tags": ["java", "equals", "hashcode"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "java",
    },

    # SQL/DBMS
    {
        "question": "What are database indexes and how do they improve performance?",
        "answer": (
            "Indexes are auxiliary data structures (e.g., B-trees) that speed up lookups and sorting by avoiding full table scans. They trade faster reads for slower writes and extra storage."
        ),
        "tags": ["sql", "indexes", "performance"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "sql",
    },
    {
        "question": "Explain ACID properties and transaction isolation levels.",
        "answer": (
            "ACID: Atomicity, Consistency, Isolation, Durability. Isolation levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable) trade anomalies vs concurrency."
        ),
        "tags": ["dbms", "acid", "transactions", "isolation"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dbms",
    },

    # Operating Systems
    {
        "question": "Process vs Thread?",
        "answer": (
            "A process is an independent execution unit with its own memory space. A thread is a lighter unit within a process sharing its memory. Threads enable concurrency within a process."
        ),
        "tags": ["os", "process", "thread"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "os",
    },
    {
        "question": "What is deadlock and how can it be prevented?",
        "answer": (
            "Deadlock is a circular wait among processes for resources. Prevention strategies: eliminate one Coffman condition (e.g., lock ordering, hold-and-wait avoidance, resource preemption)."
        ),
        "tags": ["os", "deadlock", "concurrency"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "os",
    },

    # Networks
    {
        "question": "TCP vs UDP: differences and use cases?",
        "answer": (
            "TCP is connection-oriented, reliable, ordered; used for web, email. UDP is connectionless, faster, no guarantees; used for streaming, gaming, DNS queries."
        ),
        "tags": ["networking", "tcp", "udp"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "networking",
    },
    {
        "question": "Explain the TCP three-way handshake.",
        "answer": (
            "Client sends SYN, server replies SYN-ACK, client responds ACK. Establishes sequence numbers and connection parameters before data transfer."
        ),
        "tags": ["networking", "tcp", "handshake"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "networking",
    },

    # System Design
    {
        "question": "What is the CAP theorem and its implications?",
        "answer": (
            "In a distributed system, you can only guarantee two of Consistency, Availability, and Partition tolerance. Under network partition, you must trade consistency vs availability."
        ),
        "tags": ["system-design", "cap", "distributed"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },
    {
        "question": "Explain caching strategies and cache invalidation.",
        "answer": (
            "Strategies: write-through, write-back, write-around; eviction policies: LRU, LFU, FIFO. Invalidation: time-based (TTL), event-driven, versioned keys."
        ),
        "tags": ["system-design", "caching", "performance"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },

    # DevOps
    {
        "question": "Docker container vs Virtual Machine?",
        "answer": (
            "Containers share the host OS kernel, are lightweight and start fast. VMs virtualize hardware with separate OS instances, heavier but stronger isolation."
        ),
        "tags": ["devops", "docker", "vm"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "devops",
    },
    {
        "question": "What is Kubernetes and its core components?",
        "answer": (
            "Kubernetes orchestrates containers. Core components: Pod, Deployment, Service, ConfigMap/Secret, Node, Cluster, etcd (state), Controller Manager, Scheduler, kubelet."
        ),
        "tags": ["devops", "kubernetes", "orchestration"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "devops",
    },

    # Machine Learning
    {
        "question": "Bias vs Variance trade-off?",
        "answer": (
            "Bias: error from wrong assumptions (underfitting). Variance: error from sensitivity to data (overfitting). Regularization and more data help balance the trade-off."
        ),
        "tags": ["ml", "bias-variance", "generalization"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "ml",
    },
    {
        "question": "Precision, Recall, and F1-score definitions.",
        "answer": (
            "Precision = TP/(TP+FP), Recall = TP/(TP+FN). F1 is harmonic mean of precision and recall. Use when class imbalance exists or both metrics matter."
        ),
        "tags": ["ml", "metrics", "classification"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "ml",
    },
    
    # Additional DSA
    {
        "question": "What are the time complexities of common sorting algorithms (Quick, Merge, Heap, Insertion)?",
        "answer": (
            "QuickSort: average O(n log n), worst O(n^2); MergeSort: O(n log n) worst/avg; HeapSort: O(n log n) worst/avg; InsertionSort: O(n^2) worst/avg, O(n) best for nearly-sorted."
        ),
        "tags": ["sorting", "algorithms", "complexity"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },
    {
        "question": "Explain linked list operations and when to use singly vs doubly linked lists.",
        "answer": (
            "Singly lists support O(1) insert/delete at head; finding previous takes O(n). Doubly lists store prev/next to allow O(1) insert/delete at both ends and efficient deletion when node known; cost is more memory."
        ),
        "tags": ["linked-list", "doubly", "singly"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "dsa",
    },
    {
        "question": "Explain heap and priority queue with operations and complexities.",
        "answer": (
            "Binary heap implements a priority queue. Insert: O(log n), get-min/max: O(1), extract: O(log n), decrease-key: O(log n). Useful for Dijkstra, scheduling, and streaming top-k."
        ),
        "tags": ["heap", "priority-queue", "dsa"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },
    {
        "question": "Two-pointer and sliding window techniques: when and how?",
        "answer": (
            "Two pointers: sorted arrays, pair sums, dedupe; Sliding window: contiguous subarray problems (max sum, longest substring) managing start/end indices with O(n) complexity."
        ),
        "tags": ["two-pointers", "sliding-window", "patterns"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dsa",
    },

    # Python Advanced
    {
        "question": "How do context managers work in Python? Implement one.",
        "answer": (
            "Context managers define __enter__/__exit__ to manage resources. Implement via class or @contextmanager. Ensures cleanup (e.g., files, locks) even on exceptions."
        ),
        "tags": ["python", "context-manager", "with"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "python",
    },
    {
        "question": "Mutable default arguments pitfall in Python.",
        "answer": (
            "Default arguments evaluated once at function definition. Using mutable defaults (e.g., list) shares state across calls. Use None + create inside function."
        ),
        "tags": ["python", "defaults", "pitfall"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "python",
    },
    {
        "question": "async/await in Python vs threads: differences and use cases.",
        "answer": (
            "async/await provides cooperative concurrency for I/O-bound tasks with an event loop (single thread). Threads can run blocking I/O simultaneously; CPU-bound requires multiprocessing."
        ),
        "tags": ["python", "asyncio", "concurrency"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "python",
    },

    # JavaScript Advanced
    {
        "question": "Explain JavaScript prototype chain and how inheritance works.",
        "answer": (
            "Objects delegate property lookups to their [[Prototype]]. Functions have prototype used for instances via new. Class syntax sugar over prototypal inheritance."
        ),
        "tags": ["javascript", "prototype", "inheritance"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "javascript",
    },
    {
        "question": "Promise vs async/await: differences and error handling.",
        "answer": (
            "Promises represent async values with then/catch; async/await is syntactic sugar enabling try/catch. Await pauses within async functions; both rely on microtask queue."
        ),
        "tags": ["javascript", "promises", "async-await"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "javascript",
    },
    {
        "question": "Event loop phases (Node.js) and microtasks vs macrotasks.",
        "answer": (
            "Macrotasks: timers, I/O, check; Microtasks: promises, queueMicrotask. Microtasks drain after each macrotask. Ordering influences observable behavior."
        ),
        "tags": ["javascript", "event-loop", "nodejs"],
        "category": "technical",
        "difficulty": "hard",
        "skill": "javascript",
    },

    # Java Advanced
    {
        "question": "JVM memory areas and garbage collection basics.",
        "answer": (
            "Heap (young/old), stacks, metaspace. GC algorithms: Serial, Parallel, CMS, G1, ZGC. Concepts: stop-the-world, minor/major GC, GC tuning trade-offs."
        ),
        "tags": ["java", "jvm", "gc"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "java",
    },
    {
        "question": "volatile vs synchronized in Java.",
        "answer": (
            "volatile ensures visibility and ordering (no atomicity beyond single read/write). synchronized provides mutual exclusion and visibility upon lock release/acquire."
        ),
        "tags": ["java", "concurrency", "volatile", "synchronized"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "java",
    },

    # SQL/DBMS Advanced
    {
        "question": "INNER vs LEFT vs RIGHT vs FULL OUTER JOIN with examples.",
        "answer": (
            "INNER: matching rows only. LEFT: all left + matches, RIGHT: all right + matches, FULL: union of both with NULLs for non-matches."
        ),
        "tags": ["sql", "joins"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "sql",
    },
    {
        "question": "Normalization forms (1NF, 2NF, 3NF, BCNF) in brief.",
        "answer": (
            "1NF: atomic values; 2NF: no partial dependency on PK; 3NF: no transitive dependency; BCNF: every determinant is a candidate key. Reduces anomalies."
        ),
        "tags": ["dbms", "normalization"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "dbms",
    },
    {
        "question": "Indexing pitfalls: when indexes hurt performance.",
        "answer": (
            "Too many/unused indexes slow writes; low-selectivity columns; functions on indexed columns prevent usage; mismatched collation; wrong order in composite indexes."
        ),
        "tags": ["sql", "indexing", "performance"],
        "category": "technical",
        "difficulty": "hard",
        "skill": "sql",
    },

    # Operating Systems Advanced
    {
        "question": "Paging vs segmentation and modern virtual memory.",
        "answer": (
            "Paging splits memory into fixed-size pages/frames; segmentation is variable-sized logical units. Modern systems combine paging with protection and virtual address translation via TLBs."
        ),
        "tags": ["os", "memory", "paging", "segmentation"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "os",
    },
    {
        "question": "CPU scheduling algorithms and trade-offs (FCFS, SJF, RR, Priority).",
        "answer": (
            "FCFS: simple, can convoy; SJF: optimal avg wait but needs prediction; RR: fair time slicing; Priority: starvation risk; use aging to mitigate."
        ),
        "tags": ["os", "scheduling"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "os",
    },

    # Networks Advanced
    {
        "question": "OSI vs TCP/IP models and mapping of layers.",
        "answer": (
            "OSI: 7 layers; TCP/IP: 4 (Link, Internet, Transport, Application). Mapping: OSI app/presentation/session → TCP/IP application; network → internet; transport → transport; data link/physical → link."
        ),
        "tags": ["networking", "osi", "tcpip"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "networking",
    },
    {
        "question": "HTTP/1.1 vs HTTP/2 vs HTTP/3 differences.",
        "answer": (
            "HTTP/2: multiplexing over single TCP, header compression (HPACK), server push. HTTP/3 uses QUIC (UDP), reduces HOL blocking, faster handshakes and better mobility."
        ),
        "tags": ["http", "http2", "http3", "quic"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "networking",
    },
    {
        "question": "TLS handshake overview and certificates.",
        "answer": (
            "ClientHello/ServerHello negotiate cipher suites; server sends cert; key exchange (ECDHE) derives shared secret; Finished messages confirm; cert chain validated to root CA."
        ),
        "tags": ["tls", "security", "certificates"],
        "category": "technical",
        "difficulty": "hard",
        "skill": "networking",
    },

    # System Design Advanced
    {
        "question": "Design a rate limiter: token bucket vs leaky bucket.",
        "answer": (
            "Token bucket allows bursts up to bucket size; refills at rate r. Leaky bucket enforces constant outflow. Choose based on tolerance for bursts and smoothing needs."
        ),
        "tags": ["system-design", "rate-limiting"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },
    {
        "question": "Message queues vs stream platforms (RabbitMQ vs Kafka).",
        "answer": (
            "MQ: work queues, per-message ack, flexible routing, consumer state. Kafka: immutable logs, partitioned, consumer-managed offsets, high-throughput event streaming."
        ),
        "tags": ["system-design", "kafka", "rabbitmq"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },
    {
        "question": "Consistency models: strong vs eventual; read-your-writes, monotonic reads.",
        "answer": (
            "Strong: all clients see latest writes; Eventual: converges over time. Session guarantees improve UX. Trade-offs with CAP and latency/availability."
        ),
        "tags": ["consistency", "cap", "distributed"],
        "category": "technical",
        "difficulty": "hard",
        "skill": "system-design",
    },
    {
        "question": "Globally unique ID generation strategies.",
        "answer": (
            "UUID (random/time), database sequences, Snowflake (timestamp + datacenter + worker + sequence), KSUID/ULID for k-sortable IDs; trade-offs: ordering, size, coordination."
        ),
        "tags": ["system-design", "ids", "uuid"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "system-design",
    },

    # DevOps Advanced
    {
        "question": "Blue-green vs canary deployments.",
        "answer": (
            "Blue-green: two prod environments, instant switch, easy rollback. Canary: progressive rollout to subset, monitors metrics, lower blast radius, slower. Often used together with feature flags."
        ),
        "tags": ["devops", "deployments", "blue-green", "canary"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "devops",
    },
    {
        "question": "Observability pillars and key SRE metrics.",
        "answer": (
            "Logs, metrics, traces. SRE: SLI/SLO/SLA, error budgets. Golden signals: latency, traffic, errors, saturation."
        ),
        "tags": ["observability", "sre", "metrics"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "devops",
    },

    # ML Advanced
    {
        "question": "Overfitting mitigation techniques.",
        "answer": (
            "Regularization (L1/L2), dropout, data augmentation, cross-validation, early stopping, simpler models, more data."
        ),
        "tags": ["ml", "overfitting", "regularization"],
        "category": "technical",
        "difficulty": "easy",
        "skill": "ml",
    },
    {
        "question": "Gradient descent variants and when to use them.",
        "answer": (
            "Batch, mini-batch, SGD; adaptive optimizers: AdaGrad, RMSProp, Adam. Choose based on noise, curvature, and convergence speed."
        ),
        "tags": ["ml", "optimization", "gradient-descent"],
        "category": "technical",
        "difficulty": "medium",
        "skill": "ml",
    },
]

HR_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "Tell me about yourself.",
        "answer": (
            "Use a concise, role-focused summary: background, 2-3 relevant achievements, current focus, and why this role/company. Keep it 60–90 seconds using the elevator pitch format."
        ),
        "tags": ["hr", "introduction", "communication"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
    {
        "question": "What are your strengths and weaknesses?",
        "answer": (
            "Pick strengths that match the role and give a brief example. Choose a genuine, non-critical weakness with actions you’re taking to improve (growth mindset)."
        ),
        "tags": ["hr", "strengths", "weaknesses"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
    {
        "question": "Describe a time you handled conflict in a team.",
        "answer": (
            "Use STAR: Situation, Task, Action, Result. Show empathy, active listening, collaborative problem-solving, and a measurable positive outcome."
        ),
        "tags": ["hr", "conflict", "teamwork"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "Why do you want to work here?",
        "answer": (
            "Show you’ve researched the company: mission, products, culture, recent news. Connect to your skills and career goals with specifics."
        ),
        "tags": ["hr", "motivation", "company-fit"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
    {
        "question": "Tell me about a failure and what you learned.",
        "answer": (
            "Choose a real, safe failure. Emphasize accountability, what you changed, and subsequent success. Keep it constructive and growth-oriented."
        ),
        "tags": ["hr", "failure", "learning"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "Where do you see yourself in five years?",
        "answer": (
            "Show ambition aligned with the role and industry, highlight learning goals, leadership aspirations, and contribution to the company’s roadmap."
        ),
        "tags": ["hr", "career-goals", "planning"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
    {
        "question": "How do you handle tight deadlines and pressure?",
        "answer": (
            "Describe prioritization, communication, breaking down tasks, timeboxing, and asking for help early when needed. Give a concise example with outcome."
        ),
        "tags": ["hr", "pressure", "prioritization"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "What is your expected salary?",
        "answer": (
            "Share a researched range based on market and your experience; express flexibility and focus on overall fit, responsibilities, and growth."
        ),
        "tags": ["hr", "salary", "negotiation"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "Describe a time you led without authority.",
        "answer": (
            "Use STAR. Show initiative, influence, aligning stakeholders, and measurable results. Emphasize communication and collaboration over positional power."
        ),
        "tags": ["hr", "leadership", "influence"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "How do you prioritize when everything is important?",
        "answer": (
            "Clarify goals, impact vs effort matrix, deadlines, dependencies; negotiate scope; timebox; keep stakeholders informed; iterate. Give a concrete example."
        ),
        "tags": ["hr", "prioritization", "planning"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "Tell me about a disagreement with your manager and how you handled it.",
        "answer": (
            "Stay professional: seek to understand, present data, propose options, align on objectives, accept decision, follow up with results. Avoid blaming."
        ),
        "tags": ["hr", "conflict", "communication"],
        "category": "hr",
        "difficulty": "medium",
        "skill": "hr",
    },
    {
        "question": "Give an example of delivering feedback to a peer.",
        "answer": (
            "Use SBI (Situation-Behavior-Impact), be specific, timely, and actionable; invite dialogue; agree next steps. Share outcome improvement."
        ),
        "tags": ["hr", "feedback", "collaboration"],
        "category": "hr",
        "difficulty": "easy",
        "skill": "hr",
    },
]

INTERVIEW_QUESTIONS: List[Dict[str, Any]] = TECHNICAL_QUESTIONS + HR_QUESTIONS

__all__ = [
    "TECHNICAL_QUESTIONS",
    "HR_QUESTIONS",
    "INTERVIEW_QUESTIONS",
]
