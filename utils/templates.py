"""Core templates to drive template-based RAG generation.

Each template is seeded into the vector DB as a knowledge item (type='template'),
with metadata including template_for and full text under 'template'.
"""

TEMPLATES = [
    {
        "template_for": "resume_analysis",
        "title": "Resume Analysis & Feedback Template v1",
        "version": "v1",
        "template": (
            "You are an ATS-aware resume coach. Using the provided resume facts, produce:\n"
            "1) A brief assessment (1-2 sentences)\n"
            "2) 5 concrete recommendations (numbered)\n"
            "3) Section-wise feedback: Contact, Skills, Education, Experience, Formatting (1-2 bullets each)\n\n"
            "Rules:\n- Be specific and actionable.\n- Prefer measurable suggestions (add numbers, outcomes).\n- Avoid adding facts not present in the resume.\n- Keep total under 250-300 words.\n"
        ),
    },
    {
        "template_for": "resume_recommendations",
        "title": "Resume Recommendations Template v1",
        "version": "v1",
        "template": (
            "From the resume snapshot, produce exactly 5 prioritized recommendations that improve ATS and clarity.\n"
            "Each item: one sentence; start with an action verb; mention the target section when relevant.\n"
        ),
    },
    {
        "template_for": "pdf_summary",
        "title": "PDF Summary Template v1",
        "version": "v1",
        "template": (
            "Summarize the document excerpt into 1 paragraph covering: purpose, key points, data/findings, and conclusion.\n"
            "Avoid generic filler. Do not invent content beyond the excerpt. Limit to ~150-200 words.\n"
        ),
    },
    {
        "template_for": "pdf_qa",
        "title": "PDF Q&A Template v1",
        "version": "v1",
        "template": (
            "Answer strictly from the retrieved document context. If unknown, say you cannot find it in the document.\n"
            "Keep answers concise (2-5 sentences). Include a short quote only if it directly supports the answer.\n"
        ),
    },
    {
        "template_for": "skill_gap",
        "title": "Skill Gap & Learning Plan Template v1",
        "version": "v1",
        "template": (
            "Given target role and missing skills, produce 3-5 learning recommendations.\n"
            "For each: skill, priority (essential|important|nice_to_have), difficulty, estimated_time, and 2-3 resources.\n"
            "Close with 3-step learning path for each skill. Keep JSON-friendly lists where possible.\n"
        ),
    },
    {
        "template_for": "career_roadmap",
        "title": "Career Roadmap Template v1",
        "version": "v1",
        "template": (
            "Return ONLY JSON with key 'phases'. Each phase has: name, duration, skills[], resources[], projects[].\n"
            "Duration strings in months; exactly 3-4 phases total; escalate difficulty; prefer free resources where possible.\n"
        ),
    },
    {
        "template_for": "project_ideas",
        "title": "Role-Specific Project Ideas Template v1",
        "version": "v1",
        "template": (
            "Generate portfolio-ready project ideas tailored to a target role and skills.\n"
            "Rules:\n"
            "- Each idea must be specific to the role/context, not generic.\n"
            "- Use realistic modern stacks; align with provided skills.\n"
            "- Provide clear value proposition and what it demonstrates.\n"
            "- Difficulty should progress across items.\n"
            "Return ONLY JSON with key 'projects' (name, description, technologies[], difficulty, estimated_hours).\n"
        ),
    },
    {
        "template_for": "chatbot_answer",
        "title": "Placement Chatbot Answering Template v1",
        "version": "v1",
        "template": (
            "You are a placement mentor. Use retrieved context plus conversation history.\n"
            "Be concise, practical, and avoid hallucinations. If unknown, say so and suggest how to find out.\n"
        ),
    },
]
