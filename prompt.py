def grounded_prompt(context, question):
    return f"""
You must answer ONLY from the context below.
If the answer is not present, say:
"Not found in the provided documents."

Context:
{context}

Question:
{question}

Answer (cite sources):
"""
