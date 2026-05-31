def build_prompt(context, question):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided in the context.

If the answer is not available in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt