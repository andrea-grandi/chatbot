import os
from openai import OpenAI


class Generator:

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, query, docs):
        context = "\n\n".join(docs)
        prompt = f"""You are an expert assistant. Use the following context to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

        response = self.client.chat.completions.create(
          model=self.model,
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
          ],
          temperature=0.2
        )

        return response.choices[0].message.content.strip()
