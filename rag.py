import json
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
from torch import Tensor
import numpy as np
from typing import List, Dict, Any  

# Load environment variables
load_dotenv()


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    if not isinstance(b, Tensor):
        b = torch.tensor(b)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


class LocalRAG:
    def __init__(self):
        self.config = load_config()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.load_vault()

    def load_vault(self):
        # Check if vault file exists
        if not os.path.exists(self.config["vault_file"]):
            raise FileNotFoundError(
                f"Vault file {self.config['vault_file']} not found. Please create it first."
            )

        # Load the text content
        with open(self.config["vault_file"], "r", encoding="utf-8") as f:
            self.vault_text = f.read().split("\n")

        # Check if embeddings file exists, if not, generate embeddings
        if not os.path.exists(self.config["embeddings_file"]):
            print("Embeddings file not found. Generating embeddings...")
            self.generate_embeddings()

        # Load the embeddings
        with open(self.config["embeddings_file"], "r") as f:
            self.embeddings = json.load(f)

        # Convert embeddings to tensor
        self.embeddings_tensor = torch.tensor(self.embeddings)

    def generate_embeddings(self):
        """Generate embeddings for the vault content"""
        print("Generating embeddings... This may take a while.")
        embeddings = []
        for i, text in enumerate(self.vault_text):
            if text.strip():  # Skip empty lines
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002", input=text
                )
                embeddings.append(response.data[0].embedding)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} lines...")

        # Save embeddings
        with open(self.config["embeddings_file"], "w") as f:
            json.dump(embeddings, f)
        print("Embeddings generated and saved successfully!")

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI's embedding model"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding

    def get_relevant_context(self, query: str) -> str:
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]

        # Get top-k most similar indices
        top_k = self.config["top_k"]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Construct context from top-k most similar documents
        context = "\n".join([self.vault_text[i] for i in top_indices])
        return context

    def query(self, user_input: str) -> str:
        # Get relevant context
        context = self.get_relevant_context(user_input)

        # Construct the messages for ChatGPT
        messages = [
            {"role": "system", "content": self.config["system_message"]},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_input}",
            },
        ]

        # Get response from ChatGPT
        response = self.client.chat.completions.create(
            model=self.config["openai"]["model"],
            messages=messages,
            temperature=self.config["openai"]["temperature"],
            max_tokens=self.config["openai"]["max_tokens"],
        )

        return response.choices[0].message.content


def main():
    rag = LocalRAG()

    print("Welcome to LocalRAG! Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "quit":
            break

        try:
            response = rag.query(user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
