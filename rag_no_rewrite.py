import torch
import os
from openai import OpenAI
import argparse
from dotenv import load_dotenv

# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Load environment variables
load_dotenv()


# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


# Function to get relevant context from the vault based on user input
def get_relevant_context(
    rewritten_input, vault_embeddings, vault_content, client, top_k=3
):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []

    # Get embedding for the input using OpenAI
    response = client.embeddings.create(
        model="text-embedding-ada-002", input=rewritten_input
    )
    input_embedding = response.data[0].embedding

    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(
        torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
    )

    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))

    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context


# Function to interact with the OpenAI model
def chat_with_gpt(
    user_input,
    system_message,
    vault_embeddings,
    vault_content,
    model,
    conversation_history,
    client,
):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(
        user_input, vault_embeddings_tensor, vault_content, client, top_k=3
    )

    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = f"Context:\n{context_str}\n\nQuestion: {user_input}"

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    # Send the completion request to the OpenAI model
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=1000
    )

    # Append the model's response to the conversation history
    conversation_history.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )

    # Return the content of the response from the model
    return response.choices[0].message.content


# Parse command-line arguments
parser = argparse.ArgumentParser(description="ChatGPT RAG System")
parser.add_argument(
    "--model",
    default="gpt-3.5-turbo",
    help="OpenAI model to use (default: gpt-3.5-turbo)",
)
args = parser.parse_args()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the vault content
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding="utf-8") as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using OpenAI
print("Generating embeddings for vault content...")
vault_embeddings = []
for content in vault_content:
    if content.strip():  # Skip empty lines
        response = client.embeddings.create(
            model="text-embedding-ada-002", input=content
        )
        vault_embeddings.append(response.data[0].embedding)

# Convert to tensor and print embeddings
vault_embeddings_tensor = torch.tensor(vault_embeddings)
print("Embeddings generated successfully!")

# Conversation loop
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text"

print(PINK + "\nWelcome to the RAG Chat System!" + RESET_COLOR)
print(CYAN + "Type 'quit' to exit the chat." + RESET_COLOR)

while True:
    user_input = input(YELLOW + "\nAsk a question about your documents: " + RESET_COLOR)
    if user_input.lower() == "quit":
        break

    try:
        response = chat_with_gpt(
            user_input,
            system_message,
            vault_embeddings_tensor,
            vault_content,
            args.model,
            conversation_history,
            client,
        )
        print(NEON_GREEN + "\nResponse: \n\n" + response + RESET_COLOR)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
