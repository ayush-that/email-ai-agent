import torch
import os
import json
from openai import OpenAI
import argparse
import yaml
from dotenv import load_dotenv

# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Load environment variables
load_dotenv()


def load_config(config_file):
    print("Loading configuration...")
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found.")
        exit(1)


def open_file(filepath):
    print("Opening file...")
    try:
        with open(filepath, "r", encoding="utf-8") as infile:
            return infile.read()
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None


def load_or_generate_embeddings(vault_content, embeddings_file, client):
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from '{embeddings_file}'...")
        try:
            with open(embeddings_file, "r", encoding="utf-8") as file:
                return torch.tensor(json.load(file))
        except json.JSONDecodeError:
            print(f"Invalid JSON format in embeddings file '{embeddings_file}'.")
            embeddings = []
    else:
        print(f"No embeddings found. Generating new embeddings...")
        embeddings = generate_embeddings(vault_content, client)
        save_embeddings(embeddings, embeddings_file)
    return torch.tensor(embeddings)


def generate_embeddings(vault_content, client):
    print("Generating embeddings...")
    embeddings = []
    for content in vault_content:
        if content.strip():  # Skip empty lines
            try:
                response = client.embeddings.create(
                    model="text-embedding-ada-002", input=content
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
    return embeddings


def save_embeddings(embeddings, embeddings_file):
    print(f"Saving embeddings to '{embeddings_file}'...")
    try:
        with open(embeddings_file, "w", encoding="utf-8") as file:
            json.dump(embeddings, file)
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")


def get_relevant_context(
    rewritten_input, vault_embeddings, vault_content, top_k, client
):
    print("Retrieving relevant context...")
    if vault_embeddings.nelement() == 0:
        return []
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002", input=rewritten_input
        )
        input_embedding = response.data[0].embedding
        cos_scores = torch.cosine_similarity(
            torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
        )
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        return [vault_content[idx].strip() for idx in top_indices]
    except Exception as e:
        print(f"Error getting relevant context: {str(e)}")
        return []


def chat_with_gpt(
    user_input,
    system_message,
    vault_embeddings,
    vault_content,
    model,
    conversation_history,
    top_k,
    client,
):
    relevant_context = get_relevant_context(
        user_input, vault_embeddings, vault_content, top_k, client
    )
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print("No relevant context found.")

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = f"Context:\n{context_str}\n\nQuestion: {user_input}"

    conversation_history.append({"role": "user", "content": user_input_with_context})
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.7, max_tokens=1000
        )
        conversation_history.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        return "An error occurred while processing your request."


def main():
    parser = argparse.ArgumentParser(description="Email RAG System")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear the embeddings cache"
    )
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.clear_cache and os.path.exists(config["embeddings_file"]):
        print(f"Clearing embeddings cache at '{config['embeddings_file']}'...")
        os.remove(config["embeddings_file"])

    if args.model:
        config["openai"]["model"] = args.model

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    vault_content = []
    if os.path.exists(config["vault_file"]):
        print(f"Loading content from vault '{config['vault_file']}'...")
        with open(config["vault_file"], "r", encoding="utf-8") as vault_file:
            vault_content = vault_file.readlines()

    vault_embeddings_tensor = load_or_generate_embeddings(
        vault_content, config["embeddings_file"], client
    )

    conversation_history = []
    system_message = config["system_message"]

    print(PINK + "\nWelcome to the Email RAG System!" + RESET_COLOR)
    print(CYAN + "Type 'quit' to exit the chat." + RESET_COLOR)

    while True:
        user_input = input(
            YELLOW + "\nAsk a question about your emails: " + RESET_COLOR
        )
        if user_input.lower() == "quit":
            break

        try:
            response = chat_with_gpt(
                user_input,
                system_message,
                vault_embeddings_tensor,
                vault_content,
                config["openai"]["model"],
                conversation_history,
                config["top_k"],
                client,
            )
            print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
