# RAG & Document Uploader

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Open Issues](https://img.shields.io/github/issues/yourusername/yourproject.svg)](https://github.com/ayush-that/email-ai-agent/issues)

A powerful open source Retrieval Augmented Generation (RAG) and document management system that lets you:

- **Upload and Process Documents:** Use a simple Tkinter GUI to upload PDF, TXT, or JSON files. Contents are automatically chunked and stored in a vault file.
- **Generate Embeddings:** Generate and cache embeddings for document chunks using OpenAI's text-embedding-ada-002 model with PyTorch.
- **Local RAG System:** Query your documents and get relevant context with a conversation interface powered by ChatGPT-like models.
- **Email RAG System:** Process and search emails from IMAP (Gmail) and query them with contextual relevance.
- **Easy Configuration:** Configure project settings (e.g., file paths, OpenAI parameters, etc.) via the provided YAML configuration file.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Document Upload GUI](#document-upload-gui)
  - [Local RAG Query](#local-rag-query)
  - [Email RAG System](#email-rag-system)
  - [Generating Embeddings](#generating-embeddings)
  - [Collecting Emails](#collecting-emails)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Multi-Format Document Upload:** Easily upload and process PDFs, text files, and JSON files.
- **Document Chunking:** Intelligent text chunking based on sentence breaks and character limits ensures quality context extraction.
- **Embeddings Generation & Caching:** Uses OpenAI's embedding model and PyTorch to generate and store document embeddings for efficient similarity search.
- **Contextual Querying:** Retrieve relevant document context to power a conversation interface similar to ChatGPT.
- **Email Integration:** Automatically fetch, process, and search through emails to assist with document-based email queries.
- **Configurable and Extendable:** YAML-based configuration and clear code structure allow for extended customizations and contributions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Install Dependencies:**

   Make sure you have Python 3.8 or later installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables:**

   Create a `.env` file in your project root and add your OpenAI API key along with Gmail credentials (if using email features):

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   GMAIL_USERNAME=your_email@gmail.com
   GMAIL_PASSWORD=your_email_password_or_app_password
   ```

## Configuration

Project settings are controlled via the `config.yaml` file. You can adjust parameters such as:

- `vault_file`: Path to the document vault (default: "vault.txt")
- `embeddings_file`: File to cache embeddings (default: "vault_embeddings.json")
- `top_k`: Number of top similar contexts to return for queries
- `system_message`: The system message guiding the ChatGPT responses
- OpenAI model settings including temperature and max tokens

Example snippet from `config.yaml`:
