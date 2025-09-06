# RAG Pipeline - Data Ingestion

A comprehensive RAG (Retrieval-Augmented Generation) pipeline project focused on data ingestion using LangChain. This project demonstrates how to load and process various document types for building knowledge bases.

## 🚀 Features

- **Document Loading**: Support for text files and directory-based loading
- **Metadata Management**: Rich metadata tracking for document sources
- **LangChain Integration**: Built on LangChain for robust document processing
- **Progress Tracking**: Visual progress indicators for bulk operations
- **Jupyter Notebooks**: Interactive examples and tutorials

## 📁 Project Structure

```
├── data/                    # Sample documents and text files
│   ├── koh_samet.txt       # Travel guide content
│   ├── machine_learning.txt # ML concepts content
│   └── text_files/         # Additional text documents
├── notebook/               # Jupyter notebooks with examples
│   └── document.ipynb     # Data ingestion tutorials
├── main.py                # Main application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.13 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 02-build-rag-pipeline
   ```

2. **Install dependencies**
   
   Using uv (recommended):
   ```bash
   uv sync
   ```
   
   Alternative with pip:
   ```bash
   uv add -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python main.py
   ```

## 📚 Usage

### Basic Document Loading

```python
from langchain.document_loaders import TextLoader

# Load a single text file
loader = TextLoader("data/koh_samet.txt", encoding="utf-8")
documents = loader.load()
```

### Directory-based Loading

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all text files from a directory
loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    show_progress=True,
)
documents = loader.load()
```

### Document with Metadata

```python
from langchain_core.documents import Document

document = Document(
    page_content="Your content here",
    metadata={
        "source": "example.pdf",
        "pages": 1,
        "author": "Author Name",
        "date": "2024-01-01"
    }
)
```

## 🧪 Interactive Examples

Open the Jupyter notebook for hands-on examples:

```bash
jupyter notebook notebook/document.ipynb
```

The notebook includes:
- Document creation with metadata
- Text file loading
- Directory-based bulk loading
- Progress tracking examples

## 📦 Dependencies

- **langchain**: Core LangChain framework
- **langchain-core**: Core LangChain components
- **langchain-community**: Community-contributed loaders
- **pypdf**: PDF processing capabilities
- **pymupdf**: Advanced PDF handling
- **tqdm**: Progress bars for long operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Pipeline Examples](https://github.com/langchain-ai/langchain)

---

**Note**: This project is part of a larger RAG pipeline tutorial series. Check out the other modules for complete RAG implementation.
