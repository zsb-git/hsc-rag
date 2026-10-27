from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

PDF_PATH = PROJECT_ROOT / "data" / "source_pdfs" / "Topic 1 International Economic Integration.pdf"
CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "econ_topic1"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOP_K = 5

ANSWER_MODEL = "claude-haiku-5-5"
ANSWER_MAX_TOKENS = 1024
ANSWER_EFFORT = "low"

CHAPTER_TITLE = "Topic 1: International Economic Integration"
