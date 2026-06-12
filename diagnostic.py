"""
MARIS Diagnostic — Tests every step of the ingestion pipeline in isolation.
Run this to find exactly where things break.
"""
import sys, os, logging

# Enable full logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s")

# Step 0: Imports
print("=" * 60)
print("STEP 0: Imports")
try:
    from src.config import get_settings
    from src.storage.database import MARISDatabase
    from src.ingestion.arxiv_client import ArxivClient, ArxivPaper
    from src.ingestion.parser import PDFParser
    from src.storage.vector_store import VectorStore
    print("  ✅ All imports OK")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

settings = get_settings()
print(f"  Embedding provider: {settings.embedding_provider}")
print(f"  Embedding model: {settings.embedding_model}")

# Step 1: ArXiv Search
print("\n" + "=" * 60)
print("STEP 1: ArXiv Search")
db = MARISDatabase()
client = ArxivClient(db=db)
try:
    papers = client.search_papers("EEG brain computer interface deep learning", max_results=2)
    print(f"  ✅ Found {len(papers)} papers")
    for p in papers:
        print(f"     - [{p.arxiv_id}] {p.title[:60]}...")
        print(f"       PDF URL: {p.pdf_url}")
except Exception as e:
    print(f"  ❌ Search failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

if not papers:
    print("  ❌ No papers found. Exiting.")
    sys.exit(1)

paper = papers[0]

# Step 2: PDF Download (sync)
print("\n" + "=" * 60)
print(f"STEP 2: PDF Download (sync) for '{paper.arxiv_id}'")
try:
    pdf_path = client.download_pdf_sync(paper)
    if pdf_path and pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  ✅ Downloaded: {pdf_path} ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ download_pdf_sync returned: {pdf_path}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Download failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 3: PDF Parsing
print("\n" + "=" * 60)
print(f"STEP 3: PDF Parsing")
parser = PDFParser(db=db)
try:
    chunks = parser.parse_pdf(
        pdf_path=pdf_path,
        paper_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        pdf_url=paper.pdf_url,
    )
    print(f"  ✅ Parsed {len(chunks)} chunks")
    if chunks:
        for c in chunks[:3]:
            print(f"     - [{c.section}] {c.text[:80]}...")
    else:
        print(f"  ⚠️  0 chunks parsed! The PDF might be scanned/image-only.")
except Exception as e:
    print(f"  ❌ Parsing failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Step 4: Embedding & Indexing
print("\n" + "=" * 60)
print(f"STEP 4: Embedding & Indexing into Qdrant")
if not chunks:
    print("  ⚠️  Skipping (no chunks to index)")
else:
    vs = VectorStore(db=db)
    try:
        indexed = vs.index_chunks(chunks)
        print(f"  ✅ Indexed {indexed} / {len(chunks)} chunks")
    except Exception as e:
        print(f"  ❌ Indexing failed: {e}")
        import traceback; traceback.print_exc()

# Step 5: Hybrid Search
print("\n" + "=" * 60)
print(f"STEP 5: Hybrid Search")
try:
    results = vs.hybrid_search("EEG brain computer interface", top_k=3)
    print(f"  ✅ Found {len(results)} results")
    for r in results:
        print(f"     - score={r.get('rrf_score', 0):.4f} | {r.get('text', '')[:80]}...")
except Exception as e:
    print(f"  ❌ Search failed: {e}")
    import traceback; traceback.print_exc()

# Step 6: Stats
print("\n" + "=" * 60)
print("STEP 6: Database Stats")
stats = db.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
