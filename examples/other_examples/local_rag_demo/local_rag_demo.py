"""
Local RAG Demo - Self-Managed Retrieval-Augmented Generation

This example demonstrates how to use the LocalRAG provider with pluggable
components for embeddings, vector storage, and chunking.

Unlike managed RAG (Gemini/OpenAI), with LocalRAG:
- Documents are chunked locally (never uploaded)
- Only text chunks are sent to embedding API (vectors returned to you)
- Vectors are stored locally in memory
- Search happens locally (no API call)

This gives you control over the pipeline while keeping data local.

Requirements:
- OPENAI_API_KEY environment variable set (for embeddings)

Usage:
    uv run python local_rag_demo.py
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from vision_agents.core.rag import (
    Document,
    FixedSizeChunker,
    InMemoryVectorStore,
    LocalRAG,
    OpenAIEmbeddings,
    SentenceChunker,
)
from vision_agents.core.rag.events import (
    RAGDocumentAddedEvent,
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
)
from vision_agents.plugins.openai import LLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample documents to demonstrate RAG capabilities
SAMPLE_DOCUMENTS = {
    "company_policy.txt": """
ACME Corporation Employee Handbook

VACATION POLICY
All full-time employees are entitled to 25 days of paid vacation per year.
Part-time employees receive vacation days proportional to their hours worked.
Vacation requests must be submitted at least 2 weeks in advance through the HR portal.
Unused vacation days can be carried over to the next year, up to a maximum of 5 days.
Vacation days do not accrue during unpaid leave.

REMOTE WORK POLICY
Employees may work remotely up to 3 days per week with manager approval.
Remote work requests must be submitted through the HR portal.
Employees must maintain regular working hours and be available for meetings.
A stable internet connection and appropriate workspace are required.
Remote work privileges may be revoked if performance standards are not met.

EXPENSE REIMBURSEMENT
Business expenses must be submitted within 30 days of the expense date.
Receipts are required for all expenses over $25.
Travel expenses require pre-approval from your department head.
Reimbursements are processed within 2 weeks of submission.
""",
    "product_faq.txt": """
ACME Widget Pro - Frequently Asked Questions

Q: What is the warranty period for the Widget Pro?
A: The Widget Pro comes with a 2-year limited warranty covering manufacturing defects.
   Extended warranty options are available for purchase within 30 days of the original purchase.

Q: How do I reset my Widget Pro to factory settings?
A: To reset your Widget Pro:
   1. Press and hold the power button for 10 seconds
   2. When the LED turns red, release the button
   3. Press the reset button (small hole on the back) with a paperclip
   4. Wait for the device to restart (approximately 30 seconds)
   5. The LED will turn green when the reset is complete

Q: What are the system requirements for the Widget Pro app?
A: The Widget Pro app requires:
   - iOS 15.0 or later / Android 11 or later
   - Bluetooth 5.0 or higher
   - 100MB free storage space
   - Active internet connection for initial setup

Q: How do I contact customer support?
A: You can reach our support team through:
   - Email: support@acme-widgets.example.com
   - Phone: 1-800-WIDGETS (available Mon-Fri, 9am-6pm EST)
   - Live chat: Available on our website 24/7

Q: Can I use the Widget Pro internationally?
A: Yes! The Widget Pro works in over 50 countries. However, some features
   may be limited based on local regulations. Check our website for a full
   list of supported countries and features.
""",
    "technical_specs.txt": """
ACME Widget Pro - Technical Specifications

HARDWARE
- Processor: ARM Cortex-M4 @ 120MHz
- Memory: 256KB RAM, 1MB Flash
- Connectivity: Bluetooth 5.0, Wi-Fi 802.11 b/g/n
- Battery: 2000mAh Li-Po, up to 72 hours standby
- Dimensions: 45mm x 45mm x 12mm
- Weight: 35g
- Operating Temperature: -10°C to 45°C

SENSORS
- 9-axis IMU (accelerometer, gyroscope, magnetometer)
- Ambient light sensor
- Temperature sensor (±0.5°C accuracy)
- Proximity sensor (up to 2m range)

CONNECTIVITY
- Bluetooth Low Energy (BLE) for mobile app connection
- Wi-Fi for firmware updates and cloud sync
- USB-C for charging and data transfer

POWER
- USB-C charging (5V/1A)
- Full charge time: 2 hours
- Battery life: Up to 7 days normal use, 72 hours standby

CERTIFICATIONS
- FCC Part 15 Class B
- CE Mark
- RoHS Compliant
- IP54 Water and Dust Resistance
""",
}


async def create_sample_files() -> list[str]:
    """Create temporary sample files for the demo."""
    temp_dir = tempfile.mkdtemp(prefix="local_rag_demo_")
    file_paths = []

    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = Path(temp_dir) / filename
        file_path.write_text(content)
        file_paths.append(str(file_path))
        logger.info(f"Created sample file: {file_path}")

    return file_paths


async def demo_with_sentence_chunker():
    """Demo using SentenceChunker for natural text boundaries."""
    print("\n" + "=" * 60)
    print("📝 Demo 1: LocalRAG with SentenceChunker")
    print("=" * 60)

    # Create LocalRAG with sentence-aware chunking
    rag = LocalRAG(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        vector_store=InMemoryVectorStore(),
        chunker=SentenceChunker(max_chunk_size=500, min_chunk_size=100),
    )

    # Subscribe to events
    @rag.events.subscribe
    async def on_doc_added(event: RAGDocumentAddedEvent):
        print(f"   📄 Added document with {event.chunk_count} chunks")

    @rag.events.subscribe
    async def on_retrieval(event: RAGRetrievalCompleteEvent):
        print(f"   🔎 Search completed in {event.retrieval_time_ms:.1f}ms")

    # Add documents directly (no file upload needed)
    print("\n📦 Adding documents with SentenceChunker...")
    docs = [
        Document(content=content, metadata={"filename": name})
        for name, content in SAMPLE_DOCUMENTS.items()
    ]
    await rag.add_documents(docs)
    await rag.events.wait()

    # Check vector store
    count = await rag.vector_store.count()
    print(f"\n✅ Vector store contains {count} chunks")

    # Search
    print("\n🔍 Searching for 'vacation policy'...")
    results = await rag.search("How many vacation days do employees get?", top_k=3)
    await rag.events.wait()

    print(f"\n📋 Top {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n   [{i + 1}] Score: {result.score:.3f}")
        print(f"       {result.content[:100]}...")

    return rag


async def demo_with_fixed_chunker():
    """Demo using FixedSizeChunker with overlap."""
    print("\n" + "=" * 60)
    print("📝 Demo 2: LocalRAG with FixedSizeChunker (overlap)")
    print("=" * 60)

    # Create LocalRAG with fixed-size chunking and overlap
    rag = LocalRAG(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        vector_store=InMemoryVectorStore(),
        chunker=FixedSizeChunker(chunk_size=300, overlap=50),
    )

    @rag.events.subscribe
    async def on_doc_added(event: RAGDocumentAddedEvent):
        print(f"   📄 Added document with {event.chunk_count} chunks")

    print("\n📦 Adding documents with FixedSizeChunker (300 chars, 50 overlap)...")
    docs = [
        Document(content=content, metadata={"filename": name})
        for name, content in SAMPLE_DOCUMENTS.items()
    ]
    await rag.add_documents(docs)
    await rag.events.wait()

    count = await rag.vector_store.count()
    print(f"\n✅ Vector store contains {count} chunks (more due to smaller size)")

    return rag


async def demo_with_llm():
    """Demo LocalRAG integrated with LLM."""
    print("\n" + "=" * 60)
    print("🤖 Demo 3: LocalRAG with LLM Integration")
    print("=" * 60)

    # Create LocalRAG
    rag = LocalRAG(
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        vector_store=InMemoryVectorStore(),
        chunker=SentenceChunker(max_chunk_size=500),
    )

    @rag.events.subscribe
    async def on_retrieval(event: RAGRetrievalCompleteEvent):
        print(
            f"   🔎 Retrieved {event.result_count} chunks in {event.retrieval_time_ms:.0f}ms"
        )

    # Add documents
    print("\n📦 Building local knowledge base...")
    docs = [
        Document(content=content, metadata={"filename": name})
        for name, content in SAMPLE_DOCUMENTS.items()
    ]
    await rag.add_documents(docs)

    # Create LLM and attach RAG
    print("\n🤖 Initializing LLM with LocalRAG...")
    llm = LLM(model="gpt-4o-mini")
    llm.set_rag_provider(rag, top_k=3, include_citations=True)

    # Demo queries
    queries = [
        "How many vacation days do employees get?",
        "How do I reset the Widget Pro?",
        "What's the battery capacity?",
    ]

    print("\n" + "-" * 60)
    print("💬 Q&A with automatic local retrieval")
    print("-" * 60)

    for query in queries:
        print(f"\n❓ {query}")
        response = await llm.simple_response(query)
        await rag.events.wait()
        response_text = response.text if hasattr(response, "text") else str(response)
        print(f"💡 {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    return rag


async def demo_from_files():
    """Demo adding files directly."""
    print("\n" + "=" * 60)
    print("📁 Demo 4: LocalRAG from Files")
    print("=" * 60)

    rag = LocalRAG(
        embeddings=OpenAIEmbeddings(),
        vector_store=InMemoryVectorStore(),
        chunker=SentenceChunker(max_chunk_size=400),
    )

    @rag.events.subscribe
    async def on_file_added(event: RAGFileAddedEvent):
        print(f"   ✅ Added: {Path(event.file_path).name}")

    # Create temp files
    print("\n📄 Creating sample files...")
    file_paths = await create_sample_files()

    print("\n📦 Adding files to LocalRAG...")
    for file_path in file_paths:
        await rag.add_file(file_path)
    await rag.events.wait()

    count = await rag.vector_store.count()
    print(f"\n✅ Indexed {len(file_paths)} files into {count} chunks")

    # Cleanup temp files
    for file_path in file_paths:
        Path(file_path).unlink(missing_ok=True)
    if file_paths:
        Path(file_paths[0]).parent.rmdir()

    return rag


async def main():
    print("\n" + "=" * 60)
    print("🔍 Local RAG Demo - Self-Managed RAG Pipeline")
    print("=" * 60)
    print("\nThis demo shows LocalRAG with:")
    print("  • Local chunking (documents never uploaded)")
    print("  • OpenAI embeddings (only chunks sent, vectors returned)")
    print("  • In-memory vector store (search is local)")
    print("  • Pluggable components (swap any part)")

    # Run demos
    await demo_with_sentence_chunker()
    await demo_with_fixed_chunker()
    await demo_with_llm()
    await demo_from_files()

    print("\n" + "=" * 60)
    print("✅ All demos complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
