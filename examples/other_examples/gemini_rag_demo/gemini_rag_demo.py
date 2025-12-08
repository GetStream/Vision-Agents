"""
Gemini RAG Demo - Retrieval-Augmented Generation with Gemini File Search

This example demonstrates how to use Gemini's native File Search tool for RAG.
The demo creates a knowledge base from sample documents and shows how to:
1. Upload files to Gemini's File Search Store
2. Attach the RAG provider to an LLM for automatic context injection
3. Query the knowledge base with natural language questions

Requirements:
- GOOGLE_API_KEY environment variable set

Usage:
    uv run python gemini_rag_demo.py
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from vision_agents.plugins.gemini import GeminiFileSearchRAG, LLM
from vision_agents.core.rag.events import (
    RAGFileAddedEvent,
    RAGRetrievalCompleteEvent,
)

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
    temp_dir = tempfile.mkdtemp(prefix="gemini_rag_demo_")
    file_paths = []

    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = Path(temp_dir) / filename
        file_path.write_text(content)
        file_paths.append(str(file_path))
        logger.info(f"Created sample file: {file_path}")

    return file_paths


async def main():
    print("\n" + "=" * 60)
    print("🔍 Gemini RAG Demo - File Search with Retrieval-Augmented Generation")
    print("=" * 60 + "\n")

    # Create the RAG provider
    print("📦 Initializing Gemini File Search RAG provider...")
    rag = GeminiFileSearchRAG()

    # Subscribe to events for observability
    @rag.events.subscribe
    async def on_file_added(event: RAGFileAddedEvent):
        print(f"   ✅ Uploaded: {Path(event.file_path).name}")

    @rag.events.subscribe
    async def on_retrieval_complete(event: RAGRetrievalCompleteEvent):
        print(f"   🔎 Retrieved {event.result_count} results in {event.retrieval_time_ms:.0f}ms")

    # Create and upload sample files
    print("\n📄 Creating sample documents...")
    file_paths = await create_sample_files()

    print("\n☁️  Uploading documents to Gemini File Search Store...")
    for file_path in file_paths:
        await rag.add_file(file_path)

    # Wait for events to be processed
    await rag.events.wait()

    print(f"\n✅ File Search Store created: {rag.store_name}")

    # Create LLM and attach RAG provider
    print("\n🤖 Initializing Gemini LLM with RAG...")
    llm = LLM(model="gemini-2.0-flash")
    llm.set_rag_provider(rag, top_k=5, include_citations=True)

    # Demo queries
    queries = [
        "How many vacation days do employees get per year?",
        "How do I reset the Widget Pro to factory settings?",
        "What is the battery life of the Widget Pro?",
        "How can I contact customer support?",
        "Can I carry over unused vacation days?",
    ]

    print("\n" + "-" * 60)
    print("💬 Demo: Asking questions about the knowledge base")
    print("-" * 60)

    for query in queries:
        print(f"\n❓ Question: {query}")

        # The RAG provider automatically augments the query with relevant context
        response = await llm.simple_response(query)

        print(f"💡 Answer: {response.text[:500]}{'...' if len(response.text) > 500 else ''}")

    # Cleanup
    print("\n" + "-" * 60)
    print("🧹 Cleaning up...")
    await rag.clear()

    # Clean up temp files
    for file_path in file_paths:
        Path(file_path).unlink(missing_ok=True)
    if file_paths:
        Path(file_paths[0]).parent.rmdir()

    print("✅ Demo complete!")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

