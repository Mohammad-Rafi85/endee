#!/usr/bin/env python3
"""
Quick-start script: ingest the sample knowledge base into DocRAG.

Usage:
    python scripts/ingest_sample.py
    python scripts/ingest_sample.py --file path/to/doc.txt --api http://localhost:8000
"""

import argparse
import os
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Ingest a document into DocRAG")
    parser.add_argument("--file", default="data/sample_knowledge_base.txt")
    parser.add_argument("--api", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print(f"📄 Ingesting: {args.file}")
    print(f"🔗 API: {args.api}")

    with open(args.file, "rb") as f:
        r = requests.post(
            f"{args.api}/ingest",
            files={"file": (os.path.basename(args.file), f, "text/plain")},
            data={"doc_name": os.path.basename(args.file)},
            timeout=60,
        )

    if r.ok:
        d = r.json()
        print(f"✅ Indexed {d['chunks']} chunks from '{d['doc']}'")
    else:
        print(f"❌ Failed: {r.status_code} {r.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
