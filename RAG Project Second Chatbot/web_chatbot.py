#!/usr/bin/env python3
"""
Web Chatbot & 3D Multiverse Server Entrypoint
Delegates to server.py to provide the full 3D WebGL spatial experience
and the complete 7-stage dynamic RAG file ingestion studio.
"""

from server import run_server
import sys

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
