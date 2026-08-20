# 👁️ CAD-Vision-MCP

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CAD-Vision-MCP** is a robust Model Context Protocol (MCP) server that bridges the gap between Large Language Models (LLMs) and complex engineering documents. By leveraging FastAPI, Anthropic's Claude, and advanced geometric/image processing libraries, this server empowers AI agents with the ability to "see," parse, and analyze CAD drawings (`.dxf`) and PDF documents.

## ✨ Key Features

* **CAD/DXF Parsing:** Deep analysis of `.dxf` files using `ezdxf`, evaluating geometries, lines, curves, and layers.
* **Spatial & Geometric Math:** Uses `shapely`, `numpy`, and `scipy` for complex spatial reasoning, bounding box calculations, and structural validations.
* **PDF & Image Processing:** Extracts and processes visual data from PDFs using `PyMuPDF` and `Pillow`.
* **Vision LLM Integration:** Built-in connection to Anthropic's Vision API to run intelligent context analysis on engineering drawings.
* **FastAPI Backend:** High-performance, async-ready REST/MCP interfaces.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone [https://github.com/RaghavO222/CAD-Vision-MCP.git](https://github.com/RaghavO222/CAD-Vision-MCP.git)
cd CAD-Vision-MCP

2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Dependencies
Install all required packages in a single command:
pip install fastapi uvicorn ezdxf pymupdf shapely scipy numpy pillow httpx requests anthropic dotenv python-multipart

⚙️ Configuration
Create a .env file in the root directory of your project to store your API keys and environment variables securely:
# .env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HOST=0.0.0.0
PORT=8000

🏃‍♂️ Usage
Start the MCP server using Uvicorn:
uvicorn server:app --reload --host 0.0.0.0 --port 8000

Once the server is running, you can access the automatic interactive API documentation at:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

🛠️ Tech Stack & Dependencies
Core API: fastapi, uvicorn, python-multipart
CAD Processing: ezdxf, shapely, scipy, numpy
PDF & Vision Processing: pymupdf (fitz), pillow
LLM & HTTP: anthropic, httpx, requests
Config Management: python-dotenv
