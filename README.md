# Ilm Search

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=flat&logo=react&logoColor=61DAFB)
![Expo](https://img.shields.io/badge/Expo-000020?style=flat&logo=expo&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)

> **A hybrid semantic search engine for Islamic texts that bridges the gap between Natural Language Understanding (AI) and Deterministic Data Integrity.**

---

## 📖 The Problem
Standard AI models (like ChatGPT) often "hallucinate" religious citations, mixing up verse numbers or slightly altering texts. Traditional search engines require exact keyword matches (e.g., user must type "sadaqah" instead of "giving money").

## 💡 The Solution
**[Your Project Name]** utilizes a **Hybrid Retrieval Architecture**. It uses an LLM (GPT-4o-mini) strictly for *intent recognition* and *keyword extraction*, while performing the actual data retrieval against a local, immutable dataset of authentic texts (Quran & Sahih Hadith). 

**Result:** The flexibility of AI conversation with the mathematical accuracy of a database.

---

## 📸 Interface

| Home / Search | Quran Results | Hadith Results |
|:---:|:---:|:---:|
| <img src="./screenshots/home.png" width="250" /> | <img src="./screenshots/quran.png" width="250" /> | <img src="./screenshots/hadith.png" width="250" /> |

*(Note: Replace the paths above with your actual image paths)*

---

## 🚀 Key Features

* **Natural Language Querying:** Users can ask questions like *"How do I treat my wife?"* or *"What is the punishment for theft?"* without knowing specific Arabic terminology.
* **Zero-Hallucination Architecture:** By decoupling the *reasoning layer* (AI) from the *knowledge layer* (Local JSON), the app guarantees that every displayed verse and hadith exists 100% as written in the source texts.
* **Dual-Source Retrieval:** Simultaneously queries the **Quran** and major Hadith collections (**Sahih Bukhari**, **Sahih Muslim**, **Riyad as-Salihin**).
* **Smart Numbering Resolution:** Handles complex numbering inconsistencies (e.g., USC-MSA vs. Arabic numbering in Sahih Muslim) via intelligent mapping.
* **Optimized Latency:** Utilizes `gpt-4o-mini` for sub-second keyword extraction and local Python filtering for instant result rendering.

---

## 🛠 Technical Architecture

This project employs a **Client-Server** architecture:

### 1. The Frontend (React Native + Expo)
* Delivers a cross-platform (iOS/Android) native experience.
* Features a clean, card-based UI with "Split-Action" copying (Arabic vs. English).

### 2. The Backend (Python + FastAPI)
The backend functions as a **Semantic Proxy**:
1.  **Receive Query:** Endpoint accepts natural language string.
2.  **Intent Parsing (OpenAI):** The LLM analyzes the query and returns a JSON array of optimized search terms (e.g., Input: *"Being good to parents"* -> Output: `["filial piety", "parents", "birr", "kindness"]`).
3.  **Deterministic Search:** The Python service performs an O(n) scan on locally hosted, verified JSON datasets.
4.  **Response Aggregation:** Quran verses and Hadiths are unified into a single polymorphic JSON response.

---

## 💻 Getting Started

### Prerequisites
* Node.js & npm
* Python 3.9+
* OpenAI API Key

### Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
