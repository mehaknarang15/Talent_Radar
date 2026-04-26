# 🎯 TALENT RADAR: AI SCOUTING & ENGAGEMENT ENGINE

---

## 🧠 1. The Approach: A Neuro-Symbolic Agentic Workflow
Traditional applicant tracking systems treat recruiting as a rigid text-matching problem. Talent Radar reimagines scouting as a multi-stage, neuro-symbolic agentic workflow that evaluates both technical constraints and behavioral psychology. Instead of an immediate keyword comparison, the system establishes a highly detailed baseline. It then processes candidates through parallel evaluations—calculating semantic match, simulating dynamic conversations, and applying mathematical reward/penalty models to behavioral signals. The result is a unified composite score that balances hard skills with genuine human interest, categorizing candidates into an actionable matrix.

## ⚙️ 2. Comprehensive Agent Capabilities
The system deploys specialized AI agents across five distinct phases:

* **📝 Phase 1: JD Analysis & Optimization:** The agent parses the raw job description to extract required skills, core domain, and implicit cultural signals. It then evaluates the JD itself, generating a Clarity Score, an Attractiveness Score, and an Overall Grade (A-F). It identifies missing hooks, flags structural red flags (e.g., missing salary bands), and provides explicit rewrite suggestions to help recruiters improve their pitch.

* **👤 Phase 2: Ghost Candidate Benchmarking:** The agent synthesizes an idealized "Ghost Candidate" profile based on the parsed JD. This includes an ideal title, optimal skill stack, a background narrative, and explicitly defined "must-have" vs. "nice-to-have" signals.

* **🔍 Phase 3: Semantic Candidate Matching:** Candidates are evaluated against both the JD and the Ghost Profile. The agent extracts inferred skills and calculates total experience. It generates a Match Score (0-100) and a Ghost Proximity Score (0-100). It explicitly lists satisfied constraints, missing constraints, and the "Ghost Delta" (what separates the candidate from the ideal), backed by semantic reasoning.

* **💬 Phase 4: Simulated Engagement & Behavioral Analysis:** For candidates passing a technical threshold (Match > 50), the agent simulates a realistic 3-turn conversational interview. It analyzes the transcript to calculate an Interest Score (0-100) and an Authenticity Score. It categorizes the candidate into a Motivation Archetype (Prestige-seeker, Growth-seeker, Stability-seeker, Misaligned) and actively flags Deception Signals (e.g., rehearsed or evasive answers).

* **📊 Phase 5: Recruiter Briefing & Output:** Finally, the agent packages the data for human action. It generates a concise 2-sentence summary, assigns a Hiring Recommendation (Strong Yes, Conditional Yes, Hold, No), identifies specific Risk Flags, and drafts exactly 3 targeted follow-up questions (with strategic reasoning) for the human recruiter to ask.

## 📈 3. Scoring Logic & The Decision Matrix
The final Composite Score is a weighted calculation:  
`Base Score = (Match * 0.40) + (Ghost Proximity * 0.25) + (Interest * 0.20) + (Authenticity * 0.15)`

Based on the raw Match and Interest scores, the agent plots every candidate into a strict 2x2 action matrix:

* **🚀 Q1: Fast-Track (Match ≥ 70, Interest ≥ 70):** High alignment + high engagement → move immediately to interviews  
* **🛠️ Q2: Needs Nurturing (Match ≥ 70, Interest < 70):** Strong technically but low engagement → recruiter must sell the role  
* **🌱 Q3: Enthusiastic but Underqualified (Match < 70, Interest ≥ 70):** High potential → consider for future/junior roles  
* **📦 Q4: Archive (Match < 70, Interest < 70):** Low fit → auto reject  

## 🏗️ 4. Trade-offs & Architecture
* **🤖 Simulated vs. Live Chat:** We simulate candidate responses instead of real-time chat.  
  **Trade-off:** Faster scalability vs. real human spontaneity  

* **📦 LLM Batching:** Candidates are processed in batches with retry logic.  
  **Trade-off:** Slight delay vs. better reliability and JSON consistency  

---

## 🚀 LOCAL SETUP INSTRUCTIONS

```bash
git clone https://github.com/yourusername/talent-radar.git
cd talent-radar
python -m venv venv
```

# Activate environment
# macOS/Linux
```bash
source venv/bin/activate
```

# Windows
```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Sample input file `job_description.txt` is present in the root directory: to paste the target JD in the input section when website opens


Run the application:
```bash
uvicorn main:app --reload
```

Open in browser:
http://127.0.0.1:8000

## 🎥 Demo Video
https://drive.google.com/file/d/1Oo9xoREh2fosoLaCx8ET7XiJ3jimUoJz/view?usp=sharing
