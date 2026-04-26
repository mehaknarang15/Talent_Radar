import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==========================================
# CONFIGURATION (GROQ)
# ==========================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile" 

def call_groq(prompt: str, system_instruction: str, response_schema: dict) -> dict:
    """Calls Groq API and forces JSON output matching the schema."""
    sys_prompt = (
        f"{system_instruction}\n\n"
        f"You MUST return ONLY a valid JSON object matching this exact schema:\n"
        f"{json.dumps(response_schema)}"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[!] Groq API Error: {e}")
        return _generate_mock_response(prompt, system_instruction)


def call_groq_text(prompt: str, system_instruction: str) -> str:
    """Free-text (non-JSON) Groq call for interview transcript generation."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[!] Groq Text API Error: {e}")
        return "Recruiter: Tell me about your experience.\nCandidate: I have worked on many relevant projects."

# ==========================================
# JSON SCHEMAS
# ==========================================

JD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "required_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "required_years_experience": {"type": "INTEGER"},
        "core_domain": {"type": "STRING"},
        "seniority_level": {"type": "STRING"},
        "implicit_culture_signals": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Implicit culture/values signals inferred from language style"
        }
    },
    "required": ["required_skills", "required_years_experience", "core_domain",
                 "seniority_level", "implicit_culture_signals"]
}

JD_QUALITY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "clarity_score": {"type": "INTEGER", "description": "0-100, how clear and specific the JD is"},
        "attractiveness_score": {"type": "INTEGER", "description": "0-100, how appealing the role sounds"},
        "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Phrases that deter strong candidates"},
        "missing_hooks": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Things strong candidates care about that are absent"},
        "rewrite_suggestions": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Specific edits to improve the JD"},
        "overall_jd_grade": {"type": "STRING", "description": "A, B, C, D, or F"}
    },
    "required": ["clarity_score", "attractiveness_score", "red_flags",
                 "missing_hooks", "rewrite_suggestions", "overall_jd_grade"]
}

GHOST_CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ideal_title": {"type": "STRING"},
        "ideal_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "ideal_years_experience": {"type": "INTEGER"},
        "ideal_background_narrative": {"type": "STRING",
                                       "description": "1-paragraph profile of the perfect candidate"},
        "must_have_signals": {
            "type": "ARRAY", "items": {"type": "STRING"},
            "description": "Non-negotiable signals that must appear in any winning resume"
        },
        "nice_to_have_signals": {
            "type": "ARRAY", "items": {"type": "STRING"},
            "description": "Bonus signals that differentiate great from good"
        }
    },
    "required": ["ideal_title", "ideal_skills", "ideal_years_experience",
                 "ideal_background_narrative", "must_have_signals", "nice_to_have_signals"]
}

CONSOLIDATED_PROFILE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "inferred_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "total_years_experience": {"type": "INTEGER"},
        "current_title": {"type": "STRING"},
        "personality_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "satisfied_constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missing_constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
        "ghost_delta": {"type": "ARRAY", "items": {"type": "STRING"}},
        "semantic_reasoning": {"type": "STRING"},
        "match_score": {"type": "INTEGER", "description": "0-100 match vs JD"},
        "ghost_proximity_score": {"type": "INTEGER", "description": "0-100 closeness to ghost"}
    },
    "required": ["inferred_skills", "total_years_experience", "current_title", 
                 "personality_signals", "satisfied_constraints", "missing_constraints", 
                 "ghost_delta", "semantic_reasoning", "match_score", "ghost_proximity_score"]
}
ENGAGEMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "motivation_archetype": {
            "type": "STRING",
            "description": "One of: Prestige-seeker, Growth-seeker, Stability-seeker, Misaligned"
        },
        "deception_signals": {
            "type": "ARRAY", "items": {"type": "STRING"},
            "description": "Specific phrases that suggest rehearsed, vague, or evasive answers"
        },
        "authenticity_score": {"type": "INTEGER", "description": "0-100, how genuine the candidate sounds"},
        "engagement_level": {"type": "STRING", "description": "Low, Medium, or High"},
        "behavioral_analysis": {"type": "STRING"},
        "interest_score": {"type": "INTEGER", "description": "0-100"}
    },
    "required": ["motivation_archetype", "deception_signals", "authenticity_score",
                 "engagement_level", "behavioral_analysis", "interest_score"]
}

FOLLOWUP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "recruiter_brief": {"type": "STRING",
                            "description": "2-sentence summary of this candidate for the hiring manager"},
        "followup_questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "reason": {"type": "STRING", "description": "Why this question targets a key gap"}
                },
                "required": ["question", "reason"]
            },
            "description": "Exactly 3 targeted follow-up questions for the human recruiter to ask"
        },
        "hiring_recommendation": {
            "type": "STRING",
            "description": "One of: Strong Yes, Conditional Yes, Hold, No"
        },
        "risk_flags": {
            "type": "ARRAY", "items": {"type": "STRING"},
            "description": "Red flags the recruiter should watch for in next conversation"
        }
    },
    "required": ["recruiter_brief", "followup_questions",
                 "hiring_recommendation", "risk_flags"]
}

# ==========================================
# AGENT CLASSES
# ==========================================
class JDParserAgent:
    def parse(self, jd_text: str) -> dict:
        print("  -> [JDParserAgent] Extracting requirements...")
        return call_groq(jd_text, "You are a senior HR analyst...", JD_SCHEMA)

class JDQualityAgent:
    def evaluate(self, jd_text: str) -> dict:
        print("  -> [JDQualityAgent] Grading the JD...")
        return call_groq(jd_text, "You are an expert talent branding consultant...", JD_QUALITY_SCHEMA)

class GhostCandidateAgent:
    def synthesize(self, jd_constraints: dict) -> dict:
        print("  -> [GhostCandidateAgent] Synthesizing ghost profile...")
        return call_groq(f"JD Requirements: {json.dumps(jd_constraints)}", "You are an expert headhunter...", GHOST_CANDIDATE_SCHEMA)

class ConsolidatedMatcherAgent:
    def evaluate(self, resume_text: str, jd_constraints: dict, ghost: dict) -> dict:
        print("  -> [ConsolidatedMatcherAgent] Matching vs JD/Ghost...")
        prompt = f"JD Constraints: {json.dumps(jd_constraints)}\nGhost: {json.dumps(ghost)}\nResume: {resume_text}"
        return call_groq(prompt, "You are an expert resume parser...", CONSOLIDATED_PROFILE_SCHEMA)

class DynamicChatSimulator:
    def simulate(self, name: str, parsed_resume: dict, jd_constraints: dict) -> str:
        print(f"  -> [DynamicChatSimulator] Generating interview for {name}...")
        prompt = f"Name: {name}\nJD: {json.dumps(jd_constraints)}\nResume: {json.dumps(parsed_resume)}"
        
        sys_instruction = (
            "Simulate a realistic text-message interview (3 turns). "
            "The recruiter MUST ask targeted questions based on this candidate's specific gaps or strengths. "
            "The candidate MUST respond naturally — if they lack a skill, they don't magically have it. "
            "CRITICAL INSTRUCTION: The candidate's tone, attitude, and level of enthusiasm MUST perfectly match the personality described in their resume. "
            "FORMATTING REQUIREMENT: You MUST format the output exactly like this with no bolding or markdown:\n"
            "Recruiter: [Text]\nCandidate: [Text]\nRecruiter: [Text] etc."
        )
        return call_groq_text(prompt, sys_instruction)

class EngagementEvaluatorAgent:
    def evaluate(self, transcript: str) -> dict:
        print("  -> [EngagementEvaluatorAgent] Evaluating engagement...")
        return call_groq(transcript, "You are a behavioral psychologist...", ENGAGEMENT_SCHEMA)

class RecruiterBriefAgent:
    def generate(self, name: str, match_data: dict, engagement_data: dict, ghost: dict, jd_constraints: dict) -> dict:
        print(f"  -> [RecruiterBriefAgent] Generating action brief for {name}...")
        prompt = f"Candidate: {name}\nMatch: {json.dumps(match_data)}\nEngagement: {json.dumps(engagement_data)}\nGhost: {json.dumps(ghost)}\nJD: {json.dumps(jd_constraints)}"
        return call_groq(prompt, "You are a senior talent partner...", FOLLOWUP_SCHEMA)

# ==========================================
# QUADRANT + SCORING LOGIC
# ==========================================

def determine_quadrant(match_score: int, interest_score: int) -> tuple:
    if match_score >= 70 and interest_score >= 70:
        return ("Q1: Fast-Track", "GREEN")
    elif match_score >= 70 and interest_score < 70:
        return ("Q2: Needs Nurturing", "YELLOW")
    elif match_score < 70 and interest_score >= 70:
        return ("Q3: Enthusiastic but Underqualified", "ORANGE")
    else:
        return ("Q4: Archive", "RED")

def composite_score(match: int, ghost_proximity: int, interest: int, authenticity: int) -> float:
    return round(
        (match * 0.40) +
        (ghost_proximity * 0.25) +
        (interest * 0.20) +
        (authenticity * 0.15),
        1
    )

def process_single_candidate(name, resume_text, jd_constraints, ghost, jd_quality_report, agents):
    print(f"  [Thread] Starting analysis for {name}...")
    
    consolidated_matcher, chat_sim, evaluator, brief_agent = agents
    
    match_data = consolidated_matcher.evaluate(resume_text, jd_constraints, ghost)
    match_score = match_data.get("match_score", 0)
    
    MATCH_THRESHOLD = 50
    transcript = None
    
    if match_score >= MATCH_THRESHOLD:
        print(f"  [Processing] ✅ {name} passed match threshold ({match_score}%). Initiating chat...")
        transcript = chat_sim.simulate(name, match_data, jd_constraints)
        engagement_data = evaluator.evaluate(transcript)
        brief = brief_agent.generate(name, match_data, engagement_data, ghost, jd_constraints)
    else:
        print(f"  [Processing] ⏭️ Skipping chat for {name} (Match Score: {match_score}% < {MATCH_THRESHOLD}%)")
        engagement_data = {
            "interest_score": 0,
            "authenticity_score": 0,
            "motivation_archetype": "Not Evaluated (Low Match)",
            "deception_signals": [],
            "behavioral_analysis": "Skipped engagement phase due to low technical match."
        }
        brief = {
            "hiring_recommendation": "No",
            "recruiter_brief": "Candidate did not meet the minimum technical match threshold to trigger automated outreach.",
            "followup_questions": [],
            "risk_flags": ["Does not meet core JD requirements"]
        }

    quadrant_label, _ = determine_quadrant(match_score, engagement_data.get("interest_score", 0))
    
    score = composite_score(
        match_score,
        match_data.get("ghost_proximity_score", 0),
        engagement_data.get("interest_score", 0),
        engagement_data.get("authenticity_score", 0)
    )

    print(f"  [Thread] Finished {name}!")
    
    return {
        "name": name,
        "composite_score": score,
        "match_score": match_score,
        "ghost_proximity_score": match_data.get("ghost_proximity_score", 0),
        "interest_score": engagement_data.get("interest_score", 0),
        "authenticity_score": engagement_data.get("authenticity_score", 0),
        "quadrant": quadrant_label,
        "motivation_archetype": engagement_data.get("motivation_archetype", "Unknown"),
        "deception_signals": engagement_data.get("deception_signals", []),
        "ghost_delta": match_data.get("ghost_delta", []),
        "semantic_reasoning": match_data.get("semantic_reasoning", ""),
        "behavioral_analysis": engagement_data.get("behavioral_analysis", ""),
        "hiring_recommendation": brief.get("hiring_recommendation", "No Recommendation"),
        "recruiter_brief": brief.get("recruiter_brief", "No brief available."),
        "followup_questions": brief.get("followup_questions", []),
        "risk_flags": brief.get("risk_flags", []),
        "chat_transcript": transcript
    }

# ==========================================
# PIPELINE
# ==========================================

def run_pipeline(raw_jd: str, synthetic_resumes: dict):
    print("\n" + "=" * 60)
    print("  TALENT RADAR — Advanced AI Talent Scouting Pipeline")
    print("=" * 60)
        
    jd_parser = JDParserAgent()
    jd_quality = JDQualityAgent()
    ghost_agent = GhostCandidateAgent()
    consolidated_matcher = ConsolidatedMatcherAgent()
    chat_sim = DynamicChatSimulator()
    evaluator = EngagementEvaluatorAgent()
    brief_agent = RecruiterBriefAgent()

    print("\n[PHASE 1: JD Analysis]")
    jd_constraints = jd_parser.parse(raw_jd)
    jd_quality_report = jd_quality.evaluate(raw_jd)

    print(f"\n  JD Grade: {jd_quality_report.get('overall_jd_grade', 'N/A')}")
    print(f"  Clarity: {jd_quality_report.get('clarity_score', 0)}/100  |  "
          f"Attractiveness: {jd_quality_report.get('attractiveness_score', 0)}/100")
    if jd_quality_report.get("red_flags"):
        print(f"  Red Flags: {', '.join(jd_quality_report['red_flags'])}")
    if jd_quality_report.get("rewrite_suggestions"):
        print(f"  Suggestions: {jd_quality_report['rewrite_suggestions'][0]}...")

    print("\n[PHASE 2: Ghost Candidate Synthesis]")
    ghost = ghost_agent.synthesize(jd_constraints)
    print(f"  Ghost Profile: {ghost.get('ideal_title', 'Unknown')} | "
          f"{ghost.get('ideal_years_experience', 0)}+ yrs | "
          f"Must-haves: {', '.join(ghost.get('must_have_signals', [])[:3])}")

    print("\n[PHASE 3: Analyzing Candidates Sequentially...]")
    results = []
     
    agents_bundle = (consolidated_matcher, chat_sim, evaluator, brief_agent)

    for name, resume_text in synthetic_resumes.items():
        try:
            candidate_result = process_single_candidate(
                name, 
                resume_text, 
                jd_constraints, 
                ghost, 
                jd_quality_report, 
                agents_bundle
            )
            results.append(candidate_result)
        except Exception as exc:
            print(f"  [!] Candidate {name} generated an exception: {exc}")

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    print("\n" + "=" * 60)
    print("  FINAL RANKED SHORTLIST — TALENT RADAR")
    print("=" * 60)

    for rank, r in enumerate(results, 1):
        print(f"\n#{rank}  {r['name']}  |  Composite: {r['composite_score']}/100  |  {r['quadrant']}")
        print(f"     Scores  → Match: {r['match_score']}  Ghost-Fit: {r['ghost_proximity_score']}  "
              f"Interest: {r['interest_score']}  Authenticity: {r['authenticity_score']}")
        print(f"     Archetype: {r['motivation_archetype']}  |  Rec: {r['hiring_recommendation']}")
        print(f"     Brief: {r['recruiter_brief']}")

    if len(results) > 0:
        avg_match = sum(r["match_score"] for r in results) / len(results)
        if avg_match < 65:
            print("\n[PHASE 5: JD IMPROVEMENT RECOMMENDATIONS]")
            print("  Average match score is low — your JD may be deterring strong candidates.")
            for s in jd_quality_report.get("rewrite_suggestions", []):
                print(f"    • {s}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)

    return results, ghost, jd_quality_report

# ==========================================
# MOCK DATA (Fallback when no API key)
# ==========================================
def _generate_mock_response(prompt: str, sys_instruction: str = "") -> dict:
    prompt_lower = prompt.lower()
    sys_lower = sys_instruction.lower()

    if "headhunter" in sys_lower or ("ideal" in sys_lower and "offer" in sys_lower):
        return {
            "ideal_title": "Senior Data/ML Engineer",
            "ideal_skills": ["Python", "SQL", "PyTorch", "AWS", "MLflow", "dbt"],
            "ideal_years_experience": 5,
            "ideal_background_narrative": "5 years building end-to-end ML pipelines.",
            "must_have_signals": ["Production ML deployment", "Cloud data pipeline", "SQL proficiency"],
            "nice_to_have_signals": ["OSS contributions", "MLOps tooling", "Team leadership"]
        }

    if "talent branding" in sys_lower:
        return {
            "clarity_score": 62,
            "attractiveness_score": 45,
            "red_flags": ["No salary range mentioned"],
            "missing_hooks": ["Salary range", "Remote/hybrid policy"],
            "rewrite_suggestions": ["Add salary range to attract serious candidates"],
            "overall_jd_grade": "C"
        }

    if "hr analyst" in sys_lower:
        return {
            "required_skills": ["Python", "SQL", "Machine Learning", "AWS/GCP"],
            "required_years_experience": 4,
            "core_domain": "Data Engineering",
            "seniority_level": "Senior",
            "implicit_culture_signals": ["Result-driven", "Technical depth valued"]
        }

    if "matching engine" in sys_lower or "resume parser" in sys_lower:
        return {
            "inferred_skills": ["Python", "SQL", "Machine Learning"],
            "total_years_experience": 3,
            "current_title": "Software Developer",
            "personality_signals": ["Eager to learn"],
            "satisfied_constraints": ["Python", "SQL"],
            "missing_constraints": ["AWS", "PyTorch"],
            "ghost_delta": ["Missing production ML deployment"],
            "semantic_reasoning": "Good basic match but missing specific cloud constraints.",
            "match_score": 65,
            "ghost_proximity_score": 50
        }

    if "behavioral psychologist" in sys_lower:
        return {"motivation_archetype": "Misaligned",
                "deception_signals": [],
                "authenticity_score": 61,
                "engagement_level": "Medium",
                "behavioral_analysis": "High energy but answers do not demonstrate understanding.",
                "interest_score": 72}

    if "talent partner" in sys_lower:
        return {
            "recruiter_brief": "Candidate has strong fundamentals but is missing critical skills.",
            "followup_questions": [
                {"question": "What would make you excited about this data team's work beyond the title?",
                 "reason": "Checks whether interest is genuine."}
            ],
            "hiring_recommendation": "No",
            "risk_flags": ["Significant skill gap"]
        }

    return {}

if __name__ == "__main__":
    dummy_jd = "Looking for a Python Developer with SQL experience."
    dummy_resumes = {"Alice": "5 years Python and PostgreSQL.", "Bob": "Junior Dev, knows Python."}
    run_pipeline(dummy_jd, dummy_resumes)