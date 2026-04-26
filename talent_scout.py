"""
╔══════════════════════════════════════════════════════════════╗
║   TALENT RADAR — AI Talent Scouting & Engagement Agent v2   ║
║   Hackathon Edition                                          ║
╚══════════════════════════════════════════════════════════════╝

UNIQUE DIFFERENTIATORS vs standard pipelines:
  1. GHOST CANDIDATE BENCHMARKING — LLM synthesizes a "perfect" candidate 
     profile from the JD, then measures real candidates against the ghost.
  2. REVERSE JOB SCORING — AI scores the JD itself: Is it well-written?
     Attractive? Does it have red flags that deter strong candidates?
  3. MOTIVATION ARCHETYPE DETECTION — Goes beyond "interested/not interested".
     Classifies candidates as Prestige-seeker, Growth-seeker, Stability-seeker,
     or Misaligned to predict culture fit.
  4. DECEPTION SIGNAL ANALYSIS — Flags interview responses that show rehearsed, 
     vague, or evasive language vs genuine enthusiasm.
  5. RECRUITER ACTION BRIEF — Generates a customized set of 3 follow-up 
     questions for each candidate, targeting detected skill gaps.
  6. DYNAMIC JD IMPROVEMENT SUGGESTIONS — If the JD is attracting weak 
     candidates, the AI suggests how to rewrite it.
"""
import os
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==========================================
# CONFIGURATION (GROQ)
# ==========================================
# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Llama 3.3 70B is currently the most capable model on Groq for logic and JSON tasks
MODEL = "llama-3.3-70b-versatile" 

def call_groq(prompt: str, system_instruction: str, response_schema: dict) -> dict:
    """Calls Groq API with retry + safe fallback."""

    sys_prompt = (
        f"{system_instruction}\n\n"
        f"You MUST return ONLY a valid JSON object matching this exact schema:\n"
        f"{json.dumps(response_schema)}"
    )

    for attempt in range(3):
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
            error_str = str(e)

            if "rate_limit" in error_str or "429" in error_str:
                wait_time = 10 * (attempt + 1)
                print(f"[Retry {attempt+1}] Rate limit hit. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[!] Groq API Error (non-retryable): {e}")
                break

    print("[Fallback] Using mock response.")
    return _generate_mock_response(prompt, system_instruction)

def call_groq_text(prompt: str, system_instruction: str) -> str:
    """Free-text Groq call with retry + fallback."""

    for attempt in range(3):
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
            error_str = str(e)

            if "rate_limit" in error_str or "429" in error_str:
                wait_time = 10 * (attempt + 1)
                print(f"[Retry {attempt+1}] Rate limit hit (text). Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[!] Groq Text API Error: {e}")
                break

    print("[Fallback] Using default transcript.")
    return "Recruiter: Tell me about your experience.\nCandidate: I have worked on relevant projects."

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

FULL_PIPELINE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "jd_analysis": JD_QUALITY_SCHEMA,
        "jd_parsed": JD_SCHEMA,
        "ghost_candidate": GHOST_CANDIDATE_SCHEMA,
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "match_data": CONSOLIDATED_PROFILE_SCHEMA,
                    "chat_transcript": {"type": "STRING"},
                    "engagement": ENGAGEMENT_SCHEMA,
                    "recruiter_brief": FOLLOWUP_SCHEMA
                },
                "required": ["name", "match_data", "chat_transcript", "engagement", "recruiter_brief"]
            }
        }
    },
    "required": ["jd_analysis", "jd_parsed", "ghost_candidate", "candidates"]
}

# ==========================================
# QUADRANT + SCORING LOGIC
# ==========================================

def determine_quadrant(match_score: int, interest_score: int) -> tuple:
    """Returns (quadrant_label, color_code) for display."""
    if match_score >= 70 and interest_score >= 70:
        return ("Q1: Fast-Track", "GREEN")
    elif match_score >= 70 and interest_score < 70:
        return ("Q2: Needs Nurturing", "YELLOW")
    elif match_score < 70 and interest_score >= 70:
        return ("Q3: Enthusiastic but Underqualified", "ORANGE")
    else:
        return ("Q4: Archive", "RED")

def deception_penalty(deception_signals: list) -> int:
    """
    Penalize based on number + severity of deception signals.
    """
    if not deception_signals:
        return 0
    
    # Light vs heavy penalty
    base = len(deception_signals) * 3  # each signal = -3

    # Escalate if multiple signals
    if len(deception_signals) >= 3:
        base += 5

    return min(base, 15)  # cap penalty

def archetype_modifier(archetype: str) -> int:
    """
    Boost or penalize based on motivation type.
    """
    mapping = {
        "Growth-seeker": +8,
        "Prestige-seeker": +3,
        "Stability-seeker": +1,
        "Misaligned": -10,
        "Not Evaluated (Low Match)": 0
    }
    return mapping.get(archetype, 0)

def engagement_modifier(level: str) -> int:
    """
    Small bonus for engagement quality.
    """
    return {
        "High": +5,
        "Medium": +2,
        "Low": 0
    }.get(level, 0)

def composite_score(
    match: int,
    ghost_proximity: int,
    interest: int,
    authenticity: int,
    archetype: str,
    deception_signals: list,
    engagement_level: str
) -> float:
    """
    Advanced composite score with behavioral intelligence.
    """

    base_score = (
        (match * 0.35) +
        (ghost_proximity * 0.25) +
        (interest * 0.20) +
        (authenticity * 0.20)
    )

    # Behavioral adjustments
    penalty = deception_penalty(deception_signals)
    archetype_bonus = archetype_modifier(archetype)
    engagement_bonus = engagement_modifier(engagement_level)

    final = base_score + archetype_bonus + engagement_bonus - penalty

    return round(max(0, min(final, 100)), 1)

def run_full_pipeline_single_call(jd_text: str, resumes: dict):
    print("[ONE CALL PIPELINE]")

    resumes_text = "\n\n".join([
        f"Candidate: {name}\nResume:\n{text}"
        for name, text in resumes.items()
    ])

    prompt = f"""
JOB DESCRIPTION:
{jd_text}

CANDIDATES:
{resumes_text}

INSTRUCTIONS:
You must perform ALL of the following:

1. Parse JD
2. Evaluate JD quality
3. Create ghost candidate
4. Evaluate EACH candidate:
   - match vs JD
   - ghost proximity
   - simulate chat with EXACTLY 3 Q&A turns (ONLY if match_score >= 60). Format as "Q: ...
A: ..." repeated 3 times
   - evaluate engagement
   - generate recruiter brief

IMPORTANT:
- Generate followup_questions with exactly 3 targeted questions for every candidate
- Only include chat_transcript if match_score >= 60, otherwise set chat_transcript to ""
- Return structured JSON ONLY
"""

    response = call_groq(
        prompt,
        "You are a world-class AI hiring system executing a full recruitment pipeline.",
        FULL_PIPELINE_SCHEMA
    )

    return response

# ==========================================
# PIPELINE
# ==========================================

def run_pipeline(raw_jd: str, synthetic_resumes: dict):
    data = run_full_pipeline_single_call(raw_jd, synthetic_resumes)

    results = []

    for c in data["candidates"]:
        match = c.get("match_data", {})
        engagement = c.get("engagement") or {}  
        brief = c.get("recruiter_brief", {})

        score = composite_score(
            match.get("match_score", 0),
            match.get("ghost_proximity_score", 0),
            engagement.get("interest_score", 0),
            engagement.get("authenticity_score", 0),
            engagement.get("motivation_archetype", ""),
            engagement.get("deception_signals", []),
            engagement.get("engagement_level", "Low")
        )

        quadrant, _ = determine_quadrant(
            match.get("match_score", 0),
            engagement.get("interest_score", 0)
        )

        results.append({
            "name": c["name"],
            "composite_score": score,
            "match_score": match.get("match_score", 0),
            "ghost_proximity_score": match.get("ghost_proximity_score", 0),
            "interest_score": engagement.get("interest_score", 0),
            "authenticity_score": engagement.get("authenticity_score", 0),
            "quadrant": quadrant,
            "motivation_archetype": engagement.get("motivation_archetype", ""),
            "deception_signals": engagement.get("deception_signals", []),
            "ghost_delta": match.get("ghost_delta", []),
            "semantic_reasoning": match.get("semantic_reasoning", ""),
            "behavioral_analysis": engagement.get("behavioral_analysis", ""),
            "hiring_recommendation": brief.get("hiring_recommendation", ""),
            "recruiter_brief": brief.get("recruiter_brief", ""),
            "followup_questions": brief.get("followup_questions", []),
            "risk_flags": brief.get("risk_flags", []),
            "chat_transcript": c.get("chat_transcript", "")
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)

    return results, data["ghost_candidate"], data["jd_analysis"]


# ==========================================
# MOCK DATA (Fallback when no API key)
# ==========================================

def _generate_mock_response(prompt: str, sys_instruction: str = "") -> dict:
    prompt_lower = prompt.lower()
    sys_lower = sys_instruction.lower()

    # GhostCandidateAgent — unique phrase: "build a detailed profile" or "headhunter"
    if "headhunter" in sys_lower or ("ideal" in sys_lower and "offer" in sys_lower):
        return {
            "ideal_title": "Senior Data/ML Engineer",
            "ideal_skills": ["Python", "SQL", "PyTorch", "AWS", "MLflow", "dbt"],
            "ideal_years_experience": 5,
            "ideal_background_narrative": "5 years building end-to-end ML pipelines at a data-driven company, comfortable bridging data engineering and applied ML.",
            "must_have_signals": ["Production ML deployment", "Cloud data pipeline", "SQL proficiency"],
            "nice_to_have_signals": ["OSS contributions", "MLOps tooling", "Team leadership"]
        }

    # JDQualityAgent — unique phrase: "talent branding consultant"
    if "talent branding" in sys_lower:
        return {
            "clarity_score": 62,
            "attractiveness_score": 45,
            "red_flags": ["No salary range mentioned", "No mention of team size or impact"],
            "missing_hooks": ["Salary range", "Remote/hybrid policy", "Tech stack depth", "Growth opportunities"],
            "rewrite_suggestions": [
                "Add salary range to attract serious candidates",
                "Describe what joining the data team means in terms of scale and impact",
                "Replace exposure to ML with concrete examples such as deploy models to production"
            ],
            "overall_jd_grade": "C"
        }

    # JDParserAgent — unique phrase: "senior hr analyst"
    if "hr analyst" in sys_lower:
        return {
            "required_skills": ["Python", "SQL", "Machine Learning", "AWS/GCP"],
            "required_years_experience": 4,
            "core_domain": "Data Engineering",
            "seniority_level": "Senior",
            "implicit_culture_signals": ["Result-driven", "Technical depth valued", "Minimal bureaucracy"]
        }

    # ConsolidatedMatcherAgent
    if "matching engine" in sys_lower or "resume parser" in sys_lower:
        if "alice" in prompt_lower: # Example specific mock
            return {
                "inferred_skills": ["Python", "PostgreSQL", "AWS", "PyTorch", "MLOps"],
                "total_years_experience": 5, 
                "current_title": "Data Engineer",
                "personality_signals": ["Intrinsically motivated", "Detail-oriented"],
                "satisfied_constraints": ["Python", "SQL (PostgreSQL)", "ML (PyTorch)", "AWS", "5yr exp"],
                "missing_constraints": [], 
                "ghost_delta": ["No MLflow/experiment tracking mentioned"],
                "semantic_reasoning": "Near-perfect match. PostgreSQL satisfies SQL, PyTorch satisfies ML.",
                "match_score": 93, 
                "ghost_proximity_score": 85
            }
        # Default mock for all other candidates
        return {
            "inferred_skills": ["Python", "SQL", "Machine Learning"],
            "total_years_experience": 3,
            "current_title": "Software Developer",
            "personality_signals": ["Eager to learn", "High energy"],
            "satisfied_constraints": ["Python", "SQL"],
            "missing_constraints": ["AWS", "PyTorch"],
            "ghost_delta": ["Missing production ML deployment"],
            "semantic_reasoning": "Good basic match but missing specific deep learning and cloud constraints.",
            "match_score": 65,
            "ghost_proximity_score": 50
        }

    # EngagementEvaluatorAgent — unique phrase: "deception analyst"
    if "deception analyst" in sys_lower:
        if "alice" in prompt_lower:
            return {"motivation_archetype": "Growth-seeker",
                    "deception_signals": [],
                    "authenticity_score": 91,
                    "engagement_level": "High",
                    "behavioral_analysis": "Asks thoughtful, specific technical questions. Genuine enthusiasm.",
                    "interest_score": 88}
        if "bob" in prompt_lower:
            return {"motivation_archetype": "Prestige-seeker",
                    "deception_signals": ["Deflects technical depth", "Only asks about compensation"],
                    "authenticity_score": 52,
                    "engagement_level": "Low",
                    "behavioral_analysis": "Short, transactional answers. Focus on title and salary, not work.",
                    "interest_score": 38}
        return {"motivation_archetype": "Misaligned",
                "deception_signals": ["Generic enthusiasm not tied to role specifics"],
                "authenticity_score": 61,
                "engagement_level": "Medium",
                "behavioral_analysis": "High energy but answers do not demonstrate understanding of the role.",
                "interest_score": 72}

    # RecruiterBriefAgent — unique phrase: "talent partner"
    if "talent partner" in sys_lower:
        if "alice" in prompt_lower:
            return {
                "recruiter_brief": "Alice is a strong match — production ML + AWS + PostgreSQL covers all requirements. Ghost proximity is the highest of the pool. Recommend moving to technical interview.",
                "followup_questions": [
                    {"question": "Walk me through an MLflow or experiment tracking setup you've built.",
                     "reason": "Ghost profile requires MLOps experience; not explicit in resume."},
                    {"question": "How large were the datasets you ran PyTorch models on, and how did you handle scale?",
                     "reason": "Validates depth of ML production claim."},
                    {"question": "Describe a data quality issue you caught that others missed.",
                     "reason": "Tests the 'care deeply about data quality' personality signal."}
                ],
                "hiring_recommendation": "Strong Yes",
                "risk_flags": ["May have competing offers — move quickly"]
            }
        if "bob" in prompt_lower:
            return {
                "recruiter_brief": "Bob has strong Python/Cloud fundamentals but is missing ML and SQL — critical for this role. Compensation-focused signals suggest potential culture mismatch.",
                "followup_questions": [
                    {"question": "Have you worked with any data transformation or querying tools, even casually?",
                     "reason": "Probing for hidden SQL exposure not listed."},
                    {"question": "What would make you excited about this data team's work beyond the title?",
                     "reason": "Checks whether interest is genuine or purely transactional."},
                    {"question": "Describe a project where you had to learn a new technical domain under time pressure.",
                     "reason": "Assesses learning agility to fill ML gap."}
                ],
                "hiring_recommendation": "Conditional Yes",
                "risk_flags": ["Salary expectations may exceed budget", "ML gap may require 6+ months to close"]
            }
        return {
            "recruiter_brief": "Charlie is highly enthusiastic but significantly underqualified. Consider for a junior pipeline if one exists.",
            "followup_questions": [
                {"question": "What's the most complex Python project you've completed, and what was the result?",
                 "reason": "Validates actual coding depth beyond bootcamp curriculum."},
                {"question": "Have you deployed anything to a cloud environment, even a personal project?",
                 "reason": "Checks cloud familiarity hidden by bootcamp framing."},
                {"question": "Where do you see yourself in 2 years technically?",
                 "reason": "Tests whether Growth archetype is real or performative."}
            ],
            "hiring_recommendation": "No",
            "risk_flags": ["Significant skill gap", "May over-promise and under-deliver"]
        }

    return {
    "jd_analysis": {
        "clarity_score": 60,
        "attractiveness_score": 50,
        "red_flags": [],
        "missing_hooks": [],
        "rewrite_suggestions": [],
        "overall_jd_grade": "C"
    },
    "jd_parsed": {
        "required_skills": ["Python", "SQL"],
        "required_years_experience": 3,
        "core_domain": "Data",
        "seniority_level": "Mid",
        "implicit_culture_signals": []
    },
    "ghost_candidate": {
        "ideal_title": "Data Analyst",
        "ideal_skills": ["Python", "SQL"],
        "ideal_years_experience": 3,
        "ideal_background_narrative": "Strong data background",
        "must_have_signals": [],
        "nice_to_have_signals": []
    },
    "candidates": [
        {
            "name": "Mock Candidate",
            "match_data": {
                "inferred_skills": ["Python"],
                "total_years_experience": 3,
                "current_title": "Analyst",
                "personality_signals": [],
                "satisfied_constraints": [],
                "missing_constraints": [],
                "ghost_delta": [],
                "semantic_reasoning": "Mock reasoning",
                "match_score": 70,
                "ghost_proximity_score": 60
            },
            "chat_transcript": "Recruiter: Tell me about yourself.\nCandidate: I have experience in Python.",
            "engagement": {
                "motivation_archetype": "Growth-seeker",
                "deception_signals": [],
                "authenticity_score": 80,
                "engagement_level": "High",
                "behavioral_analysis": "Good",
                "interest_score": 75
            },
            "recruiter_brief": {
                "recruiter_brief": "Good candidate",
                "followup_questions": [],
                "hiring_recommendation": "Yes",
                "risk_flags": []
            }
        }
    ]
}


if __name__ == "__main__":
    run_pipeline()