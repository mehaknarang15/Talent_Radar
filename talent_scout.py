import os
import time
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==========================================
# CONFIGURATION (GROQ)
# ==========================================
# Initialize the Groq client securely
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant" 

def call_groq(prompt: str, system_instruction: str, response_schema: dict) -> dict:
    """
    Calls Groq API with automatic retry, exponential backoff, and safe mock fallback.
    Enforces a strict JSON structure based on the provided schema.
    """
    sys_prompt = (
        f"{system_instruction}\n\n"
        f"You MUST return ONLY a valid JSON object matching this exact schema:\n"
        f"{json.dumps(response_schema)}"
    )

    for attempt in range(5):  
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
                wait_time = (2 ** attempt) * 10  # Exponential: 10s, 20s, 40s, 80s, 160s
                print(f"[Retry {attempt+1}] Rate limit hit. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[!] Groq API Error (non-retryable): {e}")
                break

    print("[Fallback] Using mock response due to repeated failures or API errors.")
    return _generate_mock_response(prompt, system_instruction)

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

BATCH_CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
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
    "required": ["candidates"]
}

JD_SETUP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "jd_analysis": JD_QUALITY_SCHEMA,
        "jd_parsed": JD_SCHEMA,
        "ghost_candidate": GHOST_CANDIDATE_SCHEMA,
    },
    "required": ["jd_analysis", "jd_parsed", "ghost_candidate"]
}

# ==========================================
# QUADRANT + SCORING LOGIC
# ==========================================

def determine_quadrant(match_score: int, interest_score: int) -> Tuple[str, str]:
    """Categorizes candidate into an actionable HR quadrant based on scores."""
    if match_score >= 70 and interest_score >= 70:
        return ("Q1: Fast-Track", "GREEN")
    elif match_score >= 70 and interest_score < 70:
        return ("Q2: Needs Nurturing", "YELLOW")
    elif match_score < 70 and interest_score >= 70:
        return ("Q3: Enthusiastic but Underqualified", "ORANGE")
    else:
        return ("Q4: Archive", "RED")

def deception_penalty(deception_signals: list) -> int:
    """Calculates a score penalty based on the presence of deceptive interview signals."""
    if not deception_signals:
        return 0
    
    base = len(deception_signals) * 3  # Light penalty per signal

    if len(deception_signals) >= 3:
        base += 5  # Escalation if multiple signals exist

    return min(base, 15)  # Cap maximum penalty

def archetype_modifier(archetype: str) -> int:
    """Adjusts score based on the candidate's core psychological motivation."""
    mapping = {
        "Growth-seeker": +8,
        "Prestige-seeker": +3,
        "Stability-seeker": +1,
        "Misaligned": -10,
        "Not Evaluated (Low Match)": 0
    }
    return mapping.get(archetype, 0)

def engagement_modifier(level: str) -> int:
    """Applies a minor adjustment based on active interview engagement."""
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
    Generates a unified assessment score combining raw technical match 
    with sophisticated behavioral intelligence.
    """
    base_score = (
        (match * 0.35) +
        (ghost_proximity * 0.25) +
        (interest * 0.20) +
        (authenticity * 0.20)
    )

    penalty = deception_penalty(deception_signals)
    archetype_bonus = archetype_modifier(archetype)
    engagement_bonus = engagement_modifier(engagement_level)

    final = base_score + archetype_bonus + engagement_bonus - penalty

    return round(max(0, min(final, 100)), 1)

# ==========================================
# AGENT EVALUATION LOGIC
# ==========================================

def _evaluate_candidate_batch(jd_text: str, ghost_text: str, batch: list) -> list:
    """Processes a chunk of candidates through the simulation and scoring models."""
    resumes_text = "\n\n".join([
        f"Candidate: {name}\nResume:\n{text}"
        for name, text in batch
    ])

    prompt = f"""
JOB DESCRIPTION:
{jd_text}

IDEAL GHOST CANDIDATE PROFILE:
{ghost_text}

CANDIDATES TO EVALUATE:
{resumes_text}

INSTRUCTIONS:
For EACH candidate above:
1. Evaluate match vs JD and ghost profile (match_score, ghost_proximity_score)
2. If match_score >= 70: simulate a realistic 3-turn interview.
   Format the chat_transcript field EXACTLY as plain text like this:
   Recruiter: [question text here]
   Candidate: [answer text here]
   Recruiter: [question text here]
   Candidate: [answer text here]
   Recruiter: [question text here]
   Candidate: [answer text here]
   Use ONLY "Recruiter:" and "Candidate:" as prefixes. Do NOT use Q: or A:. Do NOT use JSON inside chat_transcript — it must be a plain string.
   If match_score < 70: set chat_transcript to ""
3. Evaluate engagement, motivation archetype, interest_score
4. Generate recruiter_brief with exactly 3 followup_questions (for every candidate)

CRITICAL: Return ALL {len(batch)} candidates. Do NOT skip, omit, or merge any candidate. Return structured JSON ONLY.
"""
    response = call_groq(
        prompt,
        "You are a world-class AI hiring system. Evaluate every single candidate listed. Do not skip any.",
        BATCH_CANDIDATE_SCHEMA
    )
    return response.get("candidates", [])

def run_full_pipeline_single_call(jd_text: str, resumes: dict) -> dict:
    """
    Orchestrates the entire analytical process:
    Phase 1: Deep JD analysis and 'Ghost' candidate synthesis.
    Phase 2: Batched candidate evaluation to respect rate limits.
    """
    print("[BATCHED PIPELINE] Phase 1: JD setup...")

    jd_prompt = f"""
JOB DESCRIPTION:
{jd_text}

INSTRUCTIONS:
1. Parse the JD (required skills, experience, domain, seniority, culture signals)
2. Score the JD quality (clarity, attractiveness, red flags, missing hooks, rewrite suggestions, grade)
   NOTE: Red flags must be about JD structure only (vague requirements, missing salary, excessive demands).
   Company values and mission statements are NEVER red flags.
3. Create a ghost candidate profile (the ideal hire)

Return structured JSON ONLY.
"""
    setup = call_groq(
        jd_prompt,
        "You are a world-class AI hiring system.",
        JD_SETUP_SCHEMA
    )

    ghost = setup.get("ghost_candidate", {})
    ghost_text = (
        f"Title: {ghost.get('ideal_title', '')}\n"
        f"Skills: {', '.join(ghost.get('ideal_skills', []))}\n"
        f"Experience: {ghost.get('ideal_years_experience', '')} years\n"
        f"Profile: {ghost.get('ideal_background_narrative', '')}\n"
        f"Must-haves: {', '.join(ghost.get('must_have_signals', []))}"
    )

    BATCH_SIZE = 5  
    items = list(resumes.items())
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    print(f"[BATCHED PIPELINE] Phase 2: evaluating {len(items)} candidates in {len(batches)} batches of {BATCH_SIZE}...")

    all_candidates = []
    INTER_BATCH_DELAY = 15  # seconds between batches to respect Groq rate limits

    for idx, batch in enumerate(batches):
        print(f"  → Batch {idx+1}/{len(batches)} ({len(batch)} candidates: {[name for name, _ in batch]})")
        result = _evaluate_candidate_batch(jd_text, ghost_text, batch)
        if len(result) != len(batch):
            print(f"  ⚠ WARNING: Expected {len(batch)} candidates back, got {len(result)}. Groq may have truncated the batch.")
        all_candidates.extend(result)
        if idx < len(batches) - 1:
            print(f"  ⏳ Waiting {INTER_BATCH_DELAY}s to respect Groq rate limits...")
            time.sleep(INTER_BATCH_DELAY)

    print(f"[BATCHED PIPELINE] Done. Got {len(all_candidates)} candidate results.")
    jd_analysis = setup.get("jd_analysis", {})
    if not jd_analysis.get("clarity_score") and not jd_analysis.get("overall_jd_grade"):
        
        if setup.get("clarity_score") or setup.get("overall_jd_grade"):
            jd_analysis = {k: setup[k] for k in [
                "clarity_score", "attractiveness_score", "red_flags",
                "missing_hooks", "rewrite_suggestions", "overall_jd_grade"
            ] if k in setup}
        print(f"[JD DEBUG] jd_analysis extracted: {jd_analysis}")

    jd_parsed = setup.get("jd_parsed", {})
    if not jd_parsed.get("required_skills"):
        if setup.get("required_skills"):
            jd_parsed = {k: setup[k] for k in [
                "required_skills", "required_years_experience", "core_domain",
                "seniority_level", "implicit_culture_signals"
            ] if k in setup}

    return {
        "jd_analysis": jd_analysis,
        "jd_parsed": jd_parsed,
        "ghost_candidate": ghost,
        "candidates": all_candidates
    }

# ==========================================
# MAIN EXPORT
# ==========================================

def run_pipeline(raw_jd: str, synthetic_resumes: dict) -> Tuple[List[Dict], Dict, Dict]:
    """
    Main entry point for the FastAPI backend. 
    Structures data and computes final derived metrics.
    """
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

    # Rank candidates by strongest fit
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    return results, data["ghost_candidate"], data["jd_analysis"]


# ==========================================
# MOCK DATA (Fallback when API key fails)
# ==========================================

def _generate_mock_response(prompt: str, sys_instruction: str = "") -> dict:
    """Provides structured synthetic data to maintain app functionality during API outages."""
    prompt_lower = prompt.lower()
    sys_lower = sys_instruction.lower()

    if "headhunter" in sys_lower or ("ideal" in sys_lower and "offer" in sys_lower):
        return {
            "ideal_title": "Senior Data/ML Engineer",
            "ideal_skills": ["Python", "SQL", "PyTorch", "AWS"],
            "ideal_years_experience": 5,
            "ideal_background_narrative": "5 years building ML pipelines.",
            "must_have_signals": ["Production ML deployment"],
            "nice_to_have_signals": ["OSS contributions"]
        }

    if "talent branding" in sys_lower:
        return {
            "clarity_score": 62,
            "attractiveness_score": 45,
            "red_flags": ["No salary range mentioned"],
            "missing_hooks": ["Remote/hybrid policy"],
            "rewrite_suggestions": ["Add salary range to attract serious candidates"],
            "overall_jd_grade": "C"
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
                    "semantic_reasoning": "Mock reasoning fallback active.",
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
                    "recruiter_brief": "Good candidate, fallback mock generated.",
                    "followup_questions": [],
                    "hiring_recommendation": "Yes",
                    "risk_flags": []
                }
            }
        ]
    }

if __name__ == "__main__":
    # Provides safe execution if script is run directly for debugging.
    print("Testing pipeline with empty candidate DB and dummy JD...")
    res, ghost, jd = run_pipeline("Dummy Job Description requiring Python", {})
    print("Test Complete. Candidates processed:", len(res))