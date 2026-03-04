"""
Generate multi-step reasoning conversations (multi-domain).

Creates conversations where later turns depend on facts from earlier turns,
testing whether the system can chain information across long distances.
Tests: Curator (multi-source retrieval) + Assembler (context integration).
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

# Multi-domain reasoning chains
# Each chain: setup facts → filler → question requiring ALL setup facts
REASONING_CHAINS = {
    "software_engineering": [
        {
            "setup_turns": [
                ("The server runs on Ubuntu 22.04.", "Noted."),
                ("We use nginx as the reverse proxy.", "Got it."),
                ("The app runs on port 5000 internally.", "Internal port: 5000."),
            ],
            "question": "Write the nginx config to proxy requests to our app.",
            "keywords": ["nginx", "5000", "proxy_pass"],
            "test": "Does the nginx config route to port 5000?",
        },
        {
            "setup_turns": [
                ("Our database is PostgreSQL 15.", "Noted."),
                ("The main table is 'events' with columns: id, name, timestamp, user_id.", "Got it."),
                ("We need to partition by month on the timestamp column.", "Understood."),
            ],
            "question": "Write the SQL to create the partitioned events table.",
            "keywords": ["PostgreSQL", "events", "timestamp", "partition", "CREATE"],
            "test": "Does SQL create partitioned events table with correct columns?",
        },
        {
            "setup_turns": [
                ("The project uses FastAPI.", "Noted."),
                ("Auth is handled via JWT tokens with RS256.", "Got it."),
                ("Tokens expire after 15 minutes.", "Token expiry: 15 min."),
                ("Refresh tokens last 7 days.", "Refresh: 7 days."),
            ],
            "question": "Implement the token generation and validation endpoints.",
            "keywords": ["FastAPI", "JWT", "RS256", "15 min", "7 day"],
            "test": "Does implementation use FastAPI, JWT, RS256, 15min expiry?",
        },
    ],
    "medical_consultation": [
        {
            "setup_turns": [
                ("I'm a 55-year-old male.", "Noted."),
                ("I have Type 2 diabetes managed with metformin.", "Got it."),
                ("I'm allergic to sulfa drugs.", "Important — noted the allergy."),
                ("My last A1C was 7.8, slightly above target.", "I see."),
            ],
            "question": "What additional medication might help lower my A1C? Consider my allergies.",
            "keywords": ["metformin", "allerg", "sulfa", "A1C", "7.8"],
            "test": "Does recommendation account for diabetes, sulfa allergy, and current meds?",
        },
        {
            "setup_turns": [
                ("My father had a heart attack at age 52.", "Family history noted."),
                ("My cholesterol is 245 total, LDL is 165.", "Those are elevated."),
                ("I exercise 3 times per week, mostly jogging.", "Good to know."),
            ],
            "question": "Given everything you know about me, what's my cardiovascular risk and what should I do?",
            "keywords": ["heart", "cholesterol", "LDL", "165", "family history", "exercise"],
            "test": "Does assessment incorporate family history, cholesterol, and exercise?",
        },
    ],
    "legal_advisory": [
        {
            "setup_turns": [
                ("I live in California.", "Noted — California law applies."),
                ("My landlord gave me 30 days notice to vacate.", "I see."),
                ("I've been in this apartment for 3 years.", "Long-term tenant."),
                ("My rent has never been late.", "Good record."),
            ],
            "question": "Is my landlord's eviction notice legally valid? What are my options?",
            "keywords": ["California", "30 day", "3 year", "tenant", "eviction"],
            "test": "Does response reference California law, tenancy duration, and notice period?",
        },
    ],
    "data_analysis": [
        {
            "setup_turns": [
                ("We have 25,000 users in the test.", "Good sample size."),
                ("The control conversion rate is 3.2%.", "Baseline noted."),
                ("The variant shows 3.8% conversion.", "That's an 18.75% relative lift."),
                ("The test has been running for 14 days.", "Two weeks of data."),
            ],
            "question": "Should we call this test and ship the variant? What's your analysis?",
            "keywords": ["25,000", "3.2%", "3.8%", "14 day", "significant", "sample"],
            "test": "Does analysis reference sample size, both rates, and duration?",
        },
        {
            "setup_turns": [
                ("Revenue last quarter was $2.4M.", "Noted."),
                ("We spent $180K on paid ads.", "Ad budget noted."),
                ("Organic traffic grew 23% quarter-over-quarter.", "Strong organic growth."),
                ("Customer acquisition cost is $45 for paid, $12 for organic.", "Big difference."),
            ],
            "question": "Should we shift more budget from paid to organic? Make the case with our numbers.",
            "keywords": ["$2.4M", "$180K", "23%", "$45", "$12", "organic", "paid"],
            "test": "Does recommendation use all the specific numbers from our discussion?",
        },
    ],
    "project_management": [
        {
            "setup_turns": [
                ("The team is 5 developers and 1 QA.", "Small team."),
                ("Sprint velocity has been 32 story points on average.", "Noted."),
                ("The remaining backlog is 180 story points.", "That's significant."),
                ("The launch deadline is 8 weeks away.", "Tight timeline."),
            ],
            "question": "Can we hit the deadline? If not, what do we cut or change?",
            "keywords": ["32", "180", "8 week", "velocity", "backlog"],
            "test": "Does assessment use velocity, backlog, and deadline to do the math?",
        },
    ],
    "cooking_recipes": [
        {
            "setup_turns": [
                ("I'm cooking for 8 people at a dinner party.", "Noted — big group."),
                ("Two guests are gluten-free.", "I'll keep that in mind."),
                ("One guest is vegetarian.", "No problem."),
                ("My oven maxes out at 450°F.", "Good to know."),
            ],
            "question": "Plan a three-course dinner menu that works for everyone.",
            "keywords": ["8 people", "gluten-free", "vegetarian", "450", "menu"],
            "test": "Does menu accommodate dietary restrictions and guest count?",
        },
    ],
    "personal_finance": [
        {
            "setup_turns": [
                ("I make $85,000 per year before taxes.", "Good income."),
                ("My monthly expenses are about $4,200.", "Noted."),
                ("I have $15,000 in credit card debt at 22% APR.", "That's costly."),
                ("My employer matches 401k up to 4%.", "Free money."),
            ],
            "question": "Create a monthly budget plan that tackles my debt while not missing the 401k match.",
            "keywords": ["$85,000", "$4,200", "$15,000", "22%", "4%", "401k"],
            "test": "Does plan incorporate salary, expenses, debt rate, and 401k match?",
        },
    ],
    "creative_writing": [
        {
            "setup_turns": [
                ("The story is set in 1920s Paris.", "Atmospheric setting."),
                ("The protagonist is a jazz pianist named Louis.", "Great character."),
                ("The mystery involves a stolen painting from the Louvre.", "Classic heist."),
                ("The femme fatale is named Colette, she runs a speakeasy.", "Intriguing."),
            ],
            "question": "Write the scene where Louis confronts Colette about the painting at her speakeasy.",
            "keywords": ["Louis", "Colette", "painting", "speakeasy", "1920s"],
            "test": "Does scene include all established characters, setting, and plot elements?",
        },
    ],
    "customer_support": [
        {
            "setup_turns": [
                ("My account number is AC-55921.", "Let me pull that up."),
                ("I've been experiencing speed issues for 2 weeks.", "That's a long time."),
                ("A technician came last Tuesday but it didn't help.", "I see."),
                ("I'm on the 200Mbps plan but getting only 15Mbps.", "Huge gap."),
            ],
            "question": "Given all the context, what's the next step? I need this resolved today.",
            "keywords": ["AC-55921", "2 week", "technician", "200Mbps", "15Mbps"],
            "test": "Does response reference account, history, speed gap, and previous visit?",
        },
    ],
    "academic_tutoring": [
        {
            "setup_turns": [
                ("We learned that derivatives measure instantaneous rate of change.", "Correct."),
                ("The power rule says d/dx[x^n] = nx^(n-1).", "Yes, fundamental rule."),
                ("The chain rule is for composite functions: d/dx[f(g(x))] = f'(g(x))·g'(x).", "Exactly."),
            ],
            "question": "Using what we've covered, find the derivative of (3x² + 1)⁵.",
            "keywords": ["chain rule", "power rule", "5", "3x", "derivative", "30x"],
            "test": "Does solution correctly apply both chain rule and power rule?",
        },
    ],
}

# Same domain fillers as NIAH (imported pattern)
DOMAIN_FILLERS = {
    "software_engineering": [
        ("What's the best way to structure a Flask app?", "Use blueprints to organize routes..."),
        ("How do database migrations work?", "Tools like Alembic track schema changes..."),
        ("Should I use REST or GraphQL?", "REST for simplicity, GraphQL for complex queries..."),
        ("What testing strategy do you recommend?", "Unit tests for logic, integration for APIs..."),
        ("How should I handle environment config?", "Use env vars, never hardcode secrets..."),
    ],
    "medical_consultation": [
        ("How much water should I drink daily?", "About 8 glasses, more if exercising..."),
        ("Is daily exercise safe?", "Moderate daily exercise is fine..."),
        ("How does stress affect health?", "Stress hormones can affect many body systems..."),
        ("Should I take supplements?", "If your diet is balanced you may not need them..."),
    ],
    "legal_advisory": [
        ("How do I write a demand letter?", "State facts, cite law, specify what you want..."),
        ("What's small claims court like?", "File a claim, serve defendant, attend hearing..."),
        ("Do I need a business license?", "Most businesses do; check local requirements..."),
    ],
    "data_analysis": [
        ("What visualization works best for trends?", "Line charts for time series..."),
        ("How do I handle missing data?", "Depends on pattern — imputation or flagging..."),
        ("What's overfitting?", "Model learns noise; use cross-validation to prevent..."),
    ],
    "project_management": [
        ("How long should sprints be?", "2 weeks is standard..."),
        ("How do I handle scope creep?", "Document, estimate impact, get sign-off..."),
        ("What's velocity?", "Story points completed per sprint..."),
    ],
    "cooking_recipes": [
        ("What's the Maillard reaction?", "Chemical reaction between amino acids and sugars..."),
        ("How do I deglaze a pan?", "Add liquid to hot pan to dissolve browned bits..."),
        ("What's the best way to store herbs?", "Damp paper towel in bag in fridge..."),
    ],
    "personal_finance": [
        ("What's compound interest?", "Interest on interest over time..."),
        ("How do ETFs work?", "Trade like stocks, track an index..."),
        ("What's the 50/30/20 rule?", "50% needs, 30% wants, 20% savings..."),
    ],
    "creative_writing": [
        ("How do I avoid purple prose?", "Use sensory details purposefully..."),
        ("What makes good dialogue?", "Subtext and distinct character voices..."),
        ("How do I create tension?", "Withhold information, raise stakes..."),
    ],
    "customer_support": [
        ("Can I check usage online?", "Yes, your account dashboard has real-time stats..."),
        ("What's included in my plan?", "Check your plan details in the app..."),
    ],
    "academic_tutoring": [
        ("How do I study for proofs?", "Practice writing them; understand logic structure..."),
        ("What calculator can I use?", "Check your syllabus; many exams are no-calc..."),
        ("How important are exercises?", "Very — they build problem-solving muscle..."),
    ],
}


def generate_multistep_conversation(
    domain: str = None,
    chain_idx: int = 0,
    filler_between: int = 10,
    total_turns: int = 50,
    seed: int = None,
) -> dict:
    """Generate a multi-step reasoning conversation."""
    if seed is not None:
        random.seed(seed)

    if domain is None:
        domain = random.choice(list(REASONING_CHAINS.keys()))

    chains = REASONING_CHAINS[domain]
    chain = chains[chain_idx % len(chains)]

    turns = []
    checkpoints = []
    turn_num = 0

    # Get domain fillers
    fillers = list(DOMAIN_FILLERS.get(domain, [("Question?", "Answer.")]))
    fillers = fillers * ((total_turns // len(fillers)) + 2)
    random.shuffle(fillers)
    filler_idx = 0

    # Setup turns (plant facts early)
    for user_msg, asst_msg in chain["setup_turns"]:
        turn_num += 1
        turns.append({"turn": turn_num, "role": "user", "text": user_msg, "type": "setup"})
        turns.append({"turn": turn_num, "role": "assistant", "text": asst_msg, "type": "setup"})

    # Filler turns (push facts further back in context)
    for _ in range(filler_between):
        turn_num += 1
        if filler_idx < len(fillers):
            u, a = fillers[filler_idx]
            filler_idx += 1
        else:
            u, a = "Can you elaborate?", "Sure, let me expand on that..."
        turns.append({"turn": turn_num, "role": "user", "text": u, "type": "filler"})
        turns.append({"turn": turn_num, "role": "assistant", "text": a, "type": "filler"})

    # The multi-step question
    turn_num += 1
    turns.append({"turn": turn_num, "role": "user", "text": chain["question"], "type": "test"})
    turns.append({"turn": turn_num, "role": "assistant",
                  "text": "[RESPONSE_PLACEHOLDER]", "type": "test_response"})

    checkpoints.append({
        "turn": turn_num,
        "test": chain["test"],
        "answer": True,
        "keywords": chain["keywords"],
    })

    # Pad to total_turns
    while turn_num < total_turns:
        turn_num += 1
        if filler_idx < len(fillers):
            u, a = fillers[filler_idx]
            filler_idx += 1
        else:
            u, a = "Anything else on that topic?", "Let me think about other aspects..."
        turns.append({"turn": turn_num, "role": "user", "text": u, "type": "filler"})
        turns.append({"turn": turn_num, "role": "assistant", "text": a, "type": "filler"})

    return {
        "conversation_id": f"multistep_{domain}_{seed or random.randint(0, 99999):05d}",
        "domain": domain,
        "chain_type": chain["test"],
        "num_turns": turn_num,
        "filler_between": filler_between,
        "checkpoints": checkpoints,
        "turns": turns,
    }


def generate_dataset(num_conversations: int = 40, output_path: Path = None):
    """Generate full multi-step dataset across all domains."""
    output_path = output_path or Path(__file__).parent.parent / "datasets" / "multistep_reasoning"
    synthetic_dir = output_path / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    domains = list(REASONING_CHAINS.keys())
    domain_convos: Dict[str, list] = {d: [] for d in domains}

    for i in range(num_conversations):
        domain = domains[i % len(domains)]
        gap = random.choice([5, 10, 15, 20, 30])

        conv = generate_multistep_conversation(
            domain=domain,
            chain_idx=i // len(domains),
            filler_between=gap,
            total_turns=gap + 10,
            seed=i,
        )
        domain_convos[domain].append(conv)

    # Write per-domain files
    all_convos = []
    for domain, convos in domain_convos.items():
        if convos:
            out_file = synthetic_dir / f"{domain}.json"
            out_file.write_text(json.dumps(convos, indent=2))
            all_convos.extend(convos)
            print(f"  {domain}: {len(convos)} conversations → {out_file}")

    total_checkpoints = sum(len(c["checkpoints"]) for c in all_convos)
    print(f"\nGenerated {len(all_convos)} multi-step conversations")
    print(f"  Total checkpoints: {total_checkpoints}")
    print(f"  Domains: {len(domains)}")

    meta = output_path / "dataset_meta.json"
    meta.write_text(json.dumps({
        "generator": "generate_multistep.py",
        "num_conversations": len(all_convos),
        "total_checkpoints": total_checkpoints,
        "domains": {d: len(c) for d, c in domain_convos.items()},
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=40)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, output)
