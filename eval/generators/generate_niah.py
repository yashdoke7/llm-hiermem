"""
Generate Needle-in-a-Haystack conversations (multi-domain).

Plants specific facts/details at known turns, then tests recall at
varying distances (5, 10, 15, 20, 30+ turns later).
Tests: Archive retrieval (L1/L2/L3) and Retrieval Router accuracy.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

# Multi-domain needles: (fact_statement, recall_question, expected_keyword, domain)
NEEDLES = {
    "software_engineering": [
        ("The secret password for the staging environment is 'aurora-7742'.",
         "What's the password for the staging environment?",
         "aurora-7742"),
        ("Use port 8443 for the staging server, not the default 443.",
         "Which port did we decide for the staging server?",
         "8443"),
        ("The API rate limit is 100 requests per minute per user.",
         "What rate limit did we set for the API?",
         "100 requests"),
        ("My team uses branch naming: feature/TICKET-description for all feature work.",
         "What branch naming convention does your team use?",
         "feature/TICKET"),
        ("The database backup runs at 3:00 AM UTC daily.",
         "When is the database backup scheduled?",
         "3:00 AM UTC"),
        ("We chose PostgreSQL 16 as the primary database for this project.",
         "Which database did we decide on for the project?",
         "PostgreSQL"),
        ("The maximum file upload size is 25MB for the user-facing API.",
         "What's the file upload size limit we agreed on?",
         "25MB"),
    ],
    "medical_consultation": [
        ("I'm allergic to penicillin — I had a severe reaction as a child.",
         "Do I have any drug allergies you should know about?",
         "penicillin"),
        ("My blood type is B negative, which is fairly rare.",
         "What blood type did I mention?",
         "B negative"),
        ("I take metformin 500mg twice daily for my Type 2 diabetes.",
         "What medication am I currently taking for diabetes?",
         "metformin"),
        ("My last blood pressure reading was 145/92, which was elevated.",
         "What was my recent blood pressure reading?",
         "145/92"),
        ("I had my appendix removed when I was 12 years old.",
         "Have I had any previous surgeries?",
         "appendix"),
        ("My mother was diagnosed with breast cancer at age 55.",
         "What's the family history I mentioned regarding cancer?",
         "breast cancer"),
    ],
    "legal_advisory": [
        ("The lease was signed on June 15, 2024, for a 2-year term.",
         "When did I sign my current lease?",
         "June 15"),
        ("My security deposit was $3,200 when I moved in.",
         "How much was my security deposit?",
         "$3,200"),
        ("The landlord's name is Margaret Chen and she lives in unit 1A.",
         "What's my landlord's name?",
         "Margaret Chen"),
        ("The company was incorporated in Delaware on March 3, 2023.",
         "Where and when was the company incorporated?",
         "Delaware"),
        ("The non-compete clause in my contract lasts for 18 months.",
         "How long is the non-compete period in my contract?",
         "18 months"),
    ],
    "creative_writing": [
        ("The protagonist Maya has a distinctive scar above her left eyebrow from a childhood accident.",
         "What physical feature distinguishes Maya?",
         "scar"),
        ("The story is set in the fictional coastal town of Harrowcliff, Oregon.",
         "What's the name of the town where the story takes place?",
         "Harrowcliff"),
        ("The villain's name is Dr. Edmund Voss, a former marine biologist.",
         "What was the villain's name and former profession?",
         "Voss"),
        ("The hidden chamber behind the lighthouse was built during Prohibition.",
         "When was the hidden chamber built?",
         "Prohibition"),
    ],
    "data_analysis": [
        ("The dataset contains 47,832 e-commerce transactions from Q1-Q3 2025.",
         "How many transactions are in our dataset?",
         "47,832"),
        ("The average customer lifetime value came out to $284.50.",
         "What was the average customer lifetime value we calculated?",
         "$284"),
        ("Only 7.3% of customers in the dataset churned within 6 months.",
         "What was the churn rate in our analysis?",
         "7.3%"),
        ("The A/B test ran for exactly 21 days with 15,000 users per variant.",
         "How long did the A/B test run and what was the sample size?",
         "21 days"),
    ],
    "project_management": [
        ("The project deadline is November 15, and we cannot extend it.",
         "When is the hard deadline for the project?",
         "November 15"),
        ("The budget cap is $180,000 including all contractor costs.",
         "What's our total budget for this project?",
         "$180,000"),
        ("Sarah from the backend team is on parental leave until October.",
         "When does Sarah return from leave?",
         "October"),
        ("The client's primary contact is James Liu, VP of Engineering at Nextera.",
         "Who is our primary contact at the client?",
         "James Liu"),
    ],
    "personal_finance": [
        ("My total student loan balance is $34,500 at 5.5% interest.",
         "What's my student loan balance and rate?",
         "$34,500"),
        ("My employer matches 401k contributions up to 6% of salary.",
         "What's my employer's 401k match?",
         "6%"),
        ("I contribute $500/month to a Roth IRA at Fidelity.",
         "How much do I put into my Roth IRA each month?",
         "$500"),
        ("My emergency fund has 3 months of expenses saved — about $12,000.",
         "How much is in my emergency fund?",
         "$12,000"),
    ],
    "cooking_recipes": [
        ("I'm severely allergic to tree nuts, especially walnuts and cashews.",
         "What food allergies did I mention?",
         "tree nuts"),
        ("My oven runs about 15 degrees hot, so I always reduce temp by 15.",
         "What quirk did I mention about my oven?",
         "15 degrees"),
        ("I use King Arthur all-purpose flour for everything.",
         "What brand of flour do I prefer?",
         "King Arthur"),
        ("My sourdough starter is 6 months old and I feed it every Friday.",
         "How old is my sourdough starter?",
         "6 months"),
    ],
    "customer_support": [
        ("My account number is AC-88431 and I've been a customer since 2019.",
         "What's my account number?",
         "AC-88431"),
        ("The last technician visit was on February 12 and didn't fix the issue.",
         "When was the last technician visit?",
         "February 12"),
        ("I'm on the Premium Plus plan at $89.99/month.",
         "What plan am I on and what does it cost?",
         "$89.99"),
    ],
    "academic_tutoring": [
        ("I got 72% on my last calculus midterm — derivatives were my weak point.",
         "What did I score on my last midterm?",
         "72%"),
        ("My final exam is on December 10 and covers chapters 1-8.",
         "When is my final exam?",
         "December 10"),
        ("My professor is Dr. Martinez and she emphasizes proofs over computation.",
         "Who is my professor and what does she emphasize?",
         "Martinez"),
    ],
}

# Domain-specific filler conversations (coherent within domain)
DOMAIN_FILLERS = {
    "software_engineering": [
        ("What's the best way to structure a Flask app?", "Use blueprints to organize routes by feature..."),
        ("How do database migrations work?", "Tools like Alembic track schema changes as versioned scripts..."),
        ("Should I use REST or GraphQL for this API?", "REST is simpler for CRUD; GraphQL shines with complex queries..."),
        ("How do I implement pagination properly?", "Cursor-based pagination is more reliable than offset..."),
        ("What's the best logging strategy?", "Use structured logging with correlation IDs..."),
        ("How should I handle environment config?", "Use env vars or a config manager, never hardcode..."),
        ("What testing strategy do you recommend?", "Unit tests for logic, integration tests for API endpoints..."),
        ("Should I use an ORM or raw SQL?", "ORMs speed development; raw SQL gives performance control..."),
        ("How do I implement rate limiting?", "Token bucket or sliding window algorithms work well..."),
        ("What about API documentation?", "OpenAPI/Swagger with auto-generation from code annotations..."),
    ],
    "medical_consultation": [
        ("How much water should I drink daily?", "About 8 glasses, more if you exercise regularly..."),
        ("Is it safe to exercise every day?", "Moderate exercise daily is fine; vary intensity..."),
        ("How does stress affect blood sugar?", "Stress hormones like cortisol can raise blood sugar..."),
        ("What's the deal with intermittent fasting?", "Some studies show metabolic benefits, but check with your doctor..."),
        ("How much sleep do I actually need?", "Most adults need 7-9 hours for optimal health..."),
        ("Should I take a daily multivitamin?", "If your diet is balanced you may not need one..."),
        ("How often should I get blood work done?", "Annual panels are standard; more often with chronic conditions..."),
        ("What are signs of pre-diabetes I should watch for?", "Increased thirst, frequent urination, fatigue..."),
    ],
    "legal_advisory": [
        ("How do I write a proper demand letter?", "State facts, cite relevant law, specify what you want and by when..."),
        ("What's the small claims court process?", "File a claim, serve the defendant, attend a hearing..."),
        ("How do I register a trademark?", "File with the USPTO; consider state registration too..."),
        ("Do I need a business license?", "Most businesses do; requirements vary by city and type..."),
        ("What's an operating agreement?", "It defines member roles, profit sharing, and governance for an LLC..."),
        ("How do non-disclosure agreements work?", "NDAs protect confidential information shared between parties..."),
    ],
    "creative_writing": [
        ("How do I create tension in dialogue?", "Use subtext — what characters don't say matters more..."),
        ("What makes a good plot twist?", "It should be surprising yet inevitable in hindsight..."),
        ("How do I avoid purple prose?", "Use sensory details purposefully, not decoratively..."),
        ("Should I outline or write by discovery?", "Both work; outlines help with structure, discovery adds surprise..."),
        ("How do I write authentic-sounding dialogue?", "Read it aloud; each character should sound distinct..."),
        ("What's the three-act structure?", "Setup, confrontation, resolution — each with clear turning points..."),
    ],
    "data_analysis": [
        ("What visualization works best for trends?", "Line charts for time series, scatter plots for correlation..."),
        ("How do I handle missing data?", "Depends on pattern — imputation, deletion, or flagging..."),
        ("What's the difference between median and mean?", "Mean is sensitive to outliers; median gives the middle value..."),
        ("When should I use a log scale?", "When data spans several orders of magnitude..."),
        ("How do I detect multicollinearity?", "Check VIF values; above 10 suggests high correlation between features..."),
        ("What's overfitting and how do I prevent it?", "Model learns noise; use cross-validation and regularization..."),
    ],
    "project_management": [
        ("How long should sprints be?", "2 weeks is standard; 1 week for fast-moving early-stage products..."),
        ("What's a good standup format?", "What I did, what I'll do, blockers — keep it under 15 minutes..."),
        ("How do I handle scope creep?", "Document the change request, estimate impact, get stakeholder sign-off..."),
        ("What's velocity and how do I measure it?", "Story points completed per sprint; track over 3-4 sprints for a baseline..."),
        ("When should I say no to a feature request?", "When it doesn't align with current sprint goals or product vision..."),
    ],
    "personal_finance": [
        ("What's compound interest?", "Interest on interest — your returns generate their own returns over time..."),
        ("How do ETFs differ from mutual funds?", "ETFs trade like stocks; mutual funds price once daily..."),
        ("Should I use a budget app?", "Yes, tools like YNAB or Mint help track spending habits..."),
        ("What's the 50/30/20 rule?", "50% needs, 30% wants, 20% savings — a simple budgeting framework..."),
        ("How does inflation affect savings?", "Money loses purchasing power over time; savings need to beat inflation..."),
    ],
    "cooking_recipes": [
        ("What's the difference between baking soda and baking powder?", "Baking soda needs acid; baking powder is self-contained..."),
        ("How do I know when oil is hot enough?", "A small piece of bread should sizzle immediately when dropped in..."),
        ("What's the best way to store herbs?", "Wrap in damp paper towel in a plastic bag in the fridge..."),
        ("How do I properly season a cast iron pan?", "Thin layer of oil, bake upside down at 450F for an hour..."),
        ("Why does my bread come out dense?", "Could be too much flour, old yeast, or insufficient kneading..."),
    ],
    "customer_support": [
        ("Can I check my usage online?", "Yes, log into your account dashboard for real-time usage stats..."),
        ("What's included in my plan?", "Your plan includes unlimited talk/text and 50GB of data..."),
        ("How do I update my payment method?", "Go to Settings > Billing > Payment Method in the app..."),
        ("Is there a way to reduce my bill?", "We can review your usage patterns and suggest a better-fit plan..."),
        ("When is the next billing cycle?", "Billing cycles on the 1st of each month; your next charge is then..."),
    ],
    "academic_tutoring": [
        ("What's the intuition behind integration?", "It's the area under a curve — adding up infinitesimally thin slices..."),
        ("How do I study for proof-based exams?", "Practice writing proofs; understand logic structure, not just formulas..."),
        ("What's the difference between theorem and lemma?", "A lemma is a helper result; a theorem is the main result..."),
        ("How important are textbook exercises?", "Very — they build the problem-solving muscle you need for exams..."),
        ("What calculator can I use on exams?", "Check your syllabus; most calculus exams are no-calculator..."),
    ],
}


def generate_niah_conversation(
    domain: str = None,
    needle_turn: int = 3,
    recall_distances: List[int] = None,
    total_turns: int = 50,
    seed: int = None,
) -> dict:
    """Generate a NIAH conversation with domain-coherent filler."""
    if seed is not None:
        random.seed(seed)

    if domain is None:
        domain = random.choice(list(NEEDLES.keys()))

    # Pick a needle for this domain
    needle_text, recall_question, expected_answer = random.choice(NEEDLES[domain])

    if recall_distances is None:
        recall_distances = [5, 10, 15, 20, 30, 40]

    turns = []
    checkpoints = []

    # Get domain fillers
    fillers = list(DOMAIN_FILLERS.get(domain, [("General question?", "General answer.")]))
    fillers = fillers * ((total_turns // len(fillers)) + 2)
    random.shuffle(fillers)
    filler_idx = 0

    recall_turns = set(needle_turn + d for d in recall_distances if needle_turn + d <= total_turns)

    for t in range(1, total_turns + 1):
        if t == needle_turn:
            # Plant the needle naturally
            turns.append({"turn": t, "role": "user",
                         "text": f"Oh by the way — {needle_text}", "type": "needle"})
            turns.append({"turn": t, "role": "assistant",
                         "text": "Got it, I've noted that down.", "type": "ack"})
        elif t in recall_turns:
            turns.append({"turn": t, "role": "user",
                         "text": recall_question, "type": "recall"})
            turns.append({"turn": t, "role": "assistant",
                         "text": "[RESPONSE_PLACEHOLDER]", "type": "recall_response"})
            checkpoints.append({
                "turn": t,
                "distance_from_needle": t - needle_turn,
                "test": f"Does response contain '{expected_answer}'?",
                "answer": True,
                "keywords": [expected_answer],
            })
        else:
            if filler_idx < len(fillers):
                u, a = fillers[filler_idx]
                filler_idx += 1
            else:
                u, a = "Can you elaborate on that?", "Of course, let me expand on that..."
            turns.append({"turn": t, "role": "user", "text": u, "type": "filler"})
            turns.append({"turn": t, "role": "assistant", "text": a, "type": "filler"})

    return {
        "conversation_id": f"niah_{domain}_{seed or random.randint(0, 99999):05d}",
        "domain": domain,
        "needle_turn": needle_turn,
        "needle_text": needle_text,
        "recall_question": recall_question,
        "expected_answer": expected_answer,
        "num_turns": total_turns,
        "checkpoints": checkpoints,
        "turns": turns,
    }


def generate_dataset(num_conversations: int = 30, output_path: Path = None):
    """Generate the full NIAH dataset across all domains."""
    output_path = output_path or Path(__file__).parent.parent / "datasets" / "needle_in_haystack"
    synthetic_dir = output_path / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    domains = list(NEEDLES.keys())
    all_convos = []
    domain_convos: Dict[str, list] = {d: [] for d in domains}

    for i in range(num_conversations):
        domain = domains[i % len(domains)]
        needle_pos = random.randint(2, 8)
        total = random.choice([30, 40, 50])
        distances = [5, 10, 15, 20, 30]

        conv = generate_niah_conversation(
            domain=domain,
            needle_turn=needle_pos,
            recall_distances=distances,
            total_turns=total,
            seed=i,
        )
        domain_convos[domain].append(conv)
        all_convos.append(conv)

    # Write per-domain files
    for domain, convos in domain_convos.items():
        if convos:
            out_file = synthetic_dir / f"{domain}.json"
            out_file.write_text(json.dumps(convos, indent=2))
            print(f"  {domain}: {len(convos)} conversations → {out_file}")

    total_checkpoints = sum(len(c["checkpoints"]) for c in all_convos)
    print(f"\nGenerated {len(all_convos)} NIAH conversations")
    print(f"  Total checkpoints: {total_checkpoints}")
    print(f"  Domains: {len(domains)}")

    # Metadata
    meta = output_path / "dataset_meta.json"
    meta.write_text(json.dumps({
        "generator": "generate_niah.py",
        "num_conversations": len(all_convos),
        "total_checkpoints": total_checkpoints,
        "domains": {d: len(c) for d, c in domain_convos.items()},
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, output)
