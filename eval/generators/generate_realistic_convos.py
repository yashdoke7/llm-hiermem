"""
Generate realistic multi-domain constraint-tracking conversations.

Improvements over original generator:
  - 10 domains (not just coding)
  - Coherent conversation flow (turns build on each other)
  - No "remember our rules" hints in checkpoints
  - Natural constraint setup (stated as preferences, not numbered lists)
  - Domain-appropriate keywords for accurate automated evaluation
  - Varied checkpoint distances (5, 10, 20, 30, 40+ turns)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# DOMAIN DEFINITIONS
# Each domain has constraints (with eval keywords),
# conversation flows, and checkpoint prompts.
# ═══════════════════════════════════════════════════════════════

DOMAINS = {
    # ─── 1. SOFTWARE ENGINEERING ─────────────────────────────
    "software_engineering": {
        "name": "Software Engineering",
        "constraints": [
            {
                "text": "Always use type hints in all Python code examples",
                "keywords": [": str", ": int", ": float", ": bool", "-> ",
                             ": List", ": Dict", ": Optional", ": Any",
                             ": list", ": dict", ": tuple"],
                "checkpoint_prompts": [
                    "Write a function to validate email addresses",
                    "Create a class that manages a shopping cart",
                    "Write a function to parse CSV data and return structured records",
                    "Implement a retry decorator with configurable max attempts",
                    "Write a function to merge two sorted lists efficiently",
                ],
            },
            {
                "text": "Always include error handling in code examples",
                "keywords": ["try", "except", "raise", "Error", "Exception",
                             "catch", "error handling"],
                "checkpoint_prompts": [
                    "Write a function to read and parse a JSON config file",
                    "Create a database connection wrapper",
                    "Write an HTTP client that fetches data from an API",
                    "Implement a file upload handler",
                    "Write a function to process payment transactions",
                ],
            },
            {
                "text": "Always suggest writing tests when implementing new functions",
                "keywords": ["test", "unittest", "pytest", "assert", "test_",
                             "mock", "testing", "test case"],
                "checkpoint_prompts": [
                    "Implement a password strength validator",
                    "Write a function to calculate shipping costs based on weight and distance",
                    "Create a rate limiter class",
                    "Implement a simple in-memory cache with TTL",
                    "Write a URL shortener function",
                ],
            },
            {
                "text": "Always explain the time complexity of algorithms you suggest",
                "keywords": ["O(", "complexity", "time complexity",
                             "space complexity", "Big O", "linear",
                             "quadratic", "logarithmic", "constant time"],
                "checkpoint_prompts": [
                    "What's the best way to find duplicates in a list?",
                    "How should I implement a search feature for user records?",
                    "What algorithm would you recommend for sorting a nearly-sorted list?",
                    "How do I efficiently find the k-th largest element?",
                    "What's the best approach to detect cycles in a linked list?",
                ],
            },
        ],
        "natural_setups": [
            "I'm working on a Python project. A few things matter to me: {c1}. Also, {c2}. Keep these in mind throughout.",
            "Before we start — for my codebase, I need you to follow these practices: {c1}. And {c2}.",
            "Quick context on my coding standards: {c1}. Additionally, {c2}. These are non-negotiable for my project.",
        ],
        "flows": [
            [
                "I'm building a REST API for a bookstore. What framework should I use in Python?",
                "Good choice. How should I structure the project directories?",
                "Let's design the database models first. What tables do we need?",
                "How should I handle the relationship between books and authors?",
                "What about ISBN validation? How should I store that?",
                "CHECKPOINT",
                "Now let's add pagination to the book listing endpoint.",
                "How should I handle filtering by genre and price range?",
                "What caching strategy would work for the popular books list?",
                "CHECKPOINT",
                "How should I handle image uploads for book covers?",
                "What about rate limiting the API?",
                "CHECKPOINT",
                "Let's add a search feature. Full-text or keyword-based?",
                "How should I handle user authentication for the admin panel?",
                "What's the best way to handle bulk book imports from CSV?",
                "CHECKPOINT",
                "How do I add webhooks for inventory changes?",
                "What monitoring should I set up?",
                "How do I handle database migrations in production?",
                "CHECKPOINT",
            ],
            [
                "I need to build a data pipeline that processes log files. Where do I start?",
                "The logs are in different formats — Apache, nginx, and custom JSON. How to handle that?",
                "How should I handle malformed log entries?",
                "CHECKPOINT",
                "What about deduplication? Some logs might be repeated.",
                "I need to aggregate the data hourly. What approach?",
                "How should I store the processed data — Parquet or CSV?",
                "CHECKPOINT",
                "Now I need to add real-time processing alongside batch. How?",
                "What about backpressure when the pipeline falls behind?",
                "How do I handle schema evolution as log formats change?",
                "CHECKPOINT",
                "I need to add alerting when error rates spike. How?",
                "What's the best way to test this pipeline end-to-end?",
                "CHECKPOINT",
                "How do I deploy this to production?",
                "What observability should I add to the pipeline itself?",
                "How do I handle data retention and cleanup?",
                "CHECKPOINT",
            ],
            [
                "I want to build a CLI tool for managing Docker containers.",
                "What library should I use for CLI argument parsing — argparse or click?",
                "How should I handle the config file format — YAML or TOML?",
                "CHECKPOINT",
                "I need to support starting, stopping, and restarting containers. How to structure that?",
                "How do I interact with the Docker daemon from Python?",
                "What about showing container logs in real-time?",
                "CHECKPOINT",
                "How should I handle container health checks?",
                "What about supporting multiple environments (dev, staging, prod)?",
                "CHECKPOINT",
                "I need container resource monitoring — CPU and memory. How?",
                "How do I handle network setup between containers?",
                "What about volume management for persistent data?",
                "How should I handle secrets and environment variables?",
                "CHECKPOINT",
                "How do I package and distribute this CLI tool?",
                "What about shell auto-completion for the CLI commands?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 2. MEDICAL CONSULTATION ─────────────────────────────
    "medical_consultation": {
        "name": "Medical Consultation",
        "constraints": [
            {
                "text": "Always mention potential side effects when discussing any medication",
                "keywords": ["side effect", "adverse", "may cause", "risk of",
                             "common effects", "nausea", "dizziness",
                             "drowsiness", "caution", "reaction"],
                "checkpoint_prompts": [
                    "What medication is commonly used for tension headaches?",
                    "Tell me about metformin for Type 2 diabetes management",
                    "What are the treatment options for high blood pressure?",
                    "What medication helps with seasonal allergies?",
                    "What's prescribed for acid reflux?",
                ],
            },
            {
                "text": "Always recommend consulting a healthcare professional before starting or changing treatment",
                "keywords": ["consult", "doctor", "physician",
                             "healthcare professional", "medical professional",
                             "speak with", "talk to your doctor",
                             "professional advice", "qualified"],
                "checkpoint_prompts": [
                    "I've been having persistent lower back pain. What should I do?",
                    "My blood sugar readings have been higher than usual lately",
                    "I'm thinking about starting a supplement regimen for joint health",
                    "Should I switch from ibuprofen to acetaminophen for my pain?",
                    "I want to try intermittent fasting for weight loss",
                ],
            },
            {
                "text": "Always ask about existing allergies before suggesting any medication",
                "keywords": ["allerg", "allergic", "sensitivity",
                             "intolerance", "reaction to", "any allergies",
                             "known allergies"],
                "checkpoint_prompts": [
                    "What's good for a sore throat? I need something fast-acting",
                    "I have a skin rash that won't go away. What should I try?",
                    "What can I take for this bad migraine?",
                    "My child has a fever. What medicine should I give?",
                    "What antibiotic would you suggest for a UTI?",
                ],
            },
        ],
        "natural_setups": [
            "I want to use you as a health reference. Important: {c1}. Also, {c2}. I take this seriously.",
            "I'll be asking health questions. Please always {c1}, and also {c2}.",
            "Before we begin — when discussing health topics, I expect you to {c1}. And always {c2}.",
        ],
        "flows": [
            [
                "I was recently diagnosed with Type 2 diabetes. Can you help me understand it?",
                "What lifestyle changes should I make first?",
                "How does diet affect blood sugar levels?",
                "What foods should I prioritize and which should I limit?",
                "CHECKPOINT",
                "How often should I check my blood sugar?",
                "What are the target blood sugar ranges I should aim for?",
                "Is exercise safe for me? What types are best?",
                "CHECKPOINT",
                "I've heard about diabetic neuropathy. How do I prevent it?",
                "What about eye problems associated with diabetes?",
                "CHECKPOINT",
                "How does stress affect my blood sugar?",
                "What should I do if my blood sugar drops too low?",
                "Are there any natural supplements that help with blood sugar control?",
                "CHECKPOINT",
                "What about drinking alcohol with diabetes?",
                "How do I manage diabetes while traveling?",
                "What regular checkups do I need going forward?",
                "CHECKPOINT",
            ],
            [
                "I want to improve my overall health at age 45. Where do I start?",
                "What health screenings should I be getting at my age?",
                "My cholesterol was slightly elevated last check. What does that mean?",
                "CHECKPOINT",
                "How much exercise should I be getting weekly?",
                "I have trouble sleeping lately. Any approaches that might help?",
                "CHECKPOINT",
                "What vitamins should someone my age consider taking?",
                "I've been feeling more fatigued than usual. What could cause that?",
                "CHECKPOINT",
                "My family has a history of heart disease. What preventive steps?",
                "How important is stress management for cardiovascular health?",
                "What dietary changes reduce heart disease risk?",
                "CHECKPOINT",
                "I've been reading about anti-inflammatory diets. Are they effective?",
                "How do I know if I need a vitamin D supplement?",
                "What about probiotics — are they worth it?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 3. LEGAL ADVISORY ───────────────────────────────────
    "legal_advisory": {
        "name": "Legal Advisory",
        "constraints": [
            {
                "text": "Always include a disclaimer that this is general information, not legal advice",
                "keywords": ["not legal advice", "disclaimer",
                             "not a substitute for legal",
                             "consult an attorney", "consult a lawyer",
                             "general information", "qualified legal",
                             "seek legal counsel"],
                "checkpoint_prompts": [
                    "What are my rights if my landlord wants to evict me?",
                    "Can my employer fire me without cause?",
                    "What should I know about filing a small claims court case?",
                    "How does copyright work for content I create online?",
                    "What are the legal requirements for starting an LLC?",
                ],
            },
            {
                "text": "Always mention that laws vary by jurisdiction when discussing legal matters",
                "keywords": ["jurisdiction", "vary by state", "depends on your location",
                             "varies by", "state law", "local law", "federal",
                             "different states", "your area", "state to state",
                             "check your local"],
                "checkpoint_prompts": [
                    "What's the statute of limitations for breach of contract?",
                    "Is it legal to record conversations without consent?",
                    "What are the noise ordinance rules for residential areas?",
                    "How does non-compete enforcement work?",
                    "What are the tenant rights regarding security deposit returns?",
                ],
            },
        ],
        "natural_setups": [
            "I'll be asking you legal questions. Keep in mind: {c1}. Also, {c2}.",
            "I need guidance on some legal matters. Important ground rules: {c1}. And {c2}.",
            "For our legal discussions: {c1}. Plus, {c2}. These are important to me.",
        ],
        "flows": [
            [
                "I'm thinking about starting a small business. What legal structure should I choose?",
                "What's the difference between an LLC and an S-Corp?",
                "How do I actually register an LLC?",
                "CHECKPOINT",
                "What licenses or permits might I need for a food business?",
                "Do I need a separate business bank account?",
                "How do I protect my business name and logo?",
                "CHECKPOINT",
                "What insurance do I need as a small business?",
                "How do I handle taxes as a new business owner?",
                "CHECKPOINT",
                "What should be in my employment contracts when I start hiring?",
                "How do I handle independent contractors vs employees legally?",
                "What about liability waivers?",
                "CHECKPOINT",
                "What happens if a customer gets injured on my premises?",
                "How should I handle customer data from a legal perspective?",
                "CHECKPOINT",
            ],
            [
                "My landlord hasn't fixed a major water leak in 3 weeks. What are my options?",
                "What's considered 'habitable' condition for a rental unit?",
                "Can I legally withhold rent until repairs are made?",
                "CHECKPOINT",
                "My lease says no pets but I have an emotional support animal. Am I protected?",
                "How much notice must a landlord give before entering my apartment?",
                "CHECKPOINT",
                "My security deposit was $2000 and the landlord won't return it. Options?",
                "How do I write a formal demand letter?",
                "If I go to small claims court, what evidence should I gather?",
                "CHECKPOINT",
                "My neighbor is harassing me and the landlord won't act. Legal recourse?",
                "Can my landlord raise rent in the middle of a lease term?",
                "CHECKPOINT",
                "I want to break my lease early due to a job relocation. Consequences?",
                "What about subletting — can a landlord refuse?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 4. ACADEMIC TUTORING ────────────────────────────────
    "academic_tutoring": {
        "name": "Academic Tutoring",
        "constraints": [
            {
                "text": "Always show step-by-step work when solving math or science problems",
                "keywords": ["step 1", "step 2", "step 3", "first,", "next,",
                             "then,", "therefore", "step-by-step",
                             "let's break", "breaking this down",
                             "working through"],
                "checkpoint_prompts": [
                    "Solve the integral of x²·sin(x) dx",
                    "Find the eigenvalues of the matrix [[3,1],[1,3]]",
                    "Calculate the pH of a 0.05M acetic acid solution (Ka = 1.8×10⁻⁵)",
                    "Determine the moment of inertia of a solid cylinder about its axis",
                    "Solve the differential equation dy/dx = 2xy with y(0) = 1",
                ],
            },
            {
                "text": "Always use real-world analogies when explaining abstract concepts",
                "keywords": ["like", "similar to", "analogy", "imagine",
                             "think of it as", "just like", "real-world",
                             "for example", "everyday", "picture"],
                "checkpoint_prompts": [
                    "What is entropy in thermodynamics?",
                    "Explain how neural networks learn",
                    "What is quantum superposition?",
                    "How does public key cryptography work?",
                    "Explain the concept of recursion in computer science",
                ],
            },
            {
                "text": "Always provide a practice problem at the end of explanations for the student to try",
                "keywords": ["practice", "try this", "exercise", "your turn",
                             "challenge", "work out", "solve this",
                             "here's a problem", "test yourself",
                             "try solving"],
                "checkpoint_prompts": [
                    "Teach me about derivatives using the power rule",
                    "Explain the concept of probability distributions",
                    "How do balanced binary search trees work?",
                    "Explain electromagnetic induction",
                    "Teach me about chemical equilibrium",
                ],
            },
        ],
        "natural_setups": [
            "I'm studying for exams and need your help as a tutor. My preferences: {c1}. Also, {c2}.",
            "Let's do a tutoring session. I learn best when you {c1}. Also always {c2}.",
            "I need help understanding some concepts. Please {c1}, and also {c2}. That helps me learn.",
        ],
        "flows": [
            [
                "I'm starting a calculus course. Can you help me understand limits?",
                "What's the intuition behind a limit approaching infinity?",
                "How do I evaluate limits with indeterminate forms like 0/0?",
                "CHECKPOINT",
                "Now let's move to derivatives. What's the formal definition?",
                "How is the derivative related to the slope of a tangent line?",
                "CHECKPOINT",
                "Can you explain the chain rule?",
                "When do I use the product rule vs the chain rule?",
                "CHECKPOINT",
                "Let's move to integration. How is it the reverse of differentiation?",
                "What's the difference between definite and indefinite integrals?",
                "How does the fundamental theorem of calculus connect them?",
                "CHECKPOINT",
                "Teach me integration by parts",
                "When should I use substitution vs integration by parts?",
                "CHECKPOINT",
            ],
            [
                "I'm studying physics — specifically mechanics. Let's start with Newton's laws.",
                "What's the difference between mass and weight?",
                "CHECKPOINT",
                "How do free body diagrams work?",
                "Can you walk me through solving a problem with pulleys?",
                "CHECKPOINT",
                "What about friction — static vs kinetic?",
                "How does friction depend on the normal force?",
                "Let's talk about circular motion. What creates centripetal force?",
                "CHECKPOINT",
                "How does conservation of energy work in mechanics?",
                "What's the work-energy theorem?",
                "CHECKPOINT",
                "Teach me about momentum and collisions",
                "What's the difference between elastic and inelastic collisions?",
                "How do I solve 2D collision problems?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 5. CUSTOMER SUPPORT ─────────────────────────────────
    "customer_support": {
        "name": "Customer Support",
        "constraints": [
            {
                "text": "Always acknowledge the customer's frustration before providing solutions",
                "keywords": ["understand", "sorry", "frustrat", "apologize",
                             "appreciate your patience", "I can see",
                             "that must be", "inconvenien", "hear that"],
                "checkpoint_prompts": [
                    "My internet has been dropping every 10 minutes for the past week!",
                    "I was charged twice for my subscription and nobody is helping me!",
                    "Your app keeps crashing and I've lost my work three times today!",
                    "I've been waiting for my refund for 3 weeks now!",
                    "My new laptop arrived with a cracked screen!",
                ],
            },
            {
                "text": "Always offer to escalate if the issue isn't resolved immediately",
                "keywords": ["escalat", "supervisor", "specialist", "senior",
                             "team", "further assistance", "next level",
                             "priority", "manager", "higher"],
                "checkpoint_prompts": [
                    "I've tried all the basic troubleshooting and nothing works",
                    "This is the third time I'm calling about the same billing issue",
                    "The error still hasn't been fixed after my last two contacts",
                    "Your basic plan doesn't cover what I need and the upgrade path is confusing",
                    "I've been a loyal customer for 10 years and I'm considering canceling",
                ],
            },
        ],
        "natural_setups": [
            "I need customer support help. Please {c1}. And {c2}. I've had bad experiences before.",
            "I'm reaching out about an issue. I'd appreciate it if you would {c1}. Also, {c2}.",
            "Before we start troubleshooting: {c1}. Additionally, {c2}. Thanks.",
        ],
        "flows": [
            [
                "My home WiFi keeps disconnecting randomly throughout the day.",
                "It's been happening for about a week. Before that it was working fine.",
                "I have the XR-500 router you guys provided.",
                "CHECKPOINT",
                "I already tried restarting the router. It didn't help.",
                "The lights on the router — the internet light blinks orange sometimes.",
                "It happens on all my devices, not just one specific one.",
                "CHECKPOINT",
                "I work from home so this is really affecting my job.",
                "Is there a way to check if the issue is the router or the line?",
                "My speed test shows 5 Mbps but I'm paying for 100.",
                "CHECKPOINT",
                "Could it be a firmware issue on the router?",
                "I tried ethernet directly and it's still slow. So it's the line.",
                "CHECKPOINT",
                "What's the next step? I need this fixed urgently.",
                "Can I get a credit for the downtime this month?",
                "CHECKPOINT",
            ],
            [
                "I noticed I was charged $99.99 twice on my credit card from your company.",
                "The charges are dated January 15 and January 16.",
                "My account number is AC-12345. Can you investigate?",
                "CHECKPOINT",
                "I only signed up for one monthly subscription.",
                "No, I didn't get any email about a second charge.",
                "Can you tell me what each charge is actually for?",
                "CHECKPOINT",
                "So one is the subscription and the other is unexplained? That's worrying.",
                "I want the duplicate charge refunded immediately.",
                "How long will the refund take to process?",
                "CHECKPOINT",
                "I want to make sure this doesn't happen again next month.",
                "Is there a way to get email confirmations before each charge?",
                "CHECKPOINT",
                "While we're at it, can I downgrade my plan? I'm overpaying.",
                "Thanks for dealing with all of this for me.",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 6. CREATIVE WRITING ─────────────────────────────────
    "creative_writing": {
        "name": "Creative Writing",
        "constraints": [
            {
                "text": "Always use vivid sensory details in story descriptions (sight, sound, smell, touch, taste)",
                "keywords": ["smell", "sound", "taste", "touch", "sight",
                             "hear", "felt", "scent", "aroma", "whisper",
                             "echo", "glow", "rough", "smooth", "bitter",
                             "warm", "cold", "bright"],
                "checkpoint_prompts": [
                    "Write a scene where the protagonist enters an old library",
                    "Describe the villain's lair in the abandoned factory",
                    "Write the opening scene at a bustling farmers market",
                    "Describe the moment the character arrives at the coastal house",
                    "Write the scene where they discover the hidden garden behind the wall",
                ],
            },
            {
                "text": "Always include dialogue between characters in story passages",
                "keywords": ["said", "asked", "replied", "whispered",
                             "shouted", '"', "told", "answered",
                             "muttered", "exclaimed"],
                "checkpoint_prompts": [
                    "Write a scene where two strangers meet at a coffee shop during a storm",
                    "Write the confrontation between the detective and the suspect",
                    "Write the scene where the mentor gives the hero crucial advice",
                    "Write the reunion scene between the siblings after years apart",
                    "Write the scene where the team debates the plan before the mission",
                ],
            },
            {
                "text": "Every story passage should end with a hook or cliffhanger that creates suspense",
                "keywords": ["but then", "suddenly", "little did",
                             "what", "before", "unknown",
                             "couldn't have known", "about to change",
                             "door opened", "shadow", "froze", "silence",
                             "realized", "too late"],
                "checkpoint_prompts": [
                    "Write the next chapter where Maya investigates the basement",
                    "Continue the story — Alex arrives at the meeting point but something is wrong",
                    "Write the scene where they finally open the mysterious package",
                    "Continue with what happens during the night watch at the cabin",
                    "Write what happens when the power goes out in the hospital",
                ],
            },
        ],
        "natural_setups": [
            "I'm writing a novel and I want your help with scenes. My style preferences: {c1}. Also, {c2}.",
            "Let's collaborate on creative writing. I want you to {c1}. And please {c2}.",
            "Help me with my story. Important writing rules: {c1}. And {c2}. Don't forget these.",
        ],
        "flows": [
            [
                "I'm writing a mystery novel set in a coastal town. Help me develop the concept.",
                "The protagonist is a retired detective named Maya. What's her personality?",
                "What's the central mystery? I'm thinking something about a disappearance.",
                "CHECKPOINT",
                "Let's develop the supporting cast. Who are the main suspects?",
                "What's Maya's relationship with the local police chief?",
                "What's the town's dark history that connects to the mystery?",
                "CHECKPOINT",
                "Let's work on the first major clue Maya discovers.",
                "How does she find out the missing person had a secret life?",
                "CHECKPOINT",
                "Now I need a red herring. What misleads Maya?",
                "How does she realize the red herring was a dead end?",
                "What's the midpoint twist that changes everything?",
                "CHECKPOINT",
                "How does the climax build? What final clue does Maya uncover?",
                "Write the revelation scene where Maya confronts the truth.",
                "CHECKPOINT",
            ],
            [
                "I want to write a sci-fi short story about first contact. Not an invasion plot.",
                "Setting: a deep-space research station orbiting Europa.",
                "Main character is Dr. Alex Chen, a linguist sent to decode alien signals.",
                "CHECKPOINT",
                "What should the alien communication look like? Something truly alien.",
                "How does Alex first realize the signals are structured language?",
                "CHECKPOINT",
                "Alex makes a breakthrough — they decode the first concept. What is it?",
                "What's the emotional impact on the crew when they confirm it's real?",
                "How does Earth react to the announcement?",
                "CHECKPOINT",
                "The aliens ask a question that changes everything. What do they ask?",
                "How does this question divide the crew about how to respond?",
                "What's Alex's internal conflict about the implications?",
                "CHECKPOINT",
                "Write the ending scene where Alex makes the final decision on how to respond.",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 7. DATA ANALYSIS ────────────────────────────────────
    "data_analysis": {
        "name": "Data Analysis",
        "constraints": [
            {
                "text": "Always state the sample size when presenting any statistical findings",
                "keywords": ["n =", "n=", "sample size", "sample of",
                             "observations", "records", "data points",
                             "respondents", "participants", "rows"],
                "checkpoint_prompts": [
                    "Summarize the key findings from this customer churn analysis",
                    "What does the correlation between price and sales tell us?",
                    "Present the A/B test results for the new checkout flow",
                    "What trends do you see in the quarterly revenue data?",
                    "Summarize the employee satisfaction survey results",
                ],
            },
            {
                "text": "Always mention potential biases or limitations when interpreting data",
                "keywords": ["bias", "limitation", "caveat", "however",
                             "important to note", "keep in mind",
                             "may not", "does not imply causation",
                             "correlation", "confound", "caution"],
                "checkpoint_prompts": [
                    "Based on this data, should we expand into the European market?",
                    "The data shows campaign X users convert better. Should we double the budget?",
                    "Our survey shows 80% satisfaction. How should we interpret this?",
                    "The model predicts 95% accuracy. Can we deploy it to production?",
                    "Revenue clearly increases with ad spend. Should we keep spending more?",
                ],
            },
        ],
        "natural_setups": [
            "I'm analyzing data for a presentation. When presenting findings, please {c1}. Also, {c2}.",
            "I need help with data analysis. Two ground rules: {c1}. And {c2}. Rigor matters.",
            "We're doing a data project together. Always {c1}. Plus {c2}. I need bulletproof analysis.",
        ],
        "flows": [
            [
                "I have a dataset of 50,000 e-commerce transactions. Help me plan the analysis.",
                "What key metrics should I compute first?",
                "How do I calculate customer lifetime value?",
                "CHECKPOINT",
                "Average order value is $67. How should I segment customers?",
                "How do I identify which customers are at risk of churning?",
                "CHECKPOINT",
                "The RFM analysis shows 5 distinct segments. What do I do with this?",
                "How do I run a cohort analysis on this data?",
                "What visualizations would tell the best story to stakeholders?",
                "CHECKPOINT",
                "I want to build a churn prediction model. What features should I use?",
                "How do I handle the class imbalance — only 8% of customers churned?",
                "CHECKPOINT",
                "The model is ready. How do I present these findings to the CEO?",
                "What actionable recommendations should I make?",
                "CHECKPOINT",
            ],
            [
                "We're running an A/B test on our pricing page. Help me design it right.",
                "How do I calculate the minimum sample size for the test?",
                "What should the control and variation look like?",
                "CHECKPOINT",
                "The test has been running 2 weeks. When should I check results?",
                "How do I account for day-of-week effects in the data?",
                "CHECKPOINT",
                "Variation B shows 12% improvement in conversion. Is that significant?",
                "The p-value is 0.04. How should I interpret that?",
                "Should I run the test longer or call it now?",
                "CHECKPOINT",
                "Mobile users actually did worse on B. Desktop users did great. Now what?",
                "How do I present these mixed results to the product team?",
                "CHECKPOINT",
                "They want to run another test on checkout. How do we avoid test interference?",
                "What lessons from this test apply to future experiments?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 8. PROJECT MANAGEMENT ───────────────────────────────
    "project_management": {
        "name": "Project Management",
        "constraints": [
            {
                "text": "Always identify risks and mitigation strategies when proposing plans",
                "keywords": ["risk", "mitigat", "contingenc", "fallback",
                             "threat", "backup plan", "what if",
                             "potential issue", "risk of"],
                "checkpoint_prompts": [
                    "We need to migrate our monolith to microservices in 6 months. Plan?",
                    "The client wants to add 3 new features mid-sprint. How to handle it?",
                    "We're launching in a new market next quarter. What's the project plan?",
                    "Our lead developer just resigned. How do we ensure continuity?",
                    "We need to upgrade infrastructure while keeping the service running.",
                ],
            },
            {
                "text": "Always break down work into measurable milestones with clear deliverables",
                "keywords": ["milestone", "deliverable", "phase", "sprint",
                             "week 1", "week 2", "deadline", "target",
                             "checkpoint", "KPI", "by end of"],
                "checkpoint_prompts": [
                    "Plan the development of our new mobile app from scratch",
                    "We need to migrate from AWS to GCP. Break down the work.",
                    "Outline the plan for implementing the new CRM system",
                    "We're redesigning the customer onboarding flow. Plan it out.",
                    "Create a project plan for setting up a CI/CD pipeline from scratch",
                ],
            },
        ],
        "natural_setups": [
            "I'm managing a project and need your help planning. When making plans: {c1}. Also, {c2}.",
            "You're my project advisor. Important: {c1}. And always {c2}. I need thorough plans.",
            "Help me with project planning. Ground rules: {c1}. Also {c2}.",
        ],
        "flows": [
            [
                "We're launching a new SaaS product in 4 months. Help with the plan.",
                "The team is 5 developers, 2 designers, 1 QA. How should I allocate them?",
                "What should the first sprint focus on?",
                "CHECKPOINT",
                "We just found out a competitor is launching something similar in 3 months.",
                "Should we cut scope or speed up development?",
                "CHECKPOINT",
                "The CEO wants to add an AI feature that wasn't in scope.",
                "How do I push back without seeming uncooperative?",
                "What if we phase it — MVP first, AI post-launch?",
                "CHECKPOINT",
                "One of our devs is struggling with the auth module and it's blocking others.",
                "How should I handle sprint retrospectives effectively?",
                "We're behind by 1 sprint. What are our options for getting back on track?",
                "CHECKPOINT",
                "QA found 47 bugs in the latest build. How do we triage?",
                "We need to start marketing prep. When should that happen relative to dev?",
                "CHECKPOINT",
            ],
            [
                "We need to scale the engineering team from 8 to 25 in 6 months. Plan?",
                "What's the ideal team structure at that size?",
                "How do I maintain code quality while onboarding so many people?",
                "CHECKPOINT",
                "What interview process works for hiring at this pace?",
                "How do I handle the culture shift from tripling the team?",
                "CHECKPOINT",
                "Some senior engineers are showing burnout signs from all the mentoring.",
                "How do I measure if new hires are ramping up effectively?",
                "What documentation should we prepare before they start?",
                "CHECKPOINT",
                "Two new hires aren't performing well after 3 months. What now?",
                "How do I structure a fair performance improvement process?",
                "CHECKPOINT",
                "Team morale is lower since we scaled up. How to address it?",
                "What processes need to change for this team size?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 9. COOKING & RECIPES ────────────────────────────────
    "cooking_recipes": {
        "name": "Cooking & Recipes",
        "constraints": [
            {
                "text": "Always mention cooking temperatures in both Celsius and Fahrenheit",
                "keywords": ["°C", "°F", "Celsius", "Fahrenheit",
                             "degrees C", "degrees F", "C/", "F/",
                             "350F", "180C", "375F", "190C"],
                "checkpoint_prompts": [
                    "How do I make a classic sourdough bread from scratch?",
                    "What's the recipe for a proper beef Wellington?",
                    "How do I roast a whole chicken so it's juicy inside?",
                    "What's the technique for making crème brûlée?",
                    "How do I bake French macarons that don't crack?",
                ],
            },
            {
                "text": "Always suggest a common allergen substitution for each recipe",
                "keywords": ["substitut", "alternative", "replace",
                             "instead of", "dairy-free", "gluten-free",
                             "allergy", "intoleran", "for those who",
                             "if you can't have", "swap"],
                "checkpoint_prompts": [
                    "Give me a recipe for classic chocolate chip cookies",
                    "How do I make authentic pasta carbonara?",
                    "What's a good recipe for fluffy banana pancakes?",
                    "How do I make a birthday cake from scratch?",
                    "What's the recipe for homemade pizza dough?",
                ],
            },
        ],
        "natural_setups": [
            "I'm learning to cook. When you give recipes, {c1}. Also, {c2}. I have friends with allergies.",
            "Help me in the kitchen! Please {c1}. And {c2} since I cook for people with different dietary needs.",
            "I want cooking help. Two things to always include: {c1}. And {c2}.",
        ],
        "flows": [
            [
                "I'm a complete beginner at baking. Where should I start?",
                "What basic equipment do I need in my kitchen?",
                "What's the science behind how flour, sugar, eggs, and butter interact?",
                "CHECKPOINT",
                "I tried making cookies but they spread flat. What went wrong?",
                "What's the difference between baking soda and baking powder?",
                "How does altitude affect baking results?",
                "CHECKPOINT",
                "Let's try bread. How does yeast actually work?",
                "What type of flour is best for bread?",
                "How do I know when dough is kneaded enough?",
                "CHECKPOINT",
                "My bread didn't rise. What are common reasons for that?",
                "Active dry yeast vs instant — what's the difference?",
                "CHECKPOINT",
                "I want to try pastries next. What's the easiest to start with?",
                "How do I make puff pastry from scratch?",
                "CHECKPOINT",
            ],
            [
                "I want to meal prep for the whole week. Healthy but tasty.",
                "I need high-protein meals since I'm training.",
                "What proteins are most cost-effective for batch cooking?",
                "CHECKPOINT",
                "I need variety — I can't eat chicken breast every day.",
                "What marinades work well for meal prep that won't get soggy?",
                "How do I store meal-prepped food properly?",
                "CHECKPOINT",
                "Let's include a vegetarian day. What high-protein options work?",
                "How do I make tofu actually taste good?",
                "CHECKPOINT",
                "Soups seem great for meal prep. Best approach?",
                "How long do homemade soups last in the fridge vs freezer?",
                "CHECKPOINT",
                "Put together a complete 5-day meal plan with everything we discussed.",
                "How much should I budget for this weekly meal prep?",
                "CHECKPOINT",
            ],
        ],
    },

    # ─── 10. PERSONAL FINANCE ────────────────────────────────
    "personal_finance": {
        "name": "Personal Finance",
        "constraints": [
            {
                "text": "Always remind that past performance doesn't guarantee future returns when discussing investments",
                "keywords": ["past performance", "no guarantee",
                             "doesn't guarantee", "not indicative",
                             "future returns", "risk", "may lose",
                             "historical", "guarantee"],
                "checkpoint_prompts": [
                    "Should I put my savings in an S&P 500 index fund?",
                    "What do you think about investing in tech stocks right now?",
                    "My friend made 40% on crypto last year. Should I invest too?",
                    "Is real estate a safe investment compared to stocks?",
                    "What about dividend stocks for building passive income?",
                ],
            },
            {
                "text": "Always recommend consulting a certified financial advisor for personalized decisions",
                "keywords": ["financial advisor", "certified", "professional advice",
                             "planner", "consult a", "personalized",
                             "CPA", "CFP", "financial professional"],
                "checkpoint_prompts": [
                    "How should I allocate my retirement savings at age 30?",
                    "Should I pay off my mortgage early or invest that money?",
                    "I just inherited $200K. What's the smartest thing to do?",
                    "How should I adjust investments as I near retirement?",
                    "What's the best tax strategy for my income level?",
                ],
            },
        ],
        "natural_setups": [
            "I want help with financial planning. When discussing money matters: {c1}. Also, {c2}.",
            "I'm learning about finance. Please {c1}. And always {c2}. I want responsible guidance.",
            "Before we discuss investments: {c1}. And {c2}. I value careful, honest advice.",
        ],
        "flows": [
            [
                "I'm 28 and just started making decent money. How do I start investing?",
                "What's the difference between a 401k and an IRA?",
                "How much of my income should I save vs invest?",
                "CHECKPOINT",
                "Should I pay off student loans first or start investing now?",
                "What about building an emergency fund? How much do I need?",
                "How do index funds actually work?",
                "CHECKPOINT",
                "Should I have bonds at my age? What's the right allocation?",
                "How does dollar-cost averaging work?",
                "What's the difference between growth and value investing?",
                "CHECKPOINT",
                "Roth vs Traditional IRA — which is better for someone my age?",
                "What about real estate — REITs vs actually buying property?",
                "CHECKPOINT",
                "How do I know when to rebalance my portfolio?",
                "What tax implications should I understand for investments?",
                "CHECKPOINT",
            ],
            [
                "I have $45K in total debt across credit cards, student loans, and a car. Help me plan.",
                "Credit card: 22% APR. Student loans: 5.5%. Car: 6.9%. What order to pay?",
                "Explain the avalanche method vs the snowball method.",
                "CHECKPOINT",
                "Should I consider debt consolidation?",
                "How do balance transfers work? Worth doing?",
                "CHECKPOINT",
                "My employer matches 401k up to 5%. Should I still invest while paying off debt?",
                "What's the minimum I should put toward retirement?",
                "How do I negotiate lower interest rates on credit cards?",
                "CHECKPOINT",
                "Once I'm debt-free, how do I stay out of debt?",
                "How do I rebuild my credit score after paying everything off?",
                "CHECKPOINT",
                "With debt handled, what should my monthly budget look like going forward?",
                "What percentage should go to needs vs wants vs savings?",
                "CHECKPOINT",
            ],
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════

def _lower_constraint(text: str) -> str:
    """Make constraint text suitable for natural insertion (lowercase start)."""
    if text and text[0].isupper():
        return text[0].lower() + text[1:]
    return text


def generate_conversation(
    domain_key: str = None,
    num_constraints: int = 2,
    target_turns: int = 50,
    seed: int = None,
) -> dict:
    """Generate a single realistic conversation."""
    if seed is not None:
        random.seed(seed)

    # Pick domain
    if domain_key is None:
        domain_key = random.choice(list(DOMAINS.keys()))
    domain = DOMAINS[domain_key]

    # Pick constraints
    available = domain["constraints"]
    num_constraints = min(num_constraints, len(available))
    selected_constraints = random.sample(available, num_constraints)

    # Pick a flow
    flow_template = random.choice(domain["flows"])

    # Build natural setup message
    setup_template = random.choice(domain["natural_setups"])
    constraint_texts = [c["text"] for c in selected_constraints]
    c_inserts = [_lower_constraint(t) for t in constraint_texts]
    if len(c_inserts) == 1:
        setup_msg = setup_template.replace("{c1}", c_inserts[0]).replace(". Also, {c2}", "")
        setup_msg = setup_msg.replace(". And {c2}", "").replace(". Plus {c2}", "")
        setup_msg = setup_msg.replace(", and also {c2}", "").replace(". Also {c2}", "")
    else:
        setup_msg = setup_template.format(c1=c_inserts[0], c2=c_inserts[1])

    turns = []
    checkpoints = []
    turn_num = 0

    # Turn 1: natural constraint setup
    turn_num += 1
    turns.append({
        "turn": turn_num,
        "role": "user",
        "text": setup_msg,
        "type": "constraint_setup",
    })
    turns.append({
        "turn": turn_num,
        "role": "assistant",
        "text": "Got it! I'll keep those preferences in mind throughout our conversation.",
        "type": "ack",
    })

    # Process flow template
    checkpoint_idx = 0
    for item in flow_template:
        turn_num += 1

        if item == "CHECKPOINT":
            # Pick which constraint to test (round-robin)
            c = selected_constraints[checkpoint_idx % len(selected_constraints)]
            checkpoint_idx += 1

            # Pick a checkpoint prompt for this constraint
            prompt = random.choice(c["checkpoint_prompts"])

            turns.append({
                "turn": turn_num,
                "role": "user",
                "text": prompt,
                "type": "checkpoint",
            })
            turns.append({
                "turn": turn_num,
                "role": "assistant",
                "text": "[RESPONSE_PLACEHOLDER]",
                "type": "checkpoint_response",
            })

            checkpoints.append({
                "turn": turn_num,
                "constraint_tested": c["text"],
                "test": f"Does response follow: '{c['text']}'?",
                "answer": True,
                "keywords": c["keywords"],
            })
        else:
            # Regular flow turn
            turns.append({
                "turn": turn_num,
                "role": "user",
                "text": item,
                "type": "topic",
            })
            turns.append({
                "turn": turn_num,
                "role": "assistant",
                "text": "[RESPONSE_PLACEHOLDER]",
                "type": "topic_response",
            })

    # Pad to target_turns if needed with domain-relevant follow-ups
    padding_questions = _get_padding_questions(domain_key)
    pad_idx = 0
    while turn_num < target_turns:
        turn_num += 1
        if pad_idx < len(padding_questions):
            q = padding_questions[pad_idx]
            pad_idx += 1
        else:
            q = f"Can you elaborate more on what we discussed earlier about the last point?"
        turns.append({
            "turn": turn_num,
            "role": "user",
            "text": q,
            "type": "followup",
        })
        turns.append({
            "turn": turn_num,
            "role": "assistant",
            "text": "[RESPONSE_PLACEHOLDER]",
            "type": "followup_response",
        })

        # Optionally add late checkpoints for distant recall testing
        if turn_num >= target_turns - 5 and turn_num == target_turns - 2:
            turn_num += 1
            c = random.choice(selected_constraints)
            prompt = random.choice(c["checkpoint_prompts"])
            turns.append({
                "turn": turn_num,
                "role": "user",
                "text": prompt,
                "type": "checkpoint",
            })
            turns.append({
                "turn": turn_num,
                "role": "assistant",
                "text": "[RESPONSE_PLACEHOLDER]",
                "type": "checkpoint_response",
            })
            checkpoints.append({
                "turn": turn_num,
                "constraint_tested": c["text"],
                "test": f"Does response follow: '{c['text']}'?",
                "answer": True,
                "keywords": c["keywords"],
            })

    return {
        "conversation_id": f"realistic_{domain_key}_{seed or random.randint(0, 99999):05d}",
        "domain": domain_key,
        "domain_name": domain["name"],
        "constraints": constraint_texts,
        "num_turns": turn_num,
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "turns": turns,
    }


def _get_padding_questions(domain_key: str) -> List[str]:
    """Domain-relevant padding questions for extending conversations."""
    padding = {
        "software_engineering": [
            "What about logging best practices?",
            "How should I handle configuration management?",
            "What's the best approach for API versioning?",
            "How do I document the API properly?",
            "What about database connection pooling?",
            "How should I handle background jobs?",
            "What CI/CD pipeline would you recommend?",
            "How do I handle feature flags?",
            "What about database indexing strategies?",
            "How should I handle API authentication tokens?",
            "What's the best way to handle file uploads?",
            "How do I implement pagination efficiently?",
            "What about input validation and sanitization?",
            "How should I handle database transactions?",
            "What monitoring and alerting should I set up?",
        ],
        "medical_consultation": [
            "How important is hydration for overall health?",
            "What role does sleep play in recovery?",
            "How does meditation affect physical health?",
            "What are the benefits of regular walking?",
            "How does meal timing affect health?",
            "What role does gut health play in immunity?",
            "How important is social connection for health?",
            "What about the effects of screen time on health?",
            "How does air quality affect respiratory health?",
            "What about the impact of posture on back health?",
            "How does sunlight exposure affect mood?",
            "What role does fiber play in digestion?",
            "How important is stretching for flexibility?",
            "What about the health effects of coffee?",
            "How does chronic stress affect the immune system?",
        ],
        "legal_advisory": [
            "What about power of attorney — do I need one?",
            "How do trusts work compared to wills?",
            "What's the process for getting a restraining order?",
            "How does mediation compare to going to court?",
            "What are my rights during a traffic stop?",
            "How does intellectual property protection work online?",
            "What about liability for user-generated content?",
            "How do warranties work for consumer purchases?",
            "What are my rights if I buy a defective product?",
            "How does bankruptcy work — Chapter 7 vs Chapter 13?",
            "What about prenuptial agreements?",
            "How do homeowner association rules work legally?",
            "What are my rights regarding neighbors' trees?",
            "How does the small claims court process work?",
            "What about legal protections for whistleblowers?",
        ],
        "academic_tutoring": [
            "Can you review the key concepts we've covered so far?",
            "What are common mistakes students make on this topic?",
            "How is this applied in real engineering?",
            "What's the history behind this mathematical concept?",
            "How does this topic connect to the next chapter?",
            "What are the prerequisites I should be solid on?",
            "How would this appear on a typical exam?",
            "What resources do you recommend for practice?",
            "Can you explain the geometric interpretation?",
            "How do different textbooks approach this differently?",
            "What are the boundary cases I should watch out for?",
            "How does this generalize to higher dimensions?",
            "What's an intuitive proof I can remember?",
            "How do I know which technique to apply?",
            "What should I study next to build on this?",
        ],
        "customer_support": [
            "Will this fix be permanent or might the issue come back?",
            "Is there a way to monitor this on my end going forward?",
            "Are other customers experiencing similar problems?",
            "What caused this issue in the first place?",
            "Is there a service status page I can check?",
            "What's the expected timeframe for resolution?",
            "Can I get a reference number for this interaction?",
            "What happens if the problem comes back after the fix?",
            "Is there a way to prevent this in the future?",
            "Do you have any maintenance windows coming up?",
            "What's the best way to reach support if this recurs?",
            "Can I get a summary of everything we discussed by email?",
            "Are there any known issues with my equipment model?",
            "What firmware version should my router be on?",
            "Is there a diagnostic tool I can run myself?",
        ],
        "creative_writing": [
            "How do I avoid info-dumping in exposition?",
            "What makes a compelling antagonist?",
            "How do I handle pacing in a thriller?",
            "What's the best way to write a flashback?",
            "How do I make dialogue sound natural?",
            "What's the difference between showing and telling?",
            "How do I create tension without action scenes?",
            "What point of view works best for this story?",
            "How long should chapters be?",
            "How do I develop a subplot that enhances the main plot?",
            "What makes a satisfying plot twist?",
            "How do I write an unreliable narrator effectively?",
            "What's the role of setting as a character?",
            "How do I handle multiple timelines?",
            "What makes an ending feel earned?",
        ],
        "data_analysis": [
            "What tools do you recommend for visualization?",
            "How do I handle missing data in my dataset?",
            "What's the best way to detect outliers?",
            "How do I normalize data from different scales?",
            "What's the difference between correlation and regression?",
            "How do I choose between classification and regression models?",
            "What feature engineering techniques should I try?",
            "How do I validate my model properly?",
            "What's cross-validation and when should I use it?",
            "How do I explain my model to non-technical stakeholders?",
            "What about data privacy considerations?",
            "How do I handle categorical variables with many levels?",
            "What's the difference between precision and recall?",
            "How do I detect multicollinearity?",
            "What dashboard tool should I use for the final report?",
        ],
        "project_management": [
            "How do I handle conflicting priorities between stakeholders?",
            "What project management methodology fits a fast-moving startup?",
            "How do I estimate timelines when the scope is uncertain?",
            "What's the best way to track dependencies across teams?",
            "How do I handle technical debt in sprint planning?",
            "What metrics indicate a healthy project?",
            "How often should I do status reports?",
            "What's the best way to onboard a new team member mid-project?",
            "How do I handle scope creep diplomatically?",
            "What's the role of a project manager vs a product manager?",
            "How do I run effective standup meetings?",
            "What tools do you recommend for project tracking?",
            "How do I manage stakeholder expectations realistically?",
            "What's the best way to handle post-mortems?",
            "How do I balance speed vs quality in decision-making?",
        ],
        "cooking_recipes": [
            "What's the best way to sharpen kitchen knives?",
            "How do I properly season a cast iron pan?",
            "What's the difference between stock and broth?",
            "How do I prevent pasta from sticking together?",
            "What herbs go well together?",
            "How do I know when oil is hot enough for frying?",
            "What's the best way to store fresh herbs?",
            "How do I make a basic roux?",
            "What's the Maillard reaction and why does it matter?",
            "How do I deglaze a pan properly?",
            "What's the best way to make homemade broth?",
            "How do I fix a sauce that's too salty?",
            "What's the difference between sautéing and frying?",
            "How do I meal prep without food getting soggy?",
            "What are essential pantry staples I should always have?",
        ],
        "personal_finance": [
            "What's the difference between APR and APY?",
            "How does compound interest actually work?",
            "What's a good credit score to aim for?",
            "How do I read my credit report?",
            "What's the 50/30/20 budgeting rule?",
            "How do dividends get taxed?",
            "What's the difference between a stockbroker and a robo-advisor?",
            "How does inflation affect my savings?",
            "What's the benefit of an HSA account?",
            "How do I avoid lifestyle inflation?",
            "What's the difference between term and whole life insurance?",
            "How do I plan for large purchases without going into debt?",
            "What's the best approach to teaching kids about money?",
            "How do I financially prepare for a recession?",
            "What tax deductions should I know about?",
        ],
    }
    return padding.get(domain_key, [f"Can you expand on that last point?"] * 15)


def generate_dataset(
    num_conversations: int = 50,
    turns_per_convo: int = 50,
    output_path: Path = None,
    domains: List[str] = None,
):
    """Generate the full realistic constraint-tracking dataset.

    Output structure:
        output_path/
            synthetic/
                software_engineering.json
                medical_consultation.json
                ...
            dataset_meta.json
    The benchmark loader globs all *.json under the directory tree,
    so per-domain files are auto-discovered.
    """
    output_path = output_path or (
        Path(__file__).parent.parent / "datasets" / "constraint_tracking"
    )
    synthetic_dir = output_path / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # Also create placeholders for future data sources
    (output_path / "llm_generated").mkdir(exist_ok=True)
    (output_path / "real_exports").mkdir(exist_ok=True)

    if domains is None:
        domains = list(DOMAINS.keys())

    # Generate conversations grouped by domain
    domain_convos: Dict[str, list] = {d: [] for d in domains}
    convos_per_domain = max(1, num_conversations // len(domains))
    extra = num_conversations - convos_per_domain * len(domains)

    seed_counter = 0
    for domain_key in domains:
        n = convos_per_domain + (1 if extra > 0 else 0)
        if extra > 0:
            extra -= 1
        for _ in range(n):
            conv = generate_conversation(
                domain_key=domain_key,
                num_constraints=random.randint(1, 2),
                target_turns=turns_per_convo,
                seed=seed_counter,
            )
            domain_convos[domain_key].append(conv)
            seed_counter += 1

    # Write per-domain files
    all_convos = []
    domain_counts = {}
    for domain_key, convos in domain_convos.items():
        if not convos:
            continue
        domain_file = synthetic_dir / f"{domain_key}.json"
        domain_file.write_text(json.dumps(convos, indent=2))
        all_convos.extend(convos)
        domain_counts[domain_key] = len(convos)
        print(f"  {domain_key}: {len(convos)} conversations → {domain_file}")

    # Summary
    total_checkpoints = sum(c["num_checkpoints"] for c in all_convos)
    total_turns = sum(c["num_turns"] for c in all_convos)
    print(f"\nGenerated {len(all_convos)} conversations total")
    print(f"  Total turns: {total_turns}")
    print(f"  Total checkpoints: {total_checkpoints}")
    print(f"  Avg checkpoints/convo: {total_checkpoints / len(all_convos):.1f}")
    print(f"  Domains: {len(domain_counts)}")

    # Save metadata
    meta_file = output_path / "dataset_meta.json"
    meta_file.write_text(json.dumps({
        "generator": "generate_realistic_convos.py",
        "num_conversations": len(all_convos),
        "turns_per_convo": turns_per_convo,
        "total_checkpoints": total_checkpoints,
        "data_sources": ["synthetic"],
        "domains": {
            k: {
                "name": DOMAINS[k]["name"],
                "conversations": domain_counts.get(k, 0),
                "constraints_available": len(DOMAINS[k]["constraints"]),
            }
            for k in domains
        },
    }, indent=2))
    print(f"  Metadata: {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate realistic multi-domain constraint-tracking conversations"
    )
    parser.add_argument(
        "--num-conversations", type=int, default=50,
        help="Total conversations to generate (default: 50)"
    )
    parser.add_argument(
        "--turns", type=int, default=50,
        help="Target turns per conversation (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Specific domains to use (default: all)"
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, args.turns, output, args.domains)
