"""
Medical disclaimers and safety boundaries for healthcare AI
"""

# Primary medical disclaimer
PRIMARY_DISCLAIMER = """
⚠️ **Important Medical Disclaimer**: This information is for educational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for personalized medical guidance.
"""

# Situational disclaimers
SITUATIONAL_DISCLAIMERS = {
    "symptom_assessment": """
**Please Note**: While I can provide general information about gut health symptoms, only a qualified healthcare provider can properly assess your individual situation and provide appropriate medical care.
""",
    
    "treatment_suggestions": """
**Medical Guidance**: The suggestions provided are general wellness approaches. For specific treatment recommendations, please consult with your healthcare provider who can evaluate your individual medical needs.
""",
    
    "emergency_situations": """
🚨 **Seek Immediate Medical Care**: If you're experiencing severe symptoms such as intense abdominal pain, blood in stool, persistent vomiting, or fever, please contact emergency services or visit an emergency room immediately.
""",
    
    "medication_interactions": """
💊 **Medication Safety**: Before starting any new supplements or making significant dietary changes, please discuss with your healthcare provider or pharmacist to avoid potential interactions with current medications.
""",
    
    "chronic_conditions": """
🏥 **Ongoing Medical Care**: If you have a diagnosed digestive condition, please work closely with your gastroenterologist or primary care provider to ensure any new approaches complement your existing treatment plan.
"""
}

# Red flag symptoms requiring immediate professional attention
RED_FLAG_SYMPTOMS = [
    "severe abdominal pain",
    "blood in stool",
    "black, tarry stools",
    "persistent vomiting",
    "high fever",
    "unexplained weight loss",
    "difficulty swallowing",
    "severe dehydration",
    "signs of infection",
    "worsening symptoms despite treatment"
]

# Professional referral recommendations
REFERRAL_RECOMMENDATIONS = {
    "gastroenterologist": [
        "chronic digestive issues",
        "inflammatory bowel disease",
        "persistent abdominal pain",
        "changes in bowel habits",
        "suspected food intolerances"
    ],
    
    "primary_care": [
        "general health assessment",
        "routine digestive concerns",
        "medication management",
        "overall wellness planning",
        "preventive care"
    ],
    
    "registered_dietitian": [
        "personalized nutrition guidance",
        "elimination diets",
        "meal planning for gut health",
        "nutritional deficiencies",
        "specialized dietary needs"
    ],
    
    "mental_health_professional": [
        "stress-related digestive issues",
        "anxiety about health symptoms",
        "gut-brain connection concerns",
        "eating disorder support",
        "mindfulness-based approaches"
    ]
}
