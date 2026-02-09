import random
from models.schemas import VQAQuery, VQAResponse

class InferenceService:
    def predict(self, query: VQAQuery) -> VQAResponse:
        # Mock logic based on keywords
        question_lower = query.question.lower()
        
        if "cat" in question_lower:
            answer = "Yes" if "sitting" in question_lower else "Cat"
            confidence = round(random.uniform(0.85, 0.99), 2)
            candidates = [
                {"text": "Cat", "confidence": 0.92},
                {"text": "Dog", "confidence": 0.05},
                {"text": "Table", "confidence": 0.03}
            ]
            reasoning = [
                "Detected object: Feline features identified",
                "Analyzed context: Indoor domestic setting",
                "Correlated with Label: 'Cat'"
            ]
        elif "color" in question_lower:
            colors = ["Red", "Blue", "Green", "Yellow", "White"]
            answer = random.choice(colors)
            confidence = round(random.uniform(0.70, 0.95), 2)
            candidates = [{"text": c, "confidence": round(random.uniform(0.1, 0.3), 2)} for c in colors[:3]]
            reasoning = [
                "Segmented image into regions",
                "Extracted dominant pixel values",
                f"Matched wavelength to '{answer}'"
            ]
        else:
            answer = "Unknown"
            confidence = 0.45
            candidates = [{"text": "Yes", "confidence": 0.4}, {"text": "No", "confidence": 0.4}]
            reasoning = ["Pattern not recognized in knowledge base"]

        return VQAResponse(
            answer=answer,
            confidence=confidence,
            candidate_answers=candidates,
            reasoning_trace=reasoning
        )

inference_service = InferenceService()
