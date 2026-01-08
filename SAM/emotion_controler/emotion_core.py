class Emotion:
    def __init__(self, name):
        self.name = name.capitalize()
        self.value = 0


class EmotionVector:
    def __init__(self, base_emotions):
        # Create list of emotions and bind their name to their instance
        self.emotions = {name: Emotion(name) for name in base_emotions}
        self.calm()

    def as_dict(self):
        return {emotion.name: emotion.value for emotion in self.emotions.values()}

    def normalize(self):
        total = sum(emotion.value for emotion in self.emotions.values())
        if total == 0:
            # Avoid division by zero, distribute evenly
            n = len(self.emotions)
            for emotion in self.emotions.values():
                emotion.value = 1 / n
        else:
            for emotion in self.emotions.values():
                emotion.value /= total

    def add_delta(self, deltas: dict):
        """Add changes to emotions and normalize."""
        # E.add_delta({"Happiness": 0.3, "Surprise": 0.1})
        for name, delta in deltas.items():
            if name in self.emotions:
                self.emotions[name].value += delta
                if self.emotions[name].value < 0:
                    self.emotions[name].value = 0
        self.normalize()

    def calm(self):
        """Sets emotions all to a base %"""
        base_delta = 1 / len(self.emotions)
        for emotion in self.emotions.values():
            emotion.value = base_delta
        self.normalize()

    def get_dominant(self):
        max_value = max(e.value for e in self.emotions.values())
        dominant = [e.name for e in self.emotions.values() if e.value == max_value]
        return dominant, max_value


# List of possible Base Emotions
BASE_EMOTIONS = (
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
)

EMOTION_VECTOR = EmotionVector(BASE_EMOTIONS)
print(EMOTION_VECTOR.get_dominant())
print(EMOTION_VECTOR.as_dict())