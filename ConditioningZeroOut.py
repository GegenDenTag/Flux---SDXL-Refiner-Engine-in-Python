class ConditioningZeroOut:
    """
    Utility class for empty conditioning inputs.
    Used when prompts are empty.
    """
    def __init__(self):
        self.text = ""
    
    def __str__(self):
        return ""
    
    def __bool__(self):
        return False


# AND 

def _process_prompt(self, prompt: str) -> str:
    """Returns ConditioningZeroOut for empty prompts"""
    if not prompt or prompt.strip() == "":
        return ConditioningZeroOut()
    return prompt.strip()



