# Wraps text to a certain width
def word_wrap(text, width=87):
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])