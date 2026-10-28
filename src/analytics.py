from collections import Counter
import re

STOPWORDS = set("""
a an the and or but if then than so to of in on for with by from at as is are was were be been being
it its this that these those we they you your our their can could should would may might will shall do does
did have has had not no yes into about over under between across after before during also such more most
""".split())

def analyze_corpus(documents):
    per_doc = []
    counter = Counter()
    total = 0
    for doc in documents:
        words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", doc["text"].lower())
        total += len(words)
        counter.update(w for w in words if w not in STOPWORDS)
        per_doc.append({"Document": doc["name"], "Words": len(words), "Characters": len(doc["text"])})
    return {
        "total_words": total,
        "unique_terms": len(counter),
        "avg_words": total / max(1, len(documents)),
        "top_terms": counter.most_common(20),
        "documents": per_doc,
    }
