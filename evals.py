from dotenv import load_dotenv
load_dotenv()

import json
from rag import answer
from openai import OpenAI

client = OpenAI()

JUDGE = """You grade an answer about the UK Innovator Founder visa.
Compare the CANDIDATE answer to the REFERENCE answer.
Reply with ONLY a JSON object, no other text, no markdown:
{"correct": true or false, "faithful": true or false}
- correct: does the candidate match the key facts in the reference?
- faithful: does the candidate avoid inventing anything beyond the reference?"""

def judge(question, reference, candidate):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": f"Question: {question}\nREFERENCE: {reference}\nCANDIDATE: {candidate}"},
        ],
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
    return json.loads(raw)

def main():
    golden = json.load(open("golden.json"))
    correct = faithful = 0
    for item in golden:
        cand, _ = answer(item["question"])
        verdict = judge(item["question"], item["reference"], cand)
        correct += verdict["correct"]
        faithful += verdict["faithful"]
        mark = "✓" if verdict["correct"] else "✗"
        p
        
    n = len(golden)
    print(f"\nCorrect:  {correct}/{n} ({correct/n:.0%})")
    print(f"Faithful: {faithful}/{n} ({faithful/n:.0%})")

if __name__ == "__main__":
    main()