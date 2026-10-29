from rag import ask

passed = total = 0
for line in open("eval_questions.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    expected, question = [s.strip() for s in line.split("|", 1)]
    answer, sources = ask(question)
    refused = "NOT_IN_NOTES" in answer
    ok = refused if expected == "REFUSE" else not refused
    passed += ok
    total += 1
    print(f"{'✅' if ok else '❌'} [{expected}] {question}\n   → {answer[:120]}\n")

print(f"Score: {passed}/{total}")