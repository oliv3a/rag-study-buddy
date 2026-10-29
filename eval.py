from rag import retriever

# Answer key: (question, acceptable pages)
TESTS = [
    ("What is thrashing?", {38, 39}),
    ("What does the dirty bit do during page replacement?", {16}),
    ("What is the formula for effective access time?", {11}),
    ("What is the difference between global and local replacement?", {35}),
    ("Why does FIFO suffer from Belady's anomaly?", {21}),
    ("Why is LRU not affected by Belady's anomaly?", {24}),
    ("How does the working-set model prevent thrashing?", {46}),
    ("How does the page-fault frequency scheme work?", {48}),
    ("How is LRU implemented with a stack?", {26}),
    ("How does the clock algorithm work?", {30}),
    ("Why does my computer slow to a crawl when too many programs are open?", {38, 39}),
   ("Which pages can be thrown out without saving them to disk first?", {16}),
   ("Why can adding more RAM sometimes make things worse?", {18, 21}),
]

print("=" * 60)
print("🧪 RETRIEVAL TEST: does the bot find the right page?")
print("=" * 60)

hits = 0
misses = []
for i, (question, expected) in enumerate(TESTS, start=1):
    docs = retriever.invoke(question)
    fetched = sorted({d.metadata["page"] for d in docs})
    hit = bool(expected & set(fetched))
    if hit:
        hits += 1
    else:
        misses.append((question, expected, fetched))

    print(f"\nQ{i}. {question}")
    print(f"    Should find: {sorted(expected)}")
    print(f"    Bot found:   {fetched}")
    print("    ✅ Correct page found" if hit else "    ❌ Wrong page: bot looked in the wrong place")

score = 100 * hits // len(TESTS)
print("\n" + "=" * 60)
print(f"📊 RESULT: {hits}/{len(TESTS)} questions found the right page ({score}%)")
if score >= 90:
    print("🟢 Great: the bot almost always looks in the right place.")
elif score >= 70:
    print("🟡 Okay: some questions send the bot to the wrong page.")
else:
    print("🔴 Weak: the bot often looks in the wrong place.")
if misses:
    print("\nQuestions to look at:")
    for q, exp, got in misses:
        print(f"  • \"{q}\" → needed {sorted(exp)}, got {got}")
print("=" * 60)