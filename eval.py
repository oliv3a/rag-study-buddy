from rag import retriever
# Answer key: (question, expected page)
TESTS = [
    ("Why does a lousy splitting element make the sort slow?", 4),
    ("Which sort works best on data that is almost in order already?", 3),
    ("Which algorithm sorts in n log n guaranteed with no extra memory?", 5),
    ("How do I count steps when a function calls itself on half the input?", 2),
    ("How do we describe an algorithm's growth while ignoring constants?", 1),
    ("Which master theorem case applies to mergesort's recurrence?", 2),
    ("How many comparisons does insertion sort make on already sorted input?", 3),
    ("Why is picking the first element as pivot risky?", 4),
    ("Why is building a heap linear time instead of n log n?", 5),
    ("Which traversal finds the fewest-edge path in an unweighted graph?", 6),
    ("What goes wrong with shortest paths when an edge weight is below zero?", 7),
    ("Why is the O(nC) knapsack solution only pseudo-polynomial?", 8),
    ("Which sorting algorithm needs extra memory proportional to n?", 3),
    ("What is Dijkstra's running time with an adjacency matrix?", 7),
]


PAGE_NAMES = {
    1: "Asymptotic analysis", 2: "Recurrences", 3: "Insertion/Merge sort",
    4: "Quicksort", 5: "Heaps", 6: "BFS/DFS", 7: "Dijkstra", 8: "Dynamic programming",
}

print("=" * 60)
print("🧪 RETRIEVAL TEST: does the bot find the right page?")
print("=" * 60)

hits = 0
misses = []
for i, (question, expected) in enumerate(TESTS, start=1):
    docs = retriever.invoke(question)
    fetched = sorted({d.metadata["page"] for d in docs})
    hit = expected in fetched
    if hit:
        hits += 1
    else:
        misses.append((question, expected, fetched))

    print(f"\nQ{i}. {question}")
    print(f"    Should find: p.{expected} ({PAGE_NAMES[expected]})")
    print(f"    Bot found:   " + ", ".join(f"p.{p} ({PAGE_NAMES[p]})" for p in fetched))
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
        print(f"  • \"{q}\" → needed p.{exp}, got {got}")
print("=" * 60)