from rag import vector_db

data = vector_db.get()
print("Chunks:", len(data["ids"]))
for meta, text in list(zip(data["metadatas"], data["documents"]))[:5]:
    print(f"\np.{meta['page']} [{meta['topic']}]")
    print(text[:150])