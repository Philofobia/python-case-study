# Wraps text to a certain width
def word_wrap(text, width=87):
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

# Project embeddings using UMAP
def project_embeddings(embeddings, umap_transform):
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings