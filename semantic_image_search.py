"""Semantic image search using CLIP - like Roboflow's AI search.

Search your screenshots using natural language:
- "enemy ships"
- "explosions and lasers"
- "UI elements in top right"

Usage:
    python semantic_image_search.py --query "spaceships with lasers" --dir "F:/dev/bot/yolo/training_screenshots"
"""
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple
from PIL import Image


def load_clip():
    """Load CLIP model for image-text matching."""
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"CLIP loaded on {device}")
        return model, processor, device
    except ImportError:
        print("CLIP not installed. Install with:")
        print("  pip install transformers torch pillow")
        return None, None, None


def encode_images(image_paths: List[Path], model, processor, device) -> np.ndarray:
    """Encode images to embeddings."""
    import torch

    embeddings = []

    print(f"\nEncoding {len(image_paths)} images...")
    for img_path in image_paths:
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Process and encode
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.append(image_features.cpu().numpy()[0])

    return np.array(embeddings)


def encode_text(text: str, model, processor, device) -> np.ndarray:
    """Encode text query to embedding."""
    import torch

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()[0]


def search_images(query: str, image_dir: str, top_k: int = 10):
    """
    Search images using natural language query.

    Args:
        query: Natural language search query
        image_dir: Directory with images
        top_k: Number of results to return
    """
    # Load CLIP
    model, processor, device = load_clip()
    if model is None:
        return

    # Get images
    image_path = Path(image_dir)
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Encode all images
    image_embeddings = encode_images(image_files, model, processor, device)

    # Encode query
    print(f"\nSearching for: '{query}'")
    query_embedding = encode_text(query, model, processor, device)

    # Calculate similarities (cosine similarity = dot product of normalized vectors)
    similarities = np.dot(image_embeddings, query_embedding)

    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\n{'='*60}")
    print(f"Top {top_k} results:")
    print(f"{'='*60}")

    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        img_path = image_files[idx]
        print(f"{i+1}. {img_path.name} (score: {score:.3f})")

    # Show results
    print(f"\nShowing results...")
    show_results(image_files, top_indices, similarities)


def show_results(image_files: List[Path], indices: np.ndarray, scores: np.ndarray):
    """Display search results in a grid."""
    import matplotlib.pyplot as plt

    n_results = len(indices)
    cols = min(5, n_results)
    rows = (n_results + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        ax = axes[row][col]

        # Load and display image
        img = cv2.imread(str(image_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Score: {scores[idx]:.3f}", fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for i in range(n_results, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Semantic image search with CLIP')
    parser.add_argument('--query', type=str, required=True,
                       help='Natural language search query')
    parser.add_argument('--dir', type=str, default='F:/dev/bot/yolo/training_screenshots',
                       help='Directory with images to search')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return')

    args = parser.parse_args()

    search_images(args.query, args.dir, args.top_k)


if __name__ == '__main__':
    main()
