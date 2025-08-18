# import time
# import uuid
# from pathlib import Path

# from mosaic import Mosaic


# class DummyInferenceClient:
# 	"""Fast, lightweight encoder that returns 128-dim embeddings.
# 	This avoids loading heavy models and keeps the demo quick."""

# 	def encode_image(self, image):
# 		# Return a single 128-dim embedding (list of floats)
# 		return [[0.0 for _ in range(128)]]

# 	def encode_query(self, query: str):
# 		return [[0.0 for _ in range(128)]]


# def main():
# 	# Absolute path to the docs directory
# 	docs_dir = Path("/home/abdullah/Projects/mosaic/docs")
# 	assert docs_dir.is_dir(), f"Docs directory not found: {docs_dir}"

# 	# Gather PDFs
# 	pdf_files = sorted(docs_dir.glob("*.pdf"))
# 	assert pdf_files, f"No PDF files found in {docs_dir}"

# 	# Initialize Mosaic with in-memory Qdrant and dummy inference
# 	mosaic = Mosaic(
# 		collection_name="priority_demo",
# 		inference_client=DummyInferenceClient(),
# 	)

# 	# Assign descending priorities so later files have higher priority
# 	# Example: for 6 files -> priorities [1,2,3,4,5,6]
# 	priorities = list(range(1, len(pdf_files) + 1))

# 	# Build a plan and print it for clarity
# 	plan = [
# 		{
# 			"file": f,
# 			"priority": p,
# 			"file_id": f"demo-{f.stem}-{uuid.uuid4().hex[:8]}",
# 		}
# 		for f, p in zip(pdf_files, priorities)
# 	]

# 	print("\nEnqueue order (lower priority number first):")
# 	for item in plan:
# 		print(f" - {item['file'].name} -> priority={item['priority']}")

# 	expected_processing = sorted(plan, key=lambda x: x["priority"], reverse=True)
# 	print("\nExpected processing order (by priority desc):")
# 	for item in expected_processing:
# 		print(f" - {item['file'].name} -> priority={item['priority']}")

# 	# Enqueue all files
# 	for item in plan:
# 		mosaic.index_file(
# 			file_id=item["file_id"],
# 			file_path=item["file"],
# 			metadata={"priority": item["priority"], "source": "priority_demo"},
# 			store_img_bs64=False,
# 			max_image_dims=(256, 256),  # keep small for speed
# 			priority=item["priority"],
# 		)

# 	# Block until all enqueued tasks are done
# 	mosaic.wait_for_indexing()

# 	print("\nDone. Compare the '[Queue] Dequeued' lines above with the expected order.")


# if __name__ == "__main__":
# 	main() 