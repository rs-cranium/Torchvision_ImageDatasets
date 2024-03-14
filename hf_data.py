from datasets import load_dataset

# Load IMDb dataset
imdb_dataset = load_dataset('imdb')

# Print dataset info
print(imdb_dataset)

# Access a sample review and label
sample_review = imdb_dataset['train'][0]
review_text = sample_review['text']
label = sample_review['label']

print("Sample Review:")
print(review_text)
print("Label:", label)
