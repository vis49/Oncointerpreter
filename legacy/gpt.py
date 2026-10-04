from transformers import pipeline
import json
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')


# Specify the path to your JSON file
file_path = 'questions.json'

# Open the file for reading
with open(file_path, 'r') as file:
    # Load the JSON content from the file
    data = json.load(file)
print(generator(data["questions"][13]["text"], do_sample=True, min_length=50))
