import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("DataFrame.csv")
df['Sentiment'] = df['Polarity'].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
# Create and fit the LE
le = LabelEncoder()
le.fit(df["Sentiment"])
print(f"Label mapping: {list(le.classes_)}")

# Load the pipeline
pipe = pipeline("text-classification", model="multiclass-sentiment-model", tokenizer="multiclass-sentiment-model")

# examples
test_texts = [
    "This product is amazing!"
]

for text in test_texts:
    #prediction from pipeline
    output = pipe(text)
    if "label" in output[0]:
        label_id_str = output[0]["label"]

        # Check "LABEL_X" format
        if label_id_str.startswith("LABEL_"):
            label_id = int(label_id_str.split("_")[-1])
        else:
            try:
                label_id = int(label_id_str)
            except ValueError:
                print(f"Text: '{text}'")
                print(f"Predicted sentiment: {label_id_str}")
                print(f"Confidence: {output[0]['score']:.4f}")
                print()
                continue

        # Decode numeric label to original label
        predicted_label = le.classes_[label_id]
        print(f"Text: '{text}'")
        print(f"Predicted sentiment: {predicted_label}")
        print(f"Confidence: {output[0]['score']:.4f}")
        print()
    else:
        print(f"Unexpected output format: {output}")