# IMDB Sentiment Analysis with DeBERTa + MLP

## Model Architecture
- **Feature Extractor**: microsoft/deberta-v3-small
- **Pooling**: Mean + Max concatenation
- **Classifier**: MLP (1024 → 512 → 256 → 1)

## Dataset
- IMDB balanced 10k (5k positive + 5k negative)

## Performance
- Target Accuracy: ≥95%

## Usage
```bash
python src/extract_features.py
python src/train_mlp.py
python src/predict.py "This movie is fantastic!"
CI/CD
GitHub Actions automatically trains and uploads to Hugging Face.
HuggingFace
https://huggingface.co/Ch1kage/Sentiment_analyse-HuggingFace
