# Movie Review Sentiment Analysis — Word2Vec and LSTM

An NLP coursework notebook from Le Wagon exploring positive/negative classification of IMDB movie reviews with word embeddings and a recurrent neural network.

## Implemented work

- Loads a 10% subset of the IMDB train and test splits through TensorFlow Datasets.
- Trains Word2Vec embeddings with Gensim and pads review sequences to 200 tokens.
- Builds a Keras classifier with Masking, a 20-unit LSTM, a dense layer and sigmoid output.
- Trains with early stopping and explores transfer learning using pretrained GloVe embeddings.

## Evidence and limitations

The notebook retains historical training outputs for both embedding approaches. The saved final test-evaluation output contains a `NameError` for an earlier variable name (`X_test_pad_2`); the current cell source uses `X_test_pad`. The notebook has not been rerun here, and no verified test accuracy or production readiness is claimed. The baseline and initial-model test evaluation are unfinished. Validation results are not test-set results.

## Running the notebook

Open `sentiment-analysis-with-word2vec.ipynb` in a compatible Jupyter environment with NumPy, TensorFlow/Keras, TensorFlow Datasets and Gensim. Dataset/model downloads require internet access; GloVe embeddings add a separate download. The saved runs used an older TensorFlow/Keras environment, so dependency compatibility should be checked before rerunning. No trained model file is included.

## Provenance

Le Wagon educational exercise with supplied prompts and helper functions alongside the implemented experiment. It is presented as coursework rather than an original research result or commercial deployment.
