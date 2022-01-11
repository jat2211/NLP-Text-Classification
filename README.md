# NLP-Text-Classification
## Description
Project that builds a trigram language model. The main component of the language model is implemented in trigram_model.py. The model can classify the "grammatical" level of a piece of text based on a dataset of essays rated as low, medium, or high level.

First, I implemented the function get_ngrams, which takes a list of strings and an integer n as input, and returns padded n-grams over the list of strings. I then implemented the method count_ngrams that counts the occurrence frequencies for ngrams in the corpus. Next, I implemented methods that return unsmoothed probabilities computed from the trigram, bigram, and unigram counts. I then made a linear interpolation method to return the smoothed trigram probability using n-gram probabilities (1 <= n <= 3). Lastly, I computed the perplexity of the model on an entire corpus.

The final step of the model evaluates the now complete model to make predictions on how essays should be classified using the perplexity to make the prediction.

## Skills Used
- Python
- data structures
- **broader domains: natural language processing, machine learning**
