import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    n_gram_list = []
    sequence.insert(0,'START')
    
    # adding 'START' values to our sequence
    if (n > 2):
        for i_temp in range(n - 2):
            sequence.insert(0,'START')
    
    # adding 'STOP' value at the end of our sequence
    sequence.append('STOP')
    
    # main iteration through sequence
    for i in range(len(sequence) - n + 1):
        # temporary n_gram to be reset after each string in sequence
        temp_gram = ()
        
        for j in range(n):
            
            temp_tup = (sequence[i+j],)
            temp_gram = temp_gram + temp_tup
            
        n_gram_list.append(temp_gram)
        
    return n_gram_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        
                
        for sentence in corpus:
            
            unigram_list = get_ngrams(sentence, 1)
            bigram_list = get_ngrams(sentence, 2)
            trigram_list = get_ngrams(sentence, 3)
            
            # bigram dictionary builder
            for bigram in bigram_list:
                
                # if key exists, add 1 to its count
                if (bigram in self.bigramcounts):
                    self.bigramcounts[bigram] += 1
                    
                # else, set its count to 1  
                else:
                    self.bigramcounts[bigram] = 1
      
            # unigram dictionary builder
            for unigram in unigram_list:
                
                # if key exists, add 1 to its count
                if (unigram in self.unigramcounts):
                    self.unigramcounts[unigram] += 1
                    
                # else, set its count to 1  
                else:
                    self.unigramcounts[unigram] = 1
          
            # trigram dictionary builder        
            for trigram in trigram_list:
                
                # if key exists, add 1 to its count
                if (trigram in self.trigramcounts):
                    self.trigramcounts[trigram] += 1
                    
                # else, set its count to 1  
                else:
                    self.trigramcounts[trigram] = 1
                     
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        total_grams = sum(self.trigramcounts.values())
        
        #for key in self.trigramcounts:
            
            #count = self.trigramcounts[key]
            #total_grams += count
        
        if (trigram in self.trigramcounts):
            trigram_count = self.trigramcounts[trigram]
        
        else:
            trigram_count = 0
            
        raw_prob = trigram_count / total_grams
        
        return raw_prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        total_grams = sum(self.bigramcounts.values())
        
        # finds total num bigrams
        #for key in self.bigramcounts:
            
            #count = self.bigramcounts[key]
            #total_grams += count
        
        # finds bigram count
        if (bigram in self.bigramcounts):           
            bigram_count = self.bigramcounts[bigram]
            
        else:
            bigram_count = 0
        
        raw_prob = bigram_count / total_grams
        
        return raw_prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        total_grams = sum(self.unigramcounts.values())
        
        #for key in self.unigramcounts:
            
            #count = self.unigramcounts[key]
            #total_grams += count
        
        if (unigram in self.unigramcounts):
            unigram_count = self.unigramcounts[unigram]
        
        else:
            unigram_count = 0
            
        raw_prob = unigram_count / total_grams

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return raw_prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        smoothed_prob = lambda1*(self.raw_unigram_probability(trigram[2])) + lambda2*(self.raw_bigram_probability(trigram[1:2]))+ lambda3*(self.raw_trigram_probability(trigram))
        
        return smoothed_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams_list = get_ngrams(sentence, 3)
        
        log_probs_list = []
        
        for i in range(len(trigrams_list)):
            
            prob = math.log2(self.smoothed_trigram_probability(trigrams_list[i]))
            log_probs_list.append(prob)
        
        log_prob = sum(log_probs_list)
        
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_prob = 0
        total_words = 0
        
        for sequence in corpus:
            log_prob = self.sentence_logprob(sequence)
            sum_prob += log_prob
            total_words += len(sequence)
        
        l = sum_prob / total_words
        perplexity_val = pow(2,-1*l)
        
        return perplexity_val 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            if (pp_1 < pp_2):
                correct += 1
                
            total += 1
            
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            
            if (pp_2 < pp_1):
                correct += 1
                
            total += 1
        
        accuracy = correct / total
        
        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)    
    print("Perplexity of the test corpus:", pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("/Users/jay/Home/Courses/NLP/hw1/hw1_data/ets_toefl_data/train_high.txt",
                                   "/Users/jay/Home/Courses/NLP/hw1/hw1_data/ets_toefl_data/train_low.txt",
                                   "/Users/jay/Home/Courses/NLP/hw1/hw1_data/ets_toefl_data/test_high",
                                   "/Users/jay/Home/Courses/NLP/hw1/hw1_data/ets_toefl_data/test_low")
    
    print("Essay scoring accuracy:", acc)