#!/usr/bin/env python
from __future__ import division

import re
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import stopwords
import enchant
from nltk.tokenize import word_tokenize
import textmining
import language_check
from stemming.porter2 import stem
import math
import numpy as np
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

filename1="D:\\zzzz\\nature.txt"
filename2="D:\\zzzz\\cat1.txt"
filename3="D:\\zzzz\\school1.txt"
filename=filename1
documents = open(filename).read()
topic = nltk.word_tokenize(documents)
topic_name1=topic[0]
topic_name=topic_name1.lower()
print("\nTOPIC OF THE ESSAY IS:")
print(topic_name)





GRAMMAR=25
SPELL=20
TOPIC=30.0
SEMANTICS=20
LENGTH=5






##########################################LENGTH#####################################


feed = open(filename, "r+")

num_words = 0

for wordcount in feed.read().split():
    # print(wordcount)
    num_words = num_words + 1
print("TOTAL NUMBER OF WORDS IN THE ESSAY:")
print(num_words)

if num_words > 120 or num_words < 80:
    LENGTH=LENGTH-5














##########################################GRAMMAR#####################################




tool = language_check.LanguageTool('en-UK')
count1 = 0
documents = open(filename).read()
tokens = nltk.sent_tokenize(documents)
num_sen=len(tokens)
for i in tokens:
    matches = tool.check(i)
    print(i)
    print(len(matches))
    count1 = count1 + len(matches)
print("\nTHE NUMBER OF GRAMMATICAL ERRORS:")
print(count1)

t1=float(count1)/float(num_words) * float(100)
#print(t1)
GRAMMAR=float(GRAMMAR)-t1
#GRAMMAR=GRAMMAR-(count1*1)












###########################SEMANTICS########################################

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0


def get_best_synset_pair(word_1, word_2):
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.path_similarity(synset_1, synset_2)
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2
        return best_pair


def length_dist(synset_1, synset_2):
    l_dist = sys.maxint
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    h_dist = sys.maxint
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if hypernyms_1.has_key(lcs_candidate):
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if hypernyms_2.has_key(lcs_candidate):
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /
            (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))


def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) *
            hierarchy_dist(synset_pair[0], synset_pair[1]))


def most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = word_similarity(word, ref_word)
        if sim > max_sim:
            max_sim = sim
            sim_word = ref_word
    return sim_word, max_sim


def info_content(lookup_word):
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if not brown_freqs.has_key(word):
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))


def semantic_vector(words, joint_words, info_content_norm):
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec


def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


def word_order_vector(words, joint_words, windex):
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec


def word_order_similarity(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))


def similarity(sentence_1, sentence_2, info_content_norm):
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
           (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)


documents = open(filename).read()
tokens=nltk.sent_tokenize(documents)
sentence_pairs=[[]]
#print("Length")
#print(len(sentence_pairs))
i = 0

while i < len(tokens) - 1:
    # print(tokens[i+1])
    # sent1[i]=[tokens[i],tokens[i+1]]

 #   print(len(sentence_pairs))

    str1 = tokens[i]
    str2 = tokens[i + 1]
   #print("tok")
    #print([str1, str2])
    sentence_pairs.append([str1, str2])
    #print("Sentpair")
   # print(sentence_pairs)
   # print("i")
    i = i + 1

sentence_pairs.pop(0)

print("\nSEMANTIC SIMILARITY BETWEEN PAIR OF SENTENCES")
for sent_pair in sentence_pairs:
    score=similarity(sent_pair[0], sent_pair[1],False)
   # print(score)
    if score < 0.4 and score >= 0.35:
        SEMANTICS=SEMANTICS-0.25
    if score < 0.35 and score >= 0.3:
        SEMANTICS=SEMANTICS-0.5
    if score > 0.7:
        SEMANTICS=SEMANTICS-1
    if score < 0.3:
        SEMANTICS=SEMANTICS-1

    print "\t%s\t%s\t%.3f\t" % (sent_pair[0], sent_pair[1],
                                similarity(sent_pair[0], sent_pair[1], False))

#############################################################################################################################







##########################################SPELLSUGGEST######################################



documents = open(filename).read()
doc_complete = nltk.sent_tokenize(documents)

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('D:/words.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set( deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
correct_tokens=[]


def clean1(s):
  #  s='This is some \\u03c0 text that has to be cleaned\\u2026! it\\u0027s annoying!'
    return s.decode('unicode_escape').encode('ascii','ignore')

doc_clean1 = [clean1(doc) for doc in doc_complete]
#print(doc_clean1)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    correct_tokens = []
    for ws in doc.lower().split():
        #print(ws)
        w = ws.lower()
        correct_tokens.append(correction(w))
      #  print("Corrected tokens")
#        print(correct_tokens)

    stop_free = " ".join([i for i in correct_tokens if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in doc_clean1]
#print("CLEANED DOCUMENT")
#print(doc_clean)


# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('D:/deerwester.dict')
#print(dictionary)
#print(dictionary.token2id)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#print(doc_term_matrix)

Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=5000)
#print(ldamodel)
test=ldamodel.print_topics(num_topics=1, num_words=1)
print("TOPIC WEIGHTAGE")
print(test)


#res=[i for i, v in enumerate(test) if filter(None, re.split(r'\W|\d', v[1])) == "school"]
#print(res)

result=[]
result=[x[1] for x in test]

for i in result:
    #test1=filter(None, re.split(r'\W|\d', i))
    #print(test1)
    result1=i.encode('ascii', 'ignore')
   # print("Result1")
   # print(result1)
    message = result1


    str = re.findall(r"[A-Za-z0-9.]+|\S", message)
    #print("Num val")
    print(str[0])
    val=str[0]
    val1=float(val)
    if str[3] == topic_name:
        TOPIC=30
    else:
        TOPIC=0
    #string=str[3]
    #print "Weightage of \t%s topic\t is \t%.3f\t" % (string, val)
    TOPIC1=float(TOPIC)
    val2=float('10.0')
    #print("String")
    TOPIC2=TOPIC1*val1*val2
    print(TOPIC2)
    #print(val)
    #print(topic_name)
    #print("Condition")

    test1=filter(None, re.split(r'\W|\d', result1))
    #print("Test1")
    #print(test1[0])
#TOPIC=TOPIC*val
#print(SPELL,GRAMMAR,LENGTH,SEMANTICS,TOPIC)


##################################SPELLING###########################################



d = enchant.Dict("en_UK")
##example="This is the example"
documents = open(filename).read()

count = 0
tokens = nltk.sent_tokenize(documents)
# termdocumentmatrix_example()
for ws in tokens:
    # w=ws.lower()
    test = filter(None, re.split(r'\W|\d', ws))
    for i in test:
        if not (d.check(i)):
            print("\nMISSPELLED WORD IN THE ESSAY:")
            print(i)
            count = count + 1
            print("\nCORRECT SPELLING:")
            print(correction(i))
print("\nTHE NUMBER OF MISSPELLED WORDS IN THE ESSAY \t")
print(count)
#SPELL=SPELL-(count*0.5)
t4=float(count)/float(num_words) * float(100)
#print(t4)
SPELL=float(SPELL)-t4



print "SPELLING:%f/20\n" % float(SPELL)
print "GRAMMAR:%d/25\n" % float(GRAMMAR)
print "LENGTH:%d/5\n" % (LENGTH)
print "SEMANTICS:%f/20\n" % float(SEMANTICS)
print "TOPIC:%f/30\n" % float(TOPIC2)

TOTAL=SPELL+GRAMMAR+TOPIC2+SEMANTICS+LENGTH

print("TOTAL SCORE IS:")
print(float(TOTAL)/float(10))