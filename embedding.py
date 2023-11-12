# embedding vector
# https://huggingface.co/intfloat/e5-small-v2
# pip install sentence_transformers~=2.2.2

# loops to pass in prediction and answer--increase length of the string passed in on each --max length of one will be
# reached faster than the other so make sure it does max length after one gets longer
# how to compare embeddings?
# turn embeddings into a score
"""
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/e5-small-v2')
input_texts = [
    'query: how much protein should a female eat', 
    'query: summit define',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
embeddings = model.encode(input_texts, normalize_embeddings=True) 
*embeddings need to be turned into score 

query= PRED
passage= ANSWER (not cut) 20 tokens
check with cutting answer at different points (range 1-20), save the one that matches the best -> eventually have a peak graph

"""


"""
COS:
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embedding1, embedding2)

EUC:
from scipy.spatial import distance

euclidean_distance = distance.euclidean(embedding1, embedding2)
"""

#keep total GPT response, add to column 
#total GPT response, total answer (keep as strings)
#TOKENIZED: calc LD with cut GPT response/ answer - print to LD csv file
#STRING: calc EMB with total GPT response (multiple cuts LOOP)/answer- print to EMB csv file 