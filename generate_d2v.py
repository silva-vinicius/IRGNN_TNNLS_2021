# Generate document embeddings for the reviews using Gensim's Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import gzip
import spacy


# -----------------------------------------------------------------------------
# Reads the zipped reviews dataframe
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# -----------------------------------------------------------------------------

# Aggregation function used to groups (concatenate) all reviews of each 
# product into a single row (indexed by the products asin).
def concatReviews(group):
    all_reviews = '\n'.join(list(group['reviewText']))
    
    return pd.Series({
        'reviews_doc': all_reviews
        
    })

# -----------------------------------------------------------------------------
# Preprocessing and Tagging

# We truncate (grouped) reviews that get too big (bigger than max_length)
def truncate_string(value, max_length, suffix=''):
    string_value = str(value)
    string_truncated = string_value[:min(len(string_value), (max_length - len(suffix)))]
    suffix = (suffix if len(string_value) > max_length else '')
    return string_truncated+suffix

# We split each grouped review into lemmas so that gensin can do its thing
def preprocessamento(text):
    return [token.lemma_.lower() for token in nlp(text) 
    		if token.is_alpha and not token.is_stop]

# Transforms a df row into a TaggedDocument (the tag being the products asin)
def tagged_data(row, max_length):
    return TaggedDocument(
    	words=preprocessamento(
    		truncate_string(row['reviews_doc'], max_length)
    	),
    	tags=[row.name]
    )

# -----------------------------------------------------------------------------

# Trains and saves a Doc2Vec model so that IRGNN can load it
def train_and_save_d2v(tagged_reviews, path):

	# Instantiating a Doc2Vec model
	modelo = Doc2Vec(
		vector_size=300,
		min_count=2,
		dm=1,
		dm_concat=1,
		epochs=100
	)

	modelo.build_vocab(tagged_reviews)

	modelo.train(
		tagged_reviews,
		total_examples=modelo.corpus_count,
		epochs=modelo.epochs
	)

	modelo.save(path)


# -----------------------------------------------------------------------------
if __name__ == "__main__":

	# Specifying the maximum length of the grouped reviews
	max_len = 3000000

	
	df = getDF('IRGNN_TNNLS_2021/data/raw/reviews_Video_Games.json.gz')
	df = df.groupby('asin').apply(concatReviews)
	
	# We disable parser and ner in order to save RAM	
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	nlp.max_length = max_len
	
	tagged_reviews = list(df.apply(lambda x: tagged_data(x, max_len), axis=1))
	
	train_and_save_d2v(
		tagged_reviews,
		"IRGNN_TNNLS_2021/data/raw/reviews_Video_Games.d2v"
	)

