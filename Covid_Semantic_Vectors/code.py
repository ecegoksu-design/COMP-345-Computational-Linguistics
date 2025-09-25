from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
from gensim.parsing import preprocess_documents
from sklearn.utils.extmath import randomized_svd
import re
from collections import defaultdict, Counter
import os
import gensim.downloader as api
import nltk


def top_k_unigrams(tweets, stop_words, k):
    frequency = Counter()

    for line in tweets:
        for word in line.split():
            if not re.match(r'^[a-z#]', word):
                continue

            lower_word = word.lower()

            if lower_word in stop_words:
                continue

            frequency[lower_word] += 1

    if k == -1:
        return dict(frequency)
    return dict(frequency.most_common(k))

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    context_counter = Counter()
    stops = set(stop_words)
    frequent = set(frequent_unigrams)

    for tweet in tweets:
        words = tweet.split()

        for i, word in enumerate(words):
            if re.match(r'^[a-z#]', word):
                lower_word = word.lower()

                if lower_word not in stops and lower_word in frequent:
                    start = max(0, i - context_size)
                    end = min(len(words), i + context_size + 1)

                    for j in range(start, end):
                        if j != i:
                            context_word = words[j]
                            context_counter[(context_word, lower_word)] += 1

    return dict(context_counter)

def pmi(word1, word2, unigram_counter, context_counter):
    total_unigrams = sum(unigram_counter.values())

    count_word1 = unigram_counter.get(word1, 0) or 1
    count_word2 = unigram_counter.get(word2, 0) or 1
    count_word1_word2 = context_counter.get((word1, word2), 0) or 1

    prob_word1 = count_word1 / total_unigrams
    prob_word2 = count_word2 / total_unigrams
    prob_word1_word2 = count_word1_word2 / total_unigrams

    pmi_score = np.log2(prob_word1_word2 / (prob_word1 * prob_word2))

    return pmi_score

def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    word_vector = {}

    for word2 in frequent_unigrams:
        if word1 == word2:
            continue

        if (word1, word2) in context_counter:
            word_vector[word2] = pmi(word1, word2, unigram_counter, context_counter)
        else:
            word_vector[word2] = 0.0

    return word_vector

def get_top_k_dimensions(word1_vector, k):
    top_k = dict(sorted(word1_vector.items(), key=lambda x: x[1], reverse=True)[:k])
    return top_k

def get_cosine_similarity(word1_vector, word2_vector):
    keys = set(word1_vector.keys()) | set(word2_vector.keys())

    vector_1 = np.array([word1_vector.get(key, 0.0) for key in keys])
    vector_2 = np.array([word2_vector.get(key, 0.0) for key in keys])

    dot_product = np.dot(vector_1, vector_2)
    norm1 = np.linalg.norm(vector_1)
    norm2 = np.linalg.norm(vector_2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

def get_most_similar(word2vec, word, k):
    if word not in word2vec:
        return []

    similar_words = word2vec.most_similar(word, topn=k)

    return similar_words

def word_analogy(word2vec, word1, word2, word3):
    if word1 not in word2vec or word2 not in word2vec or word3 not in word2vec:
        return None

    word4, score = word2vec.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0]
    return word4, score


def cos_sim(A, B):
    arrayA = np.array(A)
    arrayB = np.array(B)

    normA = np.linalg.norm(arrayA)
    normB = np.linalg.norm(arrayB)

    if normA == 0 or normB == 0:
        return 0.0

    return np.dot(arrayA, arrayB) / (normA * normB)

def get_cos_sim_different_models(word, model1, model2, cos_sim_function):
    if word in model1.wv and word in model2.wv:
        return cos_sim_function(model1.wv[word], model2.wv[word])
    return 0.0

def get_average_cos_sim(word, neighbors, model, cos_sim_function):
    if word not in model.wv:
        return 0.0

    word_vector = model.wv[word]

    cosine_similarities = [
        cos_sim_function(word_vector, model.wv[neighbor])
        for neighbor in neighbors if neighbor in model.wv
    ]

    return np.mean(cosine_similarities) if cosine_similarities else 0.0

def create_tfidf_matrix(documents, stopwords):
    processed_docs = []
    for document in documents:
        processed = [word.lower() for word in document if word.lower() not in stopwords and word.isalnum()]
        processed_docs.append(processed)

    vocabulary = sorted(set(word for doc in processed_docs for word in doc))

    default = defaultdict(int)
    for term in vocabulary:
        default[term] = sum(1 for document in processed_docs if term in document)

    tfidf_matrix = np.zeros((len(processed_docs), len(vocabulary)))
    for i, document in enumerate(processed_docs):
        tf = Counter(document)
        for j, term in enumerate(vocabulary):
            tf_val = tf.get(term, 0)
            idf_val = np.log10(1 + len(processed_docs) / (1 + default[term]))
            tfidf_matrix[i, j] = tf_val * idf_val

    return tfidf_matrix, vocabulary


def get_idf_values(documents, stopwords):
    preprocessed_documents = []
    for document in documents:
        preprocessed_document = [word.lower() for word in document if word.lower() not in stopwords and word.isalnum()]
        preprocessed_documents.append(preprocessed_document)

    vocab = sorted(set(word for document in preprocessed_documents for word in document))

    document_frequencies = {}
    for word in vocab:
        document_frequencies[word] = sum(1 for doc in preprocessed_documents if word in doc)

    idf_values = {}
    num_documents = len(preprocessed_documents)
    for word in vocab:
        idf_values[word] = np.log10(1 + num_documents / (1 + document_frequencies[word]))

    return idf_values

def calculate_sparsity(tfidf_matrix):
    all = tfidf_matrix.size
    zero = np.count_nonzero(tfidf_matrix == 0)
    sparsity = zero / all
    return sparsity


def extract_salient_words(VT, vocabulary, k):
    salient_words = {}
    for dimension, component in enumerate(VT):
        index = np.argsort(-np.abs(component))[:k]

        sorted_index = index[::-1]

        salient_words[dimension] = [vocabulary[idx] for idx in sorted_index]

    return salient_words

def get_similar_documents(U, Sigma, VT, doc_index, k):
    scaled = U * Sigma

    vector = scaled[doc_index]
    norm = np.linalg.norm(vector)

    if norm == 0:
        return []

    cos_similarity = scaled @ vector / (np.linalg.norm(scaled, axis=1) * norm)
    cos_similarity = np.nan_to_num(cos_similarity)

    similar_docs = np.argsort(-cos_similarity)
    similar_docs = [index for index in similar_docs if index != doc_index]

    return similar_docs[:k]


def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):

    elements = [element.lower() for element in query if element.isalnum() and element.lower() in vocabulary]

    tf = Counter(elements)

    vector = np.zeros(len(vocabulary))
    index = {word: idx for idx, word in enumerate(vocabulary)}
    for i, freq in tf.items():
        idx = index[i]
        vector[idx] = freq * idf_values.get(i, 0.0)

    projection = vector @ VT.T

    scaled = U * Sigma

    norm1 = np.linalg.norm(projection)
    if norm1 == 0:
        return []

    norm2 = np.linalg.norm(scaled, axis=1)
    norm2[norm2 == 0] = 1e-10

    cos_similarity = (scaled @ projection) / (norm2 * norm1)
    cos_similarity = np.nan_to_num(cos_similarity, nan=0.0)

    top_index = np.argsort(cos_similarity)[::-1][:k]

    return top_index.tolist()


if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt') as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    
    ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    print(sample_output.most_common(10))
    """
    [(('the', 'pandemic'), 19811),
    (('a', 'pandemic'), 16615),
    (('a', 'mask'), 14353),
    (('a', 'wear'), 11017),
    (('wear', 'mask'), 10628),
    (('mask', 'wear'), 10628),
    (('do', 'n’t'), 10237),
    (('during', 'pandemic'), 8127),
    (('the', 'covid'), 7630),
    (('to', 'go'), 7527)]
    """
    ### END OF REFERENCE OUTPUT
    
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)

    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817


    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    """Word2Vec for Meaning Change"""

    # Comparing 40-60 year olds in the 1910s and 40-60 year olds in the 2000s
    model_t1 = Word2Vec.load('data/1910s_50yos.model')
    model_t2 = Word2Vec.load('data/2000s_50yos.model')

    # Cosine similarity function for vector inputs
    vector_1 = np.array([1,2,3,4])
    vector_2 = np.array([3,5,4,2])
    cos_similarity = cos_sim(vector_1, vector_2)
    print(cos_similarity)
    # 0.8198915917499229

    # Similarity between embeddings of the same word from different times
    major_cos_similarity = get_cos_sim_different_models("major", model_t1, model_t2, cos_sim)
    print(major_cos_similarity)
    # 0.19302374124526978

    # Average cosine similarity to neighborhood of words
    neighbors_old = ['brigadier', 'colonel', 'lieutenant', 'brevet', 'outrank']
    neighbors_new = ['significant', 'key', 'big', 'biggest', 'huge']
    print(get_average_cos_sim("major", neighbors_old, model_t1, cos_sim))
    # 0.6957747220993042
    print(get_average_cos_sim("major", neighbors_new, model_t1, cos_sim))
    # 0.27042335271835327
    print(get_average_cos_sim("major", neighbors_old, model_t2, cos_sim))
    # 0.2626224756240845
    print(get_average_cos_sim("major", neighbors_new, model_t2, cos_sim))
    # 0.6279034614562988

    ### The takeaway -- When comparing word embeddings from 40-60 year olds in the 1910s and 2000s,
    ###                 (i) cosine similarity to the neighborhood of words related to military ranks goes down;
    ###                 (ii) cosine similarity to the neighborhood of words related to significance goes up.


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
