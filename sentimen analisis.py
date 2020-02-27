#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
from sklearn.utils import resample


# # Memanggil Data

# In[2]:


train_data = pd.read_csv("D:\\PELATIHAN SURVEY\\train.csv", encoding='ISO-8859-1')


# In[3]:


train_data


# In[4]:


data_down = train_data[train_data.Sentiment==1]
data_down2 = train_data[train_data.Sentiment==0]


# In[5]:


data_down


# # Downsampling Data

# In[6]:


data_downsample = resample(data_down,
                    replace=True,     # sample with replacement
                    n_samples=43532,    # to match majority class
                    random_state=123) # reproducible results
#fixed=pd.concat([data_downsample],[data_down2])


#data_downsample2 = resample(data_down2,
                    #replace=True,     # sample with replacement
                    #n_samples=43532,    # to match majority class
                    #random_state=123) # reproducible results
fixed=pd.concat([data_downsample,data_down2])


# In[ ]:





# In[7]:


fixed.Sentiment.value_counts()


# # Split Data Train and Test

# In[8]:


train = fixed.sample(frac=7/10, replace=False)
test = fixed.sample(frac=3/10, replace=False)
print (train)


# # Visualize the Tweet

# In[9]:


rand_indexs = np.random.randint(1,len(train),50).tolist()
train["SentimentText"][rand_indexs]


# # Emoticons

# In[10]:


import re
tweets_text = train.SentimentText.str.cat()
emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text))
emos_count = []
for emo in emos:
    emos_count.append((tweets_text.count(emo), emo))
sorted(emos_count,reverse=True)


# In[11]:


HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(HAPPY_EMO, tweets_text)))
print("Sad emoticons:", set(re.findall(SAD_EMO, tweets_text)))


# # Most Used Words

# In[12]:


import nltk
from nltk.tokenize import word_tokenize

# Uncomment this line if you haven't downloaded punkt before
# or just run it as it is and uncomment it if you got an error.
#nltk.download('punkt')
def most_used_words(text):
    tokens = word_tokenize(text)
    frequency_dist = nltk.FreqDist(tokens)
    print("There is %d different words" % len(set(tokens)))
    return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)


# In[13]:


import nltk
nltk.download('punkt')


# In[14]:


nltk.download('stopwords')


# In[15]:


nltk.download('wordnet')


# In[16]:


most_used_words(train.SentimentText.str.cat())[:100]


# # Stopwords

# In[17]:


from nltk.corpus import stopwords

#nltk.download("stopwords")

mw = most_used_words(train.SentimentText.str.cat())
most_words = []
for w in mw:
    if len(most_words) == 1000:
        break
    if w in stopwords.words("english"):
        continue
    else:
        most_words.append(w)


# In[18]:


sorted(most_words)


# # Stemming

# In[19]:


# I'm defining this function to use it in the 
# Data Preparation Phase
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
def stem_tokenize(text):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(token) for token in word_tokenize(text.lower())]

def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text.lower())]


# # Prepare the Data

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


# In[22]:


# We need to do some preprocessing of the tweets.
# We will delete useless strings (like @, # ...)
# because we think that they will not help
# in determining if the person is Happy/Sad

class TextPreProc(BaseEstimator,TransformerMixin):
    def __init__(self, use_mention=False):
        self.use_mention = use_mention
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # We can choose between keeping the mentions
        # or deleting them
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(HAPPY_EMO, " happyemoticons ")
        X = X.str.replace(SAD_EMO, " sademoticons ")
        X = X.str.lower()
        return X


# # Select a Model

# In[23]:


# This is the pipeline that will transform our tweets to something eatable.
# You can see that we are using our previously defined stemmer, it will
# take care of the stemming process.
# For stop words, we let the inverse document frequency do the job
from sklearn.model_selection import train_test_split

sentiments = train['Sentiment']
tweets = train['SentimentText']

# I get those parameters from the 'Fine tune the model' part
vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))
pipeline = Pipeline([
    ('text_pre_processing', TextPreProc(use_mention=True)),
    ('vectorizer', vectorizer),
])

# Let's split our data into learning set and testing set
# This process is done to test the efficency of our model at the end.
# You shouldn't look at the test data only after choosing the final model
learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentiments, test_size=0.05)

# This will tranform our learning data from simple text to vector
# by going through the preprocessing tranformer.
learning_data = pipeline.fit_transform(learn_data)


# In[24]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

lr = LogisticRegression()
bnb = BernoulliNB()
mnb = MultinomialNB()

models = {
    'logitic regression': lr,
    'bernoulliNB': bnb,
    'multinomialNB': mnb,
}

for model in models.keys():
    scores = cross_val_score(models[model], learning_data, sentiments_learning, scoring="f1", cv=10)
    print("===", model, "===")
    print("scores = ", scores)
    print("mean = ", scores.mean())
    print("variance = ", scores.var())
    models[model].fit(learning_data, sentiments_learning)
    print("score on the learning data (accuracy) = ", accuracy_score(models[model].predict(learning_data), sentiments_learning))
    print("")


# # Fine Tune

# In[25]:


from sklearn.model_selection import GridSearchCV

grid_search_pipeline = Pipeline([
    ('text_pre_processing', TextPreProc()),
    ('vectorizer', TfidfVectorizer()),
    ('model', MultinomialNB()),
])

params = [
    {
        'text_pre_processing__use_mention': [True, False],
        'vectorizer__max_features': [1000, 2000, 5000, 10000, 20000, None],
        'vectorizer__ngram_range': [(1,1), (1,2)],
    },
]
grid_search = GridSearchCV(grid_search_pipeline, params, cv=5, scoring='f1')
grid_search.fit(learn_data, sentiments_learning)
print(grid_search.best_params_)


# # Test

# In[26]:


mnb.fit(learning_data, sentiments_learning)


# In[27]:



testing_data = pipeline.transform(test_data)
mnb.score(testing_data, sentiments_test)


# In[28]:


# Predecting on the test.csv
sub_data = test
sub_learning = pipeline.transform(sub_data.SentimentText)
sub = pd.DataFrame(sub_data.ItemID, columns=("ItemID", "Sentiment"))
sub["Sentiment"] = mnb.predict(sub_learning)
print(sub)


# In[29]:


sub_data


# # Test Your Tweet

# In[30]:


# Just run it
model = MultinomialNB()
model.fit(learning_data, sentiments_learning)


# In[62]:


tweet = pd.Series([input(),])
tweet = pipeline.transform(tweet)
proba = model.predict_proba(tweet)[0]
print("The probability that this tweet is sad is:", proba[0])
print("The probability that this tweet is happy is:", proba[1])


# In[ ]:





# In[ ]:




