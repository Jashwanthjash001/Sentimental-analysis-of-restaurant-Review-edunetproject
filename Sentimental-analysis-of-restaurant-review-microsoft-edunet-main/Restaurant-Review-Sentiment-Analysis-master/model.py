# Importing libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle


# Loading the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
print(df)
df.info()
df.describe()
corpus = []

# Looping till 1000 because the number of rows are 1000
for i in range(0, 1000):
    # Removing the special character from the reviews and replacing it with space character
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['Review'][i])

    # Converting the review into lower case character
    review = review.lower()

    # Tokenizing the review by words
    review_words = review.split()

    # Removing the stop words using nltk stopwords
    review_words = [word for word in review_words if not word in set(
        stopwords.words('english'))]

    # Stemming the words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]

    # Joining the stemmed words
    review = ' '.join(review)

    # Creating a corpus
    corpus.append(review)


# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv.pkl")


# Model Building
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train) 
def predict_sentiment(sample_review):
  sample_review = re.sub(pattern = '[^a-zA-Z]',repl=' ',string = sample_review)
  sample_review =sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps=PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = cv.transform([final_review]).toarray()
  return classifier.predict(temp)
def fuun(sample_review):
    if predict_sentiment(sample_review):
        sid = SentimentIntensityAnalyzer()
        positive_word_count = 0
        for word in word_tokens:
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                positive_word_count += 1
        if positive_word_count<=2:
            return 3
        elif positive_word_count>2 and positive_word_count<=3:
            return 4
        elif positive_word_count>4:
    # and positive_word_count<=4:
            return 5
    else: return 1

sample_review = 'The food is bad'

if predict_sentiment(sample_review):
  print('This is a POSITIVE review')
else:
  print('This is Negative review!')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# nltk.download('vader_lexicon')
# nltk.download('punkt')

# Assuming 'review' is your text
# review = input("Enter your review")

# # Tokenize the review
word_tokens = word_tokenize(review)
# if predict_sentiment(review):
#     print('This is a POSITIVE review') 
#     sid = SentimentIntensityAnalyzer()
#     # Count the number of positive words
#     positive_word_count = 0
#     for word in word_tokens:
#         if (sid.polarity_scores(word)['compound']) >= 0.5:
#             positive_word_count += 1
#             # ps=positive_word_count
#     if positive_word_count<=2:
#         print('The stars you have earned is 3')
#     elif positive_word_count>2 and positive_word_count<=3:
#         print('The stars you have earned is 4')
#     elif positive_word_count>4:
#     # and positive_word_count<=4:
#         print('The stars you have earned is 5')
# #     elif positive_word_count>4:
# #         print('The stars you have earned is 5' ) 
# else:
#     print('This is Negative review!')
#     print('The stars you have earned is 1')  