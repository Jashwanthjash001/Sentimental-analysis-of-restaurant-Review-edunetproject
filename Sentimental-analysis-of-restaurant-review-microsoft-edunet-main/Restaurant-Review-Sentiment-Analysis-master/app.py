from flask import Flask, request, render_template
import model  # This is your sentiment analysis script
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = model.predict_sentiment(review) 
        pp=model.fuun(review)
         # Analyze the review
        if sentiment==1:
            py='this is a positive review'
        else:
            py='this is a negative review'   
        return render_template('result.html', sentiment=py,pp=pp)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = 'hello'
    word_tokens = word_tokenize(review)
    if predict_sentiment(review):
        print('This is a POSITIVE review') 
        sid = SentimentIntensityAnalyzer()
    # Count the number of positive words
        positive_word_count = 0
        for word in word_tokens:
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                positive_word_count += 1
        if positive_word_count<=2:
            print('The stars you have earned is 3')
        elif positive_word_count>2 and positive_word_count<=3:
            print('The stars you have earned is 4')
        elif positive_word_count>4:
        # and positive_word_count<=4:
            print('The stars you have earned is 5')
    #     elif positive_word_count>4:
    #         print('The stars you have earned is 5' ) 
    else:
        print('This is Negative review!')
        print('The stars you have earned is 1')  
        # Your prediction code here
        pass

if __name__ == '__main__':
    app.run(debug=True)
