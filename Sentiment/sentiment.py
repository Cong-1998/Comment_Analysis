from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Cleaning.ms_cleaning import cleaning

def remove_stopwords(data, stop_words):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stop_words:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

def en_sentiment(text, model):
    
    sent = model.polarity_scores(text)
    return sent['compound']

def ms_sentiment(text, model):
    # clean malay text
    for i in range(len(text)):
        text[i] = cleaning(text[i])

    # remove stopword
    stop_words = set(open('stopwords.txt').read().splitlines())
    list_text = remove_stopwords(text, stop_words)
    
    # predict sentiment
    ms_sen = model.predict(list_text)
    return ms_sen
    
def detect_sentiment(df, malaya):
    
    # malay sentiment analysis
    model = malaya.sentiment.multinomial()
    clean = df['clean'].values.tolist()
    ms_sen = ms_sentiment(clean, model)
    df = df.assign(Sentiment = ms_sen)

    # english sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    df.loc[df['Language'] == "en", 'Score'] = df['clean'].apply(en_sentiment, model=analyzer)
    df.loc[df['Score']>=0.05, 'Sentiment']='positive'
    df.loc[(df['Score']<0.05) & (df['Score']>-0.05), 'Sentiment']='neutral'
    df.loc[df['Score']<=-0.05, 'Sentiment']='negative'

    # remove unwanted coulmns
    df = df.drop(['Score'], axis = 1)

    return df

