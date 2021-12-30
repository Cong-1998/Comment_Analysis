from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def en_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sent = analyzer.polarity_scores(text)
    return sent['compound']
    
def detect_sentiment(df, malaya):
    
    # malay sentiment analysis
    model = malaya.sentiment.multinomial()
    clean = df['clean'].values.tolist()
    ms_sen = model.predict(clean)
    df = df.assign(Sentiment = ms_sen)

    # english sentiment analysis
    df.loc[df['Language'] == "en", 'Score'] = df['clean'].apply(en_sentiment)
    df.loc[df['Score']>=0.05, 'Sentiment']='positive'
    df.loc[(df['Score']<0.05) & (df['Score']>-0.05), 'Sentiment']='neutral'
    df.loc[df['Score']<-0.05, 'Sentiment']='negative'

    # remove unwanted coulmns
    df = df.drop(['Score'], axis = 1)

    return df

