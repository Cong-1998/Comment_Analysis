import bz2
import pickle
import _pickle as cPickle
from Cleaning.ms_cleaning import cleaning

def en_emotion(text, model):
    #label = model.predict([text])
    #string = " ".join(label)
    proba = model.predict_proba([text])
    bi = max(proba[0])
    if bi > 0.49:
        label = model.predict([text])
        string = " ".join(label)
    else:
        string = "other"
    return string

def ms_emotion(list_text, model):
    # clean malay text
    for i in range(len(list_text)):
        list_text[i] = cleaning(list_text[i])
    
    list_emo = []
    for text in list_text:
        prob = model.predict_proba([text])
        big = max(prob[0])
        if big > 0.53:
            if model.predict([text]) == 0:
                list_emo.append("anger")
            if model.predict([text]) == 1:
                list_emo.append("fear")
            if model.predict([text]) == 2:
                list_emo.append("joy")
            if model.predict([text]) == 3:
                list_emo.append("love")
            if model.predict([text]) == 4:
                list_emo.append("sadness")
            if model.predict([text]) == 5:
                list_emo.append("surprise")
        else:
            list_emo.append("other")
            
    return list_emo
    
def detect_emotion(df, malaya):

    # malay emotion analysis
    with bz2.BZ2File('Emotion/ms_emotion.pbz2', 'rb') as trainin_model:
        ms_model = cPickle.load(trainin_model)
    
    clean = df['clean'].values.tolist()
    ms_emo = ms_emotion(clean, ms_model)
    df = df.assign(Emotion = ms_emo)

    # english emotion analysis
    with bz2.BZ2File('Emotion/en_emotion.pbz2', 'rb') as training_model:
        en_model = cPickle.load(training_model)
    
    df.loc[df['Language'] == "en", 'Emotion'] = df['clean'].apply(en_emotion, model=en_model)

    # remove unwanted coulmns
    df = df.drop(['Language', 'clean'], axis = 1)

    return df
