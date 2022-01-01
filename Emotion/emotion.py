import bz2
import pickle
import _pickle as cPickle

def en_emotion(text, model):
    label = model.predict([text])
    string = " ".join(label)
    return string
    
def detect_emotion(df, malaya):

    # malay emotion analysis
    ms_model = malaya.emotion.multinomial()
    #ms_model = malaya.emotion.transformer(model = 'albert')
    clean = df['clean'].values.tolist()
    ms_emo = ms_model.predict(clean)
    df = df.assign(Emotion = ms_emo)

    # english emotion analysis
    with bz2.BZ2File('Emotion/en_emotion.pbz2', 'rb') as training_model:
        en_model = cPickle.load(training_model)
    
    df.loc[df['Language'] == "en", 'Emotion'] = df['clean'].apply(en_emotion, model=en_model)

    # remove unwanted coulmns
    df = df.drop(['Language', 'clean'], axis = 1)

    return df
