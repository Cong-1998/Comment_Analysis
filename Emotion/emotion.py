import bz2
import pickle
import _pickle as cPickle

def en_emotion(text):
    with bz2.BZ2File('Emotion/en_emotion.pbz2', 'rb') as training_model:
        en_model = pickle.load(training_model)

    label = en_model.predict(text)
    return label
    
def detect_emotion(df, malaya):

    # malay emotion analysis
    ms_model = malaya.emotion.multinomial()
    #ms_model = malaya.emotion.transformer(model = 'albert')
    clean = df['clean'].values.tolist()
    ms_emo = ms_model.predict(clean)
    df = df.assign(Emotion = ms_emo)

    # english emotion analysis
    df.loc[df['Language'] == "en", 'Emotion'] = df['clean'].apply(en_emotion)

    # remove unwanted coulmns
    df = df.drop(['Language', 'clean'], axis = 1)

    return df
