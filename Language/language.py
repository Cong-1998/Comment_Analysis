from langdetect import detect

# detect language
def detect_lang(text):
    try:
        return detect(text)
    except:
        return None

#print(detect_lang("he is a boy"))
