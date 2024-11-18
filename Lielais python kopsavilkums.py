import nltk
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import langid
import string
import textdistance
from textblob import TextBlob
import langdetect
import random
from deep_translator import GoogleTranslator
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Uzdevums 1: Vārdu biežuma sadalījums tekstā
def task_1():
    text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    freq_dist = FreqDist(words)

    for word, frequency in freq_dist.items():
        print(f"{word}: {frequency}")

# Uzdevums 2: Valodas noteikšana vairākos tekstos
def task_2():
    texts = [
        "Šodien ir saulaina diena.",
        "Today is a sunny day.",
        "Сегодня солнечный день.",
        "Šis ir mans teksts.", 
        "我有一棵大树，想吃的时候就吃.", 
        "Сара акомпиутер laaye изҩырц исаӡьы сымоуп.", 
        "qoghDu'Daq meQchoH nuq 'oH mIwvam'e'?",
        "'eko fay tsawl Na'vi!"
    ]

    for text in texts:
        language, confidence = langid.classify(text)  
        print(f"\"{text}\" - valoda: {language} - precizitāte: {confidence:.2f}")

# Uzdevums 3: Vārdu sakritība starp diviem tekstiem
def task_3():
    def calculate_word_overlap(text1, text2):
        translator = str.maketrans('', '', string.punctuation)
        text1_cleaned = text1.translate(translator).lower().split()
        text2_cleaned = text2.translate(translator).lower().split()
        similarity = textdistance.jaccard.similarity(set(text1_cleaned), set(text2_cleaned))
        overlap_percentage = similarity * 100
        common_words = set(text1_cleaned).intersection(set(text2_cleaned))
        return common_words, overlap_percentage

    text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
    text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."

    common_words, overlap_percentage = calculate_word_overlap(text1, text2)
    print(f"Kopīgie vārdi: {', '.join(common_words)}")
    print(f"Sakritības procentuālais līmenis: {overlap_percentage:.2f}%")

# Uzdevums 4: Noskaņojuma analīze ar tulkošanas atbalstu
def task_4():
    sentences = [
        "This is great!",
        "I am dissapointed, this is shit!",
        "Neitrāls produkts, neko nejūtu."
    ]

    def analyze_sentiment(sentence):
        try:
            lang = langdetect.detect(sentence)
        except Exception as e:
            print(f"Valodas noteikšana neizdevās: {e}")
            return None
        if lang == 'en':
            analysis = TextBlob(str(sentence))
        else:
            translated = GoogleTranslator(source=lang, target='en').translate(sentence)
            blob = TextBlob(str(translated))
            analysis = blob
        
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return "pozitīvs"
        elif polarity < 0:
            return "negatīvs"
        else:
            return "neitrāls"

    for sentence in sentences:
        try:
            lang = langdetect.detect(sentence)
        except Exception as e:
            print(f"Valodas noteikšana neizdevās teikumam {sentence}: {e}")
            continue
        sentiment = analyze_sentiment(sentence)
        original_text = sentence
        detected_lang = lang
        if detected_lang != 'en':
            try:
                translated_back = translate_text(sentiment, 'en', detected_lang)
                analyzed_text = translated_back
            except Exception as e:
                print(f"Tulkošana neizdevās: {e}")
                analyzed_text = sentiment
        else:
            analyzed_text = sentiment
        print(f"Teikums: \"{original_text}\" -> Noskaņojums: {analyzed_text}")

# Uzdevums 5: Teksta tīrīšana un normalizācija
def task_5():
    def clean_and_normalize_text(raw_text):
        text = re.sub(r'@[\w]+', '', raw_text)
        text = re.sub(r'http\S+', '', text)  
        text = re.sub(r'[^\w\s]', '', text) 
        text = text.lower()
        text = re.sub(r'\s+', ' ', text) 
        text = text.strip() 
        return text

    raw_text = "@John: Šis ir lielisks produkts!!! Vaine? 👏👏👏 http://example.com"
    cleaned_text = clean_and_normalize_text(raw_text)
    print(cleaned_text)

# Uzdevums 6: Automātiska rezumēšana
def summarize_article(text, sentences_count=2):
    translator = Translator()
    translated_text = translator.translate(text, src="lv", dest="en").text

    parser = PlaintextParser.from_string(translated_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = [str(sentence) for sentence in summarizer(parser.document, sentences_count)]
    
    summary_text = " ".join(summary)
    back_translated = translator.translate(summary_text, src="en", dest="lv").text
    return back_translated
text = """Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm."""
summary = summarize_article(text)
print("Rezumējums:")
print(summary)

# Uzdevums 7
def task_7():
    print("šobrīd vēl taisu")

# Uzdevums 8: Personu un organizāciju atpazīšana
def task_8():
    text = "Valsts prezidents Egils Levits piedalījās pasākumā, ko organizēja Latvijas Universitāte."
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(tags)
    person_names = []
    organizations = []
    for subtree in entities:
        if isinstance(subtree, nltk.Tree):  
            label = subtree.label()
            if label == 'PERSON': 
                person_names.append(" ".join(word for word, tag in subtree))
            elif label == 'GPE' or label == 'LOCATION': 
                organizations.append(" ".join(word for word, tag in subtree))
    print("Personvārdi:", person_names)
    print("Organizācijas:", organizations)

# Uzdevums 9: Stāstu ģenerēšana
def task_9():
    start = "Reiz kādā tālā zemē"
    story = [start]
    story_parts = [
        "Kāds mazs zēns devās slīpēt.",
        "Bembis nopļāva pus mežu.",
        "Cilvēki dzīvoja mierā un harmonijā."
    ]
    for _ in range(1):
        next_part = random.choice(story_parts)
        story_parts.remove(next_part)  
        story.append(next_part)
    print(" ".join(story))

# Uzdevums 10: Tulkot tekstus
def task_10():
    texts = ["Labdien! Kā jums klājas?", "Es šodien lasīju interesantu grāmatu."]
    translator = Translator()
    for text in texts:
        translation = translator.translate(text, src='lv', dest='en')
        print(f"Ievades teksts: {text}")
        print(f"Tulkots teksts: {translation.text}")

# Izvēlne, lai izvēlētos uzdevumu, kuru izpildīt
def main():
    while True:
        print("\nIzvēlies uzdevumu, kuru veikt:")
        print("1. Vārdu biežuma sadalījums")
        print("2. Valodas noteikšana vairākos tekstos")
        print("3. Vārdu sakritības aprēķins")
        print("4. Noskaņojuma analīze")
        print("5. Teksta tīrīšana un normalizācija")
        print("6. Automātiska rezumēšana")
        print("7. Šobrīd vēl taisu")
        print("8. Personu un organizāciju atpazīšana")
        print("9. Stāstu ģenerēšana")
        print("10. Tulkot tekstus")
        print("0. Iziet")
        
        choice = input("Ievadiet uzdevuma numuru: ")

        if choice == '1':
            task_1()
        elif choice == '2':
            task_2()
        elif choice == '3':
            task_3()
        elif choice == '4':
            task_4()
        elif choice == '5':
            task_5()
        elif choice == '6':
            task_6()
        elif choice == '7':
            task_7()
        elif choice == '8':
            task_8()
        elif choice == '9':
            task_9()
        elif choice == '10':
            task_10()
        elif choice == '0':
            break
        else:
            print("Nepareiza izvēle. Lūdzu, mēģiniet vēlreiz.")

# Izpildīt programmu
if __name__ == "__main__":
    main()
