import os
import re
import nltk
import spacy
import emoji
import string
import unicodedata
import contractions
import importlib.util

from bs4 import BeautifulSoup
from textblob import TextBlob
from emot import EMOTICONS_EMO
from collections import Counter
from matplotlib import pyplot as plt
from typing import Callable, List, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize
from gensim.parsing.preprocessing import remove_stopwords as gensim_remove_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS
from textprepro.utils import (
    SLANG,
    WHITESPACE_PATTERN,
    URL_PATTERN,
    NUMBER_PATTERN,
    MENTION_PATTERN,
    HASHTAG_PATTERN,
    RETWEET_PREFIX_PATTERN,
    SPECIAL_CHARACTER_PATTERN, 
    EMAIL_PATTERN,
    PHONE_NUMBER_PATTERN
)


nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('wordnet', quiet=True)


TOKENIZER = TweetTokenizer()


def tokenize(text: str, method: str = "tweet") -> List[str]:
    """ Tokenize text.
    
    For example: 
        >>> import textprepro as pre
        >>> print(pre.tokenize("hello world @user #hashtag", "tweet"))
        ['hello', 'world', '@user', '#hashtag']
        >>> print(pre.tokenize("hello world @user #hashtag", "word"))
        ['hello', 'world', '@', 'user', '#', 'hashtag']
    
    Args:
        text: Text.  
        method: Tokenization method (Default=`tweet`):
        * `tweet`: Tokenization for tweets or social media text that can tokenize hashtags, metions, etc..
        * `word`: Tokenization for general text.
    
    Returns:
        List of tokens.
    """
    if method == "tweet":
        return TweetTokenizer().tokenize(text)
    
    elif method == "word":
        return word_tokenize(text)

    else:
        raise ValueError(f"Tokenization method {method} not found.")


def lower(text: str) -> str:
    """ Convert text to lowercase.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.lower("Hello World"))
        "hello world"
    
    Args:
        text: Text.
    
    Returns:
        Lowercase text.
    """
    return text.lower()


def upper(text: str) -> str:
    """ Convert text to uppercase.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.lower("Hello World"))
        "HELLO WORLD"
    
    Args:
        text: Text.
    
    Returns:
        Uppercase text.
    """
    return text.upper()


def remove_whitespace(text: str) -> str:
    """ Remove whitespace from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_whitespace("  hello  world  "))
        "hello world"
    
    Args:
        text: Text.
    
    Returns:
        Text without whitespace.
    """
    return re.sub(WHITESPACE_PATTERN, " ", text).strip()


def remove_html_tags(text: str) -> str:
    """ Remove HTML tags from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_html_tags("<head> hello </head> <body> world </body>"))
        "hello world"
    
    Args:
        text: Text.
    
    Returns:
        Text without HTML tags. 
    """
    soup = BeautifulSoup(text, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()
    return " ".join(soup.stripped_strings)


def remove_emojis(text: str) -> str:
    """ Remove emojis from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_emojis("very good 👍"))
        "very good"
    
    Args:
        text: Text.
    
    Returns:
        Text without emojis.
    """
    return emoji.replace_emoji(text, "")


def replace_emojis(text: str, replace: str = "") -> str:
    """ Replace emojis with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_emojis("very good 👍", "[EMOJI]"))
        "very good [EMOJI]"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
    
    Returns:
        Text with emojis replaced with specified string.
    """
    return emoji.replace_emoji(text, replace)


def decode_emojis(text: str) -> str:
    """ Convert emojis from Unicode to their descriptions.
    
    For example:
    >>> import textprepro as pre
    >>> print(pre.decode_emojis("very good 👍"))
    "very good :thumbs_up:"
    
    Args:
        text: Text.
    
    Returns:
        Text with emojis replaced with their descriptions.
    """
    return emoji.demojize(text)


def remove_emoticons(text: str) -> str:
    """ Remove emoticons from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_emoticons("thank you :)"))
        "thank you"
    
    Args:
        text: Text.
    
    Returns:
        Text without emoticons.
    """
    for emo in EMOTICONS_EMO:
        text = text.replace(emo, "")
    return text


def replace_emoticons(text: str, replace: str = "") -> str:
    """ Replace emoticons with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_emoticons("thank you :)", "[EMOTICON]"))
        "thank you [EMOTICON]"
        
    Args:
        text: Text.
        replace: String to replace with (Default = "").
    
    Returns:
        Text with emoticons replaced with specified string.
    """
    for emo in EMOTICONS_EMO:
        text = text.replace(emo, replace)
    return text


def decode_emoticons(text: str) -> str:
    """ Convert emoticons to their descriptions.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.decode_emoticons("thank you :)"))
        "thank you happy_face_or_smiley"
    
    Args:
        text: Text.
    
    Returns:
        Text with emoticons replaced with their descriptions.
    
    References:
        List of emoticons come from this [repo](https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py).
    """
    for emo in EMOTICONS_EMO:
        text = text.replace(emo, "_".join(EMOTICONS_EMO[emo].replace(",", "").lower().split()))
    return text


def expand_contractions(text: str) -> str:
    """ Expand contractions in text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.expand_contractions("she can't swim"))
        "she cannot swim"
        
    Args:
        text: Text.
    
    Returns:
        Text with expanded contractions.
    """
    return contractions.fix(text)


def remove_urls(text: str) -> str:
    """ Remove URLs from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_urls("my url https://www.google.com"))
        "my url"
    
    Args:
        text: Text.
    
    Returns:
        Text without URLs.
    """
    return re.sub(URL_PATTERN, "", text)


def replace_urls(text: str, replace: str = "") -> str:
    """ Replace URLs with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_urls("my url https://www.google.com", "[URL]"))
        "my url [URL]"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
        
    Returns:
        Text with URLs replaced with specified string.
    """
    return re.sub(URL_PATTERN, replace, text)


def remove_emails(text: str) -> str:
    """ Remove emails from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_emails("my email name.surname@user.com"))
        "my email"
    
    Args:
        text: Text
    
    Returns:
        Text without emails.
    """
    return re.sub(EMAIL_PATTERN, "", text)


def replace_emails(text: str, replace: str = "") -> str:
    """ Replace emails with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_emails("my email name.surname@user.com", "[EMAIL]"))
        "my email [EMAIL]"
    
    Args:
        text: Text
        replace: String to replace with (Default = "").
    
    Returns:
        Text with emails replaced with specified string.
    """
    return re.sub(EMAIL_PATTERN, replace, text)


def remove_numbers(text: str) -> str:
    """ Remove numbers from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_numbers("my number 123"))
        "my number"
    
    Args:
        text: Text.
    
    Returns:
        Text without numbers.
    """
    return re.sub(NUMBER_PATTERN, "", text)


def replace_numbers(text: str, replace: str = '') -> str:
    """ Replace numbers with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_numbers("my number 123", "[NUMBER]"))
        "my number [NUMBER]"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
    
    Returns:
        Text with numbers replaced with specified string.
    """
    return re.sub(NUMBER_PATTERN, replace, text)


def remove_phone_numbers(text: str) -> str:
    """ Remove phone numbers from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_phone_numbers("my phone number +1 (123)-456-7890"))
        "my phone number"
    
    Args:
        text: Text
    
    Returns:
        Text without phone numbers.
    """
    return re.sub(PHONE_NUMBER_PATTERN, "", text)


def replace_phone_numbers(text: str, replace: str = "") -> str:
    """ Replace phone numbers with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_phone_numbers("my phone number +1 (123)-456-7890", "[PHONE]"))
        "my phone number [PHONE]"
    
    Args:
        text: Text
        replace: String to replace with (Default = "").
    
    Returns:
        Text with phone numbers replaced with specified string.
    """
    return re.sub(PHONE_NUMBER_PATTERN, replace, text)


def remove_retweet_prefix(text: str) -> str:
    """ Remove retweet prefix (RT @user:) from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_retweet_prefix("RT @user: hello world"))
        "hello world"
    
    Args:
        text: Text.
    
    Returns:
        Text without retweet prefix.
    """
    return re.sub(RETWEET_PREFIX_PATTERN, "", text)


def remove_mentions(text: str) -> str:
    """ Remove user mentions from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_mentions("@user hello world"))
        "hello world"
    
    Args:
        text: Text
    
    Returns:
        Text without user mentions.
    """
    return re.sub(MENTION_PATTERN, "", text)


def replace_mentions(text: str, replace: str = "") -> str:
    """ Replace user mentions (@user) with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_mentions("@user hello world", "[MENTION]"))
        "[MENTION] hello world"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
    
    Returns:
        Text with user mentions replaced with specified string.
    """
    return re.sub(MENTION_PATTERN, replace, text)


def remove_hashtags(text: str) -> str:
    """ Remove hashtags (#hashtag) from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_hashtags("hello world #twitter"))
        "hello world"
    
    Args:
        text: Text
    
    Returns:
        Text without hashtags.
    """
    return re.sub(HASHTAG_PATTERN, "", text)


def replace_hashtags(text: str, replace: str = "") -> str:
    """ Replace hashtags with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_hashtags("hello world #twitter", "[HASHTAG]"))
        "hello world [HASHTAG]"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
    
    Returns:
        Text with hashtags replaced with specified string.
    
    """
    return re.sub(HASHTAG_PATTERN, replace, text)


def remove_slangs(text: str) -> str:
    """ Remove chat slangs and acronyms from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_slangs("i will brb"))
        "i will"
    
    Args:
        text: Text.
        
    Returns:
        Text without slangs.
    
    Reference:
        List of slangs come from
        [repo-1](https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing/notebook)
        and
        [repo-2](https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt)
    """
    tokens = TOKENIZER.tokenize(text)
    return " ".join(["" if token.lower() in SLANG else token for token in tokens])


def replace_slangs(text: str, replace: str = "") -> str:
    """ Replace chat slangs and acronyms with specified string.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.replace_slangs("i will brb", "[SLANG]"))
        "i will [SLANG]"
    
    Args:
        text: Text.
        replace: String to replace with (Default = "").
        
    Returns:
        Text with slangs/acronyms replaced with specified string.
    
    Reference:
        List of slangs come from
        [repo-1](https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing/notebook)
        and
        [repo-2](https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt)
    """
    tokens = TOKENIZER.tokenize(text)
    return " ".join([replace if token.lower() in SLANG else token for token in tokens])


def expand_slangs(text: str) -> str:
    """ Expand chat slangs and acronyms.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.expand_slangs("i will brb"))
        "i will be right back"
    
    Args:
        text: Text.
        
    Returns:
        Text with slangs/acronyms replaced with their meanings.
    
    Reference:
        List of slangs come from
        [repo-1](https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing/notebook)
        and
        [repo-2](https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt)
    """
    tokens = TOKENIZER.tokenize(text)
    return " ".join([SLANG[token.lower()] if token.lower() in SLANG else token for token in tokens])


def standardize_non_ascii(text: str) -> str:
    """ Standardize non-ASCII characters in text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.standardize_non_ascii("latté café"))
        "latte cafe"
    
    Args:
        text: Text.
    
    Returns:
        Text with normalized ASCII characters.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_special_characters(text: str) -> str:
    """ Remove special characters from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_special_characters("hello world!! #happy"))
        "hello world happy"
    
    Args:
        text: Text.
    
    Returns:
        Text without special characters.
    """
    return re.sub(SPECIAL_CHARACTER_PATTERN, "", text)


def remove_punctuations(text: str) -> str:
    """ Remove punctuations from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_punctuations("wow!!!"))
        "wow"
    
    Args:
        text: Text.
    
    Returns:
        Text without punctuations.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text: str, stpwords: str = "nltk") -> str:
    """ Remove stopwords from text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_stopwords("her dog is so cute"))
        "dog cute"
    
    Args:
        text: Text.
        stpwords: Library name of stopwords:
        (`nltk`, `spacy`, `sklearn`, or `gensim`)
    
    Returns:
        Text without stopwords.
    """
    if stpwords == "nltk":
        stpwords = set(stopwords.words('english'))
    
    elif stpwords == "spacy":
        if not importlib.util.find_spec("en_core_web_sm"):
            os.system("python -m spacy download en_core_web_sm")
        en = spacy.load('en_core_web_sm')
        stpwords = en.Defaults.stop_words
    
    elif stpwords == "sklearn":
        stpwords = ENGLISH_STOP_WORDS
    
    elif stpwords == "gensim":
        return gensim_remove_stopwords(text)
    
    else:
        raise ValueError(f"Stopwords {stpwords} not found.")
    
    tokens = TOKENIZER.tokenize(text)
    
    return " ".join([token for token in tokens if token.lower() not in stpwords])


def stem(text: str) -> str:
    """ Stem each word in text. 
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.stem("discover the truth"))
        "discov the truth"
    
    Args:
        text: Text.
    
    Returns:
        Text with words normalized with their stem. 
    """
    tokens = TOKENIZER.tokenize(text)
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(token) for token in tokens]) 


def lemmatize(text: str) -> str:
    """ Lemmatize each word in text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.lemmatize("he works at a school"))
        "he work at a school"
    
    Args:
        text: Text.
    
    Returns:
        Text with words normalized with their base forms.
    """
    tokens = TOKENIZER.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])


def correct_spelling(text: str) -> str:
    """ Correct misspelled words in text.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.correct_spelling("i lovee swiming"))
        "i love swimming"
    
    Args:
        text: Text.
    
    Returns:
        Text with corrected words.
    """
    corrector = TextBlob(text)
    return corrector.correct().string


def find_word_distribution(document: str) -> Counter:
    """ Find word distribution in document.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.find_word_distribution("love me love my dog"))
        Counter({"love": 2, "me": 1, "my": 1, "dog": 1})
        
    Args:
        document: Document.
    
    Returns:
        Couter object containing word distribution. 
    """
    tokens = TOKENIZER.tokenize(document)
    return Counter(tokens)


def plot_word_distribution(word_dist: Counter, title: str = "Word Distribution") -> None:
    """ Plot and display word distribution in a bar graph.
    
    Args:
        word_dist: Word distribution.
        title: title of bar graph (Default = "Word Distribution").
    """
    word_dist = word_dist.most_common()
    word = list(zip(*word_dist))[0]
    count = list(zip(*word_dist))[1]
    plt.bar(word, count)
    plt.xticks(rotation=50)
    plt.title(title)
    plt.xlabel('word')
    plt.ylabel('count')
    plt.show()


def generate_word_cloud(text: str) -> None:
    """ Generate word cloud from text.
    
    Args:
        text: Text.
    """
    word_cloud = WordCloud(
        background_color="white",
        collocations=False,
        stopwords=STOPWORDS
    ).generate(text)
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def remove_rare_words(document: str, num_words: int, word_dist: Optional[Counter] = None) -> str:
    """ Remove rare words from document.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_rare_words("love me love my dog", 2))
        "love me love"
    
    Args:
        document: Document.
        num_words: Number of rare words to remove.
        word_dist: Word distribution (Default = `None`).
    
    Returns:
        Document without rare words.
    """
    WORD_DIST = word_dist if word_dist else find_word_distribution(document)
    RARE_WORDS = list(zip(*WORD_DIST.most_common()[:-num_words-1:-1]))[0]
    tokens = TOKENIZER.tokenize(document)
    return " ".join([token for token in tokens if token not in RARE_WORDS])


def remove_freq_words(document: str, num_words: int, word_dist: Optional[Counter] = None) -> str:
    """ Remove frequent words from document.
    
    For example:
        >>> import textprepro as pre
        >>> print(pre.remove_freq_words("love me love my dog", 2))
        "my dog"
    
    Args:
        document: Document.
        num_words: Number of frequent words to remove.
        word_dist: Word distribution (Default = `None`).
    
    Returns:
        Document without frequent words.
    """
    WORD_DIST = word_dist if word_dist else find_word_distribution(document)
    FREQ_WORDS = list(zip(*WORD_DIST.most_common()[:num_words]))[0]
    tokens = TOKENIZER.tokenize(document)
    return " ".join([token for token in tokens if token not in FREQ_WORDS])


DEFAULT_PIPELINE = [
    remove_html_tags,
    lower,
    decode_emojis,
    decode_emoticons,
    standardize_non_ascii,
    remove_urls,
    remove_emails,
    remove_phone_numbers,
    remove_numbers,
    remove_retweet_prefix,
    remove_mentions,
    expand_contractions,
    expand_slangs,
    remove_punctuations,
    remove_special_characters,
    remove_stopwords,
    remove_whitespace,
    lemmatize
]


def preprocess_text(text: str, pipeline: Optional[List[Callable]] = None) -> str:
    """ Preprocessing pipeline for text.
    
    Args:
        text: Text
        pipeline: Preprocessing pipeline. If `None`, `default pipeline` will be used.
    
    Returns:
        Preprocessed text.
    """
    pipeline = pipeline if pipeline else DEFAULT_PIPELINE
    
    for func in pipeline:
        text = func(text)
    
    return text


def preprocess_document(document: List[str], pipeline: Optional[List[Callable]] = None) -> List[str]:
    return [preprocess_text(text, pipeline) for text in document]