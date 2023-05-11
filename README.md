<h1 align="center">
    <p> TextPrepro - Text Preprocessing </p>
</h1>

<p align="center">
    <a href="https://pypi.org/project/textprepro">
        <img src="https://img.shields.io/pypi/v/textprepro.svg?logo=pypi&logoColor=white"
            alt="PyPI">
    </a>
    <a href="https://pypi.org/project/textprepro">
        <img src="https://img.shields.io/pypi/pyversions/textprepro?logo=python&logoColor=white"
            alt="Python">
    </a>    
    <a href="https://codecov.io/gh/umapornp/textprepro">
        <img src="https://img.shields.io/codecov/c/github/umapornp/textprepro?logo=codecov"
            alt="Codecov">
    </a>    
    <a href="https://github.com/umapornp/textprepro/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/umapornp/textprepro.svg?logo=github"
            alt="License">
    </a>
</p>


<p align="center">
    <img src="https://raw.githubusercontent.com/umapornp/textprepro/main/assets/banner.png">
</p>

**TextPrepro** - Everything Everyway All At Once Text Preprocessing: Allow you to preprocess both general and social media text with easy-to-use features. Help you gain insight from your data with analytical tools. Stand on the shoulders of various famous libraries (e.g., NLTK, Spacy, Gensim, etc.).

---------------------------------
### Table of Contents
* [⏳ Installation](#⏳-installation)
* [🚀 Quickstart](#🚀-quickstart)
    * [🧹 Simply preprocess with the pipeline](#🧹-simply-preprocess-with-the-pipeline)
    * [📂 Work with document or dataFrame](#📂-work-with-document-or-dataframe)
    * [🪐 Customize your own pipeline](#🪐-customize-your-own-pipeline)
* [💡 Features & Guides](#💡-features--guides)
    * [📋 For General Text](#📋-for-general-text)
    * [📱 For Social Media Text](#📱-for-social-media-text)
    * [🌐 For Web Scraping Text](#🌐-for-web-scraping-text)
    * [📈 Analytical Tools](#📈-analytical-tools)

---------------------------------

## ⏳ Installation
Simply install via `pip`:

```bash
pip install textprepro
```

or:
```bash
pip install "git+https://github.com/umapornp/textprepro"
```

---------------------------------

## 🚀 Quickstart

* ### 🧹 Simply preprocess with the pipeline

    You can preprocess your textual data by using the function `preprocess_text()` with the default pipeline as follows:

    ```python
    >>> import textprepro as pre

    # Proprocess text.
    >>> text = "ChatGPT is AI chatbot developed by OpenAI It is built on top of OpenAI GPT foundational large language models and has been fine-tuned an approach to transfer learning using both supervised and reinforcement learning techniques"

    >>> text = pre.preprocess_text(text)
    >>> text
    "chatgpt ai chatbot developed openai built top openai gpt foundational large language model finetuned approach transfer learning using supervised reinforcement learning technique"
    ```

* ### 📂 Work with document or dataFrame

    You can preprocess your document or dataframe as follows:

    * If you work with a list of strings, you can use the function `preprocess_document()` to preprocess each of them.

        ```python
        import textprepro as pre

        >>> document = ["Hello123", "World!@&"]
        >>> document = pre.preprocess_document(document)
        >>> document
        ["hello", "world"]
        ```

    * If you work with a dataframe, you can use the function `apply()` with the function `preprocess_text()` to apply the function to each row.

        ```python
        import textprepro as pre
        import pandas as pd

        >>> document = {"text": ["Hello123", "World!@&"]}
        >>> df = pd.DataFrame(document)
        >>> df["clean_text"] = df["text"].apply(pre.preprocess_text)
        >>> df
        ```

        | text      | clean_text |
        | :-------- | :--------- |
        | Hello123  | hello      |
        | World!@&  | world      |

* ### 🪐 Customize your own pipeline

    You can customize your own preprocessing pipeline as follows:
    ```python
    >>> import textprepro as pre

    # Customize pipeline.
    >>> pipeline = [
            pre.lower,
            pre.remove_punctuations,
            pre.expand_contractions,
            pre.lemmatize
        ]

    >>> text = "ChatGPT is AI chatbot developed by OpenAI It is built on top of OpenAI GPT foundational large language models and has been fine-tuned an approach to transfer learning using both supervised and reinforcement learning techniques"

    >>> text = pre.preprocess_text(text=text, pipeline=pipeline)
    >>> text
    "chatgpt is ai chatbot developed by openai it is built on top of openai gpt foundational large language model and ha been finetuned an approach to transfer learning using both supervised and reinforcement learning technique"
    ```

---------------------------------

## 💡 Features & Guides
TextPrep provides many easy-to-use features for preprocessing general text as well as social media text. Apart from preprocessing tools, TextPrep also provides useful analytical tools to help you gain insight from your data (e.g., word distribution graphs and word clouds).

* ### 📋 For General Text

    <!-- Misspelling Correction -->
    <details>
    <Summary> 👇 Misspelling Correction </Summary>

    Correct misspelled words:
    ```python
    >>> import textprepro as pre

    >>> text = "she loves swiming"

    >>> text = pre.correct_spelling(text)
    >>> text
    "she loves swimming"
    ```
    </details>


    <!-- Emoji & Emoticon -->
    <details>
    <Summary> 👇 Emoji & Emoticon </Summary>

    Remove, replace, or decode emojis (e.g., 👍, 😊, ❤️):
    ```python
    >>> import textprepro as pre

    >>> text = "very good 👍"

    # Remove.
    >>> text = pre.remove_emoji(text)
    >>> text
    "very good "

    # Replace.
    >>> text = pre.replace_emoji(text, "[EMOJI]")
    >>> text
    "very good [EMOJI]"

    # Decode.
    >>> text = pre.decode_emoji(text)
    >>> text
    "very good :thumbs_up:"
    ```

    Remove, replace, or decode emoticons (e.g., :-), (>_<), (^o^)):
    ```python
    >>> import textprepro as pre

    >>> text = "thank you :)"

    # Remove.
    >>> text = pre.remove_emoticons(text)
    >>> text
    "thank you "

    # Replace.
    >>> text = pre.replace_emoticons(text, "[EMOTICON]")
    >>> text
    "thank you [EMOTICON]"

    # Decode.
    >>> text = pre.decode_emoticons(text)
    >>> text
    "thank you happy_face_or_smiley"
    ```
    </details>

    <!-- URLs -->
    <details>
    <Summary> 👇 URL </Summary>

    Remove or replace URLs:
    ```python
    >>> import textprepro as pre

    >>> text = "my url https://www.google.com"

    # Remove.
    >>> text = pre.remove_urls(text)
    >>> text
    "my url "

    # Replace.
    >>> text = pre.replace_urls(text, "[URL]")
    >>> text
    "my url [URL]"
    ```
    </details>


    <!-- Email -->
    <details>
    <Summary> 👇 Email </Summary>

    Remove or replace emails.
    ```python
    >>> import textprepro as pre

    >>> text = "my email name.surname@user.com"

    # Remove.
    >>> text = pre.remove_emails(text)
    >>> text
    "my email "

    # Replace.
    >>> text = pre.replace_emails(text, "[EMAIL]")
    >>> text
    "my email [EMAIL]"
    ```
    </details>


    <!-- Number & Phone Number -->
    <details>
    <Summary> 👇 Number & Phone Number </Summary>

    Remove or replace numbers.
    ```python
    >>> import textprepro as pre

    >>> text = "my number 123"

    # Remove.
    >>> text = pre.remove_numbers(text)
    >>> text
    "my number "

    # Replace.
    >>> text = pre.replace_numbers(text)
    >>> text
    "my number 123"
    ```

    Remove or replace phone numbers.
    ```python
    >>> import textprepro as pre

    >>> text = "my phone number +1 (123)-456-7890"

    # Remove.
    >>> text = pre.remove_phone_numbers(text)
    >>> text
    "my phone number "

    # Replace.
    >>> text = pre.replace_phone_numbers(text, "[PHONE]")
    >>> text
    "my phone number [PHONE]"
    ```
    </details>


    <!-- Contraction -->
    <details>
    <Summary> 👇 Contraction </Summary>

    Expand contractions (e.g., can't, shouldn't, don't).
    ```python
    >>> import textprepro as pre

    >>> text = "she can't swim"

    >>> text = pre.expand_contractions(text)
    >>> text
    "she cannot swim"
    ```
    </details>


    <!-- Stopwords -->
    <details>
    <Summary> 👇 Stopword </Summary>

    Remove stopwords:
    You can also specify stopwords: `nltk`, `spacy`, `sklearn`, and `gensim`.
    ```python
    >>> import textprepro as pre

    >>> text = "her dog is so cute"

    # Default stopword is NLTK.
    >>> text = pre.remove_stopwords(text)
    >>> text
    "dog cute"

    # Use stopwords from Spacy.
    >>> text = pre.remove_stopwords(text, stpwords="spacy")
    >>> text
    "dog cute"
    ```
    </details>


    <!-- Punctuation & Special Character & Whitespace -->
    <details>
    <Summary> 👇 Punctuation & Special Character & Whitespace </Summary>

    Remove punctuations:
    ```python
    >>> import textprepro as pre

    >>> text = "wow!!!"

    >>> text = pre.remove_punctuations(text)
    >>> text
    "wow"
    ```

    Remove special characters:
    ```python
    >>> import textprepro as pre

    >>> text = "hello world!! #happy"

    >>> text = pre.remove_special_characters(text)
    >>> text
    "hello world happy"
    ```

    Remove whitespace:
    ```python
    >>> import textprepro as pre

    >>> text = "  hello  world  "

    >>> text = pre.remove_whitespace(text)
    >>> text
    "hello world"
    ```
    </details>


    <!-- Non-ASCII Character (Accent Character) -->
    <details>
    <Summary> 👇 Non-ASCII Character (Accent Character) </Summary>

    Standardize non-ASCII characters (accent characters):
    ```python
    >>> import textprepro as pre

    >>> text = "latté café"

    >>> text = pre.standardize_non_ascii(text)
    >>> text
    "latte cafe"
    ```
    </details>


    <!-- Stemming & Lemmatization -->
    <details>
    <Summary> 👇 Stemming & Lemmatization </Summary>

    Stem text:
    ```python
    >>> import textprepro as pre

    >>> text = "discover the truth"

    >>> text = pre.stem(text)
    >>> text
    "discov the truth"
    ```

    Lemmatize text:
    ```python
    >>> import textprepro as pre

    >>> text = "he works at a school"

    >>> text = pre.lemmatize(text)
    >>> text
    "he work at a school"
    ```
    </details>


    <!-- Lowercase & Uppercase -->
    <details>
    <Summary> 👇 Lowercase & Uppercase </Summary>

    Convert text to lowercase & uppercase:
    ```python
    >>> import textprepro as pre

    >>> text = "Hello World"

    # Lowercase
    >>> text = pre.lower(text)
    >>> text
    "hello world"

    # Uppercase
    >>> text = pre.upper(text)
    >>> text
    "HELLO WORLD"
    ```
    </details>


    <!-- Tokenization -->
    <details>
    <Summary> 👇 Tokenization </Summary>

    Tokenize text: You can also specify types of tokenization: `word` and `tweet`.
    ```python
    >>> import textprepro as pre

    >>> text = "hello world @user #hashtag"

    # Tokenize word.
    >>> text = pre.tokenize(text, "word")
    >>> text
    ["hello", "world", "@", "user", "#", "hashtag"]

    # Tokenize tweet.
    >>> text = pre.upper(text, "tweet")
    >>> text
    ["hello", "world", "@user", "#hashtag"]
    ```
    </details>




* ### 📱 For Social Media Text

    <!-- Slang -->
    <details>
    <Summary> 👇 Slang </Summary>

    Remove, replace, or expand slangs:
    ```python
    >>> import textprepro as pre

    >>> text = "i will brb"

    # Remove
    >>> pre.remove_slangs(text)
    "i will "

    # Replace
    >>> pre.replace_slangs(text, "[SLANG]")
    "i will [SLANG]"

    # Expand
    >>> pre.expand_slangs(text)
    "i will be right back"
    ```
    </details>


    <!-- Mention -->
    <details>
    <Summary> 👇 Mention </Summary>

    Remove or replace mentions.
    ```python
    >>> import textprepro as pre

    >>> text = "@user hello world"

    # Remove
    >>> text = pre.remove_mentions(text)
    >>> text
    "hello world"

    # Replace
    >>> text = pre.replace_mentions(text)
    >>> text
    "[MENTION] hello world"
    ```
    </details>


    <!-- Hashtag -->
    <details>
    <Summary> 👇 Hashtag </Summary>

    Remove or replace hashtags.
    ```python
    >>> import textprepro as pre

    >>> text = "hello world #twitter"

    # Remove
    >>> text = pre.remove_hashtags(text)
    >>> text
    "hello world"

    # Replace
    >>> text = pre.replace_hashtags(text, "[HASHTAG]")
    >>> text
    "hello world [HASHTAG]"
    ```
    </details>


    <!-- Retweet -->
    <details>
    <Summary> 👇 Retweet </Summary>

    Remove retweet prefix.
    ```python
    >>> import textprepro as pre

    >>> text = "RT @user: hello world"

    >>> text = pre.remove_retweet_prefix(text)
    >>> text
    "hello world"
    ```
    </details>


* ### 🌐 For Web Scraping Text

    <!-- HTML Tag -->
    <details>
    <Summary> 👇 HTML Tag </Summary>

    Remove HTML tags.
    ```python
    >>> import textprepro as pre

    >>> text = "<head> hello </head> <body> world </body>"

    >>> text = pre.remove_html_tags(text)
    >>> text
    "hello world"
    ```
    </details>


* ### 📈 Analytical Tools

    <!-- Word Distribution -->
    <details>
    <Summary> 👇 Word Distribution </Summary>

    Find word distribution.
    ```python
    >>> import textprepro as pre

    >>> document = "love me love my dog"

    >>> word_dist = pre.find_word_distribution(document)
    >>> word_dist
    Counter({"love": 2, "me": 1, "my": 1, "dog": 1})
    ```

    Plot word distribution in a bar graph.
    ```python
    >>> import textprepro as pre

    >>> document = "ChatGPT is AI chatbot developed by OpenAI It is built on top of OpenAI GPT foundational large language models and has been fine-tuned an approach to transfer learning using both supervised and reinforcement learning techniques"

    >>> word_dist = pre.find_word_distribution(document)
    >>> pre.plot_word_distribution(word_dist)
    ```

    <p align="center">
    <img src="https://raw.githubusercontent.com/umapornp/textprepro/main/assets/word_dist.png">
    </p>

    </details>


    <!-- Word Cloud -->
    <details>
    <Summary> 👇 Word Cloud </Summary>

    Generate word cloud.
    ```python
    >>> import textprepro as pre

    >>> document = "ChatGPT is AI chatbot developed by OpenAI It is built on top of OpenAI GPT foundational large language models and has been fine-tuned an approach to transfer learning using both supervised and reinforcement learning techniques"

    >>> pre.generate_word_cloud(document)
    ```

    <p align="center">
    <img src="https://raw.githubusercontent.com/umapornp/textprepro/main/assets/word_cloud.png">
    </p>

    </details>


    <!-- Rare & Frequent Word -->
    <details>
    <Summary> 👇 Rare & Frequent Word</Summary>

    Remove rare or frequent words.
    ```python
    >>> import textprepro as pre

    >>> document = "love me love my dog"

    # Remove rare word
    >>> document = pre.remove_rare_words(document, num_words=2)
    "love me love"

    # Remove frequent word
    >>> document = pre.remove_freq_words(document, num_words=2)
    "my dog"
    ```
    </details>