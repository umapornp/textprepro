import textprepro as pre

from collections import Counter
from unittest import TestCase, mock


class Test_Preprocess(TestCase):

    def test_tokenize(self):
        self.assertEqual(pre.tokenize("hello world @user #hashtag", "tweet"), list(["hello", "world", "@user", "#hashtag"]))
        self.assertEqual(pre.tokenize("hello world @user #hashtag", "word"), list(["hello", "world", "@", "user", "#", "hashtag"]))
        
        with self.assertRaises(ValueError):
            pre.tokenize("hello world @user #hashtag", "")
    
    
    def test_lower(self):
        self.assertEqual(pre.lower("Hello HELLO hello"), "hello hello hello")
    
    
    def test_upper(self):
        self.assertEqual(pre.upper("Hello HELLO hello"), "HELLO HELLO HELLO")
    
    
    def test_remove_whitespace(self):
        self.assertEqual(pre.remove_whitespace("  remove  space  "), "remove space")


    def test_remove_html_tags(self):
        self.assertEqual(pre.remove_html_tags("<head> hello </head> <body> world </body>"), "hello world")

    
    def test_remove_emojis(self):
        self.assertEqual(pre.remove_emojis("very good 👍👍 :)"), "very good  :)")
    
    
    def test_replace_emojis(self):
        self.assertEqual(pre.replace_emojis("very good 👍👍 :)", "[EMO]"), "very good [EMO][EMO] :)")
    
        
    def test_decode_emojis(self):
        self.assertEqual(pre.decode_emojis("very good 👍👍 :)"), "very good :thumbs_up::thumbs_up: :)")
    
    
    def test_remove_emoticons(self):
        self.assertEqual(pre.remove_emoticons("very good 👍👍 :)"), "very good 👍👍 ")
    
    
    def test_replace_emoticons(self):
        self.assertEqual(pre.replace_emoticons("very good 👍👍 :)", "[EMOT]"), "very good 👍👍 [EMOT]")
    
    
    def test_decode_emoticons(self):
        self.assertEqual(pre.decode_emoticons("very good 👍👍 :)"), "very good 👍👍 happy_face_or_smiley")
    
    
    def test_expand_contractions(self):
        self.assertEqual(pre.expand_contractions("i CAN'T do this"), "i CANNOT do this")
    
    
    def test_remove_urls(self):
        self.assertEqual(pre.remove_urls("go https://www.go.co.th go http://www.go.co.th go http://go.co.th go www.go.co.th"), "go  go  go  go ")
    
    
    def test_replace_urls(self):
        self.assertEqual(pre.replace_urls("go https://www.go.co.th go http://www.go.co.th go http://go.co.th go www.go.co.th", "[URL]"), "go [URL] go [URL] go [URL] go [URL]")
    
    
    def test_remove_numbers(self):
        self.assertEqual(pre.remove_numbers("i have 3 cats"), "i have  cats")
    
    
    def test_replace_numbers(self):
        self.assertEqual(pre.replace_numbers("i have 3 cats", "[NUM]"), "i have [NUM] cats")
    
    
    def test_remove_punctuations(self):
        self.assertEqual(pre.remove_punctuations("my name is @_!"), "my name is ")
    
    
    def test_remove_stopwords(self):
        self.assertEqual(pre.remove_stopwords("i love her dogs"), "love dogs")
        self.assertEqual(pre.remove_stopwords("i love her dogs", "spacy"), "love dogs")
        self.assertEqual(pre.remove_stopwords("i love her dogs", "sklearn"), "love dogs")
        self.assertEqual(pre.remove_stopwords("i love her dogs", "gensim"), "love dogs")
        self.assertRaises(ValueError, pre.remove_stopwords, "i love her dogs", "")
    
    
    def test_remove_mentions(self):
        self.assertEqual(pre.remove_mentions("@user_12 hello @user_12 user@go.com"), " hello  user@go.com")
    
    
    def test_replace_mentions(self):
        self.assertEqual(pre.replace_mentions("@user_12 hello @user_12 user@go.com", "[MEN]"), "[MEN] hello [MEN] user@go.com")
    
    
    def test_remove_hashtags(self):
        self.assertEqual(pre.remove_hashtags("#lov_1e hello #love world #hello"), " hello  world ")
    
    
    def test_replace_hashtags(self):
        self.assertEqual(pre.replace_hashtags("#lov_1e hello #love world #hello", "[TAG]"), "[TAG] hello [TAG] world [TAG]")
    
    
    def test_remove_retweet_prefix(self):
        self.assertEqual(pre.remove_retweet_prefix("rt @user12_: hello world"), "hello world")
    
    
    def test_stem(self):
        self.assertEqual(pre.stem("discovery"), "discoveri")
    
    
    def test_lemmatize(self):
        self.assertEqual(pre.lemmatize("he loves dogs"), "he love dog")
    
    
    def test_remove_special_characters(self):
        self.assertEqual(pre.remove_special_characters("hello *&^%$#@ world"), "hello  world")
        self.assertEqual(pre.remove_special_characters("hello world!! #happy"), "hello world happy")
    
    
    def test_standardize_non_ascii(self):
        self.assertEqual(pre.standardize_non_ascii("latté"), "latte")
    
    
    def test_correct_spelling(self):
        self.assertEqual(pre.correct_spelling("i lovee swimingg"), "i love swimming")
    
    
    def test_find_word_distribution(self):
        self.assertEqual(pre.find_word_distribution("love you love my dog"), Counter({"love": 2, "you": 1, "my": 1, "dog": 1}))
    
    
    @mock.patch("textprepro.preprocess.plt")
    def test_plot_word_distribution(self, mock_plt):
        word_dist = Counter("love me love my dog".split())
        title = "test"

        pre.plot_word_distribution(word_dist=word_dist, title=title)
        mock_plt.title.assert_called_once_with(title)
        mock_plt.show.assert_called_once()
    
    
    @mock.patch("textprepro.preprocess.plt")
    def test_generate_word_cloud(self, mock_plt):
        pre.generate_word_cloud("love me love my dog")
        mock_plt.show.assert_called_once()

    
    def test_remove_rare_words(self):
        self.assertEqual(pre.remove_rare_words("love you love my dog", 2), "love you love")
    
    
    def test_remove_freq_words(self):
        self.assertEqual(pre.remove_freq_words("love you love my dog", 2), "my dog")
    
    
    def test_remove_slangs(self):
        self.assertEqual(pre.remove_slangs("i will brb"), "i will ")
    
    
    def test_replace_slangs(self):
        self.assertEqual(pre.replace_slangs("i will brb", "[SL]"), "i will [SL]")
    
    
    def test_expand_slangs(self):
        self.assertEqual(pre.expand_slangs("i will brb"), "i will be right back")
    
    
    def test_remove_emails(self):
        self.assertEqual(pre.remove_emails("my user@gmail.com thanks"), "my  thanks")
    
    
    def test_replace_emails(self):
        self.assertEqual(pre.replace_emails("my user12.user12_user12@g-mail.co.th thanks", "[MAIL]"), "my [MAIL] thanks")
    
    
    def test_remove_phone_numbers(self):
        self.assertEqual(pre.remove_phone_numbers("tel: +66 (97) 247 2500 thanks"), "tel:  thanks")
    
    
    def test_replace_phone_numbers(self):
        self.assertEqual(pre.replace_phone_numbers("tel: +66 (97) 247 2500 thanks", "[PHONE]"), "tel: [PHONE] thanks")
    
    
    def test_preprocess_text(self):
        self.assertEqual(pre.preprocess_text("hello world @user #hashtag"), "hello world hashtag")
    
        
    def test_preprocess_document(self):
        self.assertEqual(pre.preprocess_document(["hello world @user", "world hello @user"]), ["hello world", "world hello"])