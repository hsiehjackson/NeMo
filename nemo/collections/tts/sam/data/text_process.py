""" from https://github.com/keithito/tacotron """
import re
from unidecode import unidecode
from nemo.collections.tts.sam.data.numbers import normalize_numbers
from nemo.collections.tts.sam.data.symbols import symbols

class TextProcess:
    def __init__(self):
        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(symbols)}

        # Regular expression matching text enclosed in curly braces:
        self._curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

        # Regular expression matching whitespace:
        self._whitespace_re = re.compile(r'\s+')

        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
          ('mrs', 'misess'),
          ('mr', 'mister'),
          ('dr', 'doctor'),
          ('st', 'saint'),
          ('co', 'company'),
          ('jr', 'junior'),
          ('maj', 'major'),
          ('gen', 'general'),
          ('drs', 'doctors'),
          ('rev', 'reverend'),
          ('lt', 'lieutenant'),
          ('hon', 'honorable'),
          ('sgt', 'sergeant'),
          ('capt', 'captain'),
          ('esq', 'esquire'),
          ('ltd', 'limited'),
          ('col', 'colonel'),
          ('ft', 'fort'),
        ]]


    def expand_abbreviations(self, text):
      for regex, replacement in self._abbreviations:
        text = re.sub(regex, replacement, text)
      return text


    def expand_numbers(self, text):
      return normalize_numbers(text)


    def lowercase(self, text):
      return text.lower()


    def collapse_whitespace(self, text):
      return re.sub(self._whitespace_re, ' ', text)


    def convert_to_ascii(self, text):
      return unidecode(text)


    def basic_cleaners(self, text):
      '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
      text = self.lowercase(text)
      text = self.collapse_whitespace(text)
      return text


    def transliteration_cleaners(self, text):
      '''Pipeline for non-English text that transliterates to ASCII.'''
      text = self.convert_to_ascii(text)
      text = self.lowercase(text)
      text = self.collapse_whitespace(text)
      return text


    def english_cleaners(self, text):
      '''Pipeline for English text, including number and abbreviation expansion.'''
      text = self.convert_to_ascii(text)
      text = self.lowercase(text)
      text = self.expand_numbers(text)
      text = self.expand_abbreviations(text)
      text = self.collapse_whitespace(text)
      return text



    def get_arpabet(self, word, dictionary):
        word_arpabet = dictionary.lookup(word)
        if word_arpabet is not None:
            return "{" + word_arpabet[0] + "}"
        else:
            return word


    def text_to_sequence(self, text, cleaner_names, dictionary=None, keep_punct=True):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

          The text can optionally have ARPAbet sequences enclosed in curly braces embedded
          in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

          Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through
            dictionary: arpabet class with arpabet dictionary

          Returns:
            List of integers corresponding to the symbols in the text
        '''
        # print(text)
        # text = text.lower()

        sequence = []

        space = self._symbols_to_sequence(' ')
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = self._curly_re.match(text)
            if not m:
                clean_text = self._clean_text(text, cleaner_names)
                if dictionary is not None:
                    clean_text = [self.get_arpabet(w, dictionary) for w in re.findall(r"[\w']+|[.,!?;]", clean_text)]
                    for i in range(len(clean_text)):
                        t = clean_text[i]
                        if t.startswith("{"):
                            sequence += self._arpabet_to_sequence(t[1:-1])
                        else:
                            sequence += self._symbols_to_sequence(t, keep_punct)
                        sequence += space
                else:
                    sequence += self._symbols_to_sequence(clean_text, keep_punct)
                break
            sequence += self._symbols_to_sequence(self._clean_text(m.group(1), cleaner_names), keep_punct)
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # remove trailing space
        if dictionary is not None:
            sequence = sequence[:-1] if sequence[-1] == space[0] else sequence

        return sequence


    def sequence_to_text(self, sequence):
        '''Converts a sequence of IDs back to a string'''
        result = ''
        for symbol_id in sequence:
            if symbol_id in self._id_to_symbol:
                s = self._id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')


    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(self, name, "english_cleaners")
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)
        return text


    def _symbols_to_sequence(self, symbols, keep_punct=True):
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]


    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(['@' + s for s in text.split()])


    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s is not '_' and s is not '~'
