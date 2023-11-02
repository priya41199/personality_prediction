import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

sw = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(sample):
  sample = sample.lower()

  sample = re.sub("[^a-zA-Z]+", " ", sample)

  sample = sample.split(" ")
  sample = [re.sub("http.*","",s) for s in sample if s not in sw]
  sample = [ps.stem(s) for s in sample]
  sample = " ".join(sample)
  return sample