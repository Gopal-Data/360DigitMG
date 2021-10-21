import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
Sdeal_Review=[] 

for i in range(1,2):
  ip=[]  
  url="https://www.snapdeal.com/product/samsung-guru-music-2-duos/197236301/reviews?page="+str(i)  
  response = requests.get(url)
  response=response.text
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("div",attrs={"class","#defaultReviewsCard p"})# Extracting the content under specific tags
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
    Sdeal_Review=Sdeal_Review+ip  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("sdeal.txt","w",encoding='utf8') as output: 
    output.write(str(Sdeal_Review))
	
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(Sdeal_Review) 

import nltk
help(nltk)
#from nltk.corpus import stopwords
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words that contained in iphone XR reviews
ip_reviews_words = ip_rev_string.split(" ")

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)

with open("C:/Users/gopal/Documents/360DigiTMG/mod 18/New folder/stopwords_en.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend([" most","helpful","positive","review"])

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("C:/Users/gopal/Documents/360DigiTMG/mod 18/New folder/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)
 

# negative words Choose path for -ve words stored in system
with open("C:/Users/gopal/Documents/360DigiTMG/mod 18/New folder/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)


