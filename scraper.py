
import sklearn.datasets  as skd

categories = ['elektronik','yiyecek','kagit_kozmetik','tekstil','temizlik']

news_test = skd.load_files(r'C:\\Users\\ASUSNB\\Desktop\\test_ios',
                          categories = categories, encoding= 'ISO-8859-1')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes  import MultinomialNB
from sklearn.pipeline import make_pipeline

# Creating a model based on Multinomiaal Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#Training the model with the train data
model.fit(news_test.data, news_test.target)
#Creating labels for the test data
labels = model.predict(news_test.data)

class Scraper():

    def scrapedata(self,tag):
          pred = model.predict([tag])
          qlist = news_test.target_names[pred[0]]
          return qlist





quotes = Scraper()






