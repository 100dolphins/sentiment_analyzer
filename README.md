# Documentation and Dataset for the Bachelor-Thesis: "Sentiment Analysis - exemplarische linguistische Analyse eines informationstechnischen Verfahrens"

The dataset consists of all reviews left for German Businesses in the Yelp Dataset Challenge Round 10 ([https://www.yelp.com/dataset/challenge])

The python script trains a naive Bayes Classifier on the 1-Star-Reviews and 5-Star-Reviews and tests its accuracy on the 2-Star-Reviews and 4-Star-Reviews. The classifier can also be used to classify any other text. Check out [http://www.nltk.org/api/nltk.classify.html] for a tutorial on using the classifier.

The script assumes that you have MongoDB up and running and imported "german_reviews.json" as the collection "german_reviews" into the database "test".

Check out [https://docs.mongodb.com/manual/installation/] if you need to install MongoDB.

Check out [https://docs.mongodb.com/manual/reference/program/mongoimport/] for importing the file via mongoimport.