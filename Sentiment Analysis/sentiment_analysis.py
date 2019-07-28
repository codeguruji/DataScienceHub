from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

score = analyzer.polarity_scores('The movie is great')

#this will print the normalized (-1 to 1) polarity scores for each of pos, neg, neu and combined score
print(score)
