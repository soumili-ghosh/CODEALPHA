import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("amazon.csv")


df = df[['reviews.text']].dropna().head(100)
df.rename(columns={'reviews.text': 'Text'}, inplace=True)

emotion_keywords = {
    "joy": ["happy", "joy", "love", "amazing", "great", "delight", "fantastic", "awesome", "pleased", "wonderful"],
    "anger": ["angry", "hate", "furious", "terrible", "awful", "worst", "mad", "annoyed"],
    "sadness": ["sad", "disappointed", "unhappy", "depressed", "bad", "regret", "cry", "sorrow"],
    "fear": ["scared", "afraid", "worried", "nervous", "concerned", "anxious"],
    "surprise": ["surprised", "shocked", "unexpected", "amazed", "wow", "unbelievable"]
}


vader = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity
    vader_score = vader.polarity_scores(text)['compound']
    avg = (tb_polarity + vader_score) / 2
    if avg > 0.1:
        return "Positive"
    elif avg < -0.1:
        return "Negative"
    else:
        return "Neutral"

def detect_emotion(text):
    lowered_text = text.lower()
    for emotion, keywords in emotion_keywords.items():
        if any(word in lowered_text for word in keywords):
            return emotion.capitalize()
    
    return classify_sentiment(text)


df['Sentiment'] = df['Text'].apply(classify_sentiment)
df['Emotion'] = df['Text'].apply(detect_emotion)


for i, row in df.iterrows():
    print("\nReview:")
    print(row['Text'].strip())
    print(f"Sentiment: {row['Sentiment']} | Emotion: {row['Emotion']}")


df.to_csv("sentiment_analysis_results.csv", index=False)
