# Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
# Download stopwords (only once)
nltk.download(&#39;stopwords&#39;)
# Sample dataset (you can replace with CSV later)
data = pd.DataFrame({
&#39;label&#39;: [&#39;spam&#39;, &#39;ham&#39;, &#39;spam&#39;, &#39;ham&#39;, &#39;spam&#39;, &#39;ham&#39;],
&#39;message&#39;: [
&#39;Win money now&#39;,
&#39;Meeting scheduled tomorrow&#39;,
&#39;Claim your free prize&#39;,
&#39;Project discussion at noon&#39;,
&#39;Limited offer click now&#39;,
&#39;Lunch at cafeteria&#39;
]
})
# Display dataset
print(&quot;Dataset:\n&quot;, data)
# Convert labels to binary
data[&#39;label&#39;] = data[&#39;label&#39;].map({&#39;ham&#39;: 0, &#39;spam&#39;: 1})
# Preprocessing + TF-IDF
tfidf = TfidfVectorizer(stop_words=stopwords.words(&#39;english&#39;))
X = tfidf.fit_transform(data[&#39;message&#39;])
y = data[&#39;label&#39;]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42
)
# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# Output
print(&quot;\nAccuracy:&quot;, accuracy)
print(&quot;\nConfusion Matrix:\n&quot;, cm)
