import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv(&quot;housing.csv&quot;)
X = data.drop(&quot;price&quot;, axis=1)
y = data[&quot;price&quot;]
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(&quot;MAE:&quot;, mean_absolute_error(y_test, pred))
print(&quot;R2 Score:&quot;, r2_score(y_test, pred))
