import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 1, 100)
dmu = np.random.uniform(0.1, 1, 100)

X_train, X_val, y_train, y_val, dmu_train, dmu_val = train_test_split(x[:, np.newaxis], y, dmu, test_size=0.2, random_state=42)
weights_train = 1.0 / (dmu_train**2)

try:
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    scores = cross_val_score(model, X_train, y_train, cv=5,
    scoring='neg_mean_squared_error', params={'linearregression__sample_weight':weights_train})
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
