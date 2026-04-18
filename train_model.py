import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

data = []

for _ in range(1000):
    sleep = np.random.randint(3, 10)
    workout = np.random.randint(0, 120)
    work_hours = np.random.randint(0, 12)
    screen_time = np.random.randint(0, 12)
    diet = np.random.randint(1, 10)

    score = (
        (20 if 7 <= sleep <= 8 else 10 if sleep >= 6 else 0) +
        min(workout / 5, 20) +
        min(work_hours * 2, 20) +
        (20 if screen_time <= 2 else 10 if screen_time <= 4 else 0) +
        diet * 2
    )

    data.append([sleep, workout, work_hours, screen_time, diet, score])

df = pd.DataFrame(data, columns=[
    "sleep", "workout", "work_hours", "screen_time", "diet", "score"
])

X = df.drop("score", axis=1)
y = df["score"]

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")