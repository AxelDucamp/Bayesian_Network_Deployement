import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from flask import Flask, render_template, request

df = pd.read_csv("train.csv")

x_train, x_test, y_train, y_test = train_test_split(
    df.drop("Survived", axis=1), df["Survived"], test_size=0.2, random_state=123)

CAT = ["Pclass", "Sex"]
NUM = ["Age", "Fare"]

features = list()

cat_var = [
    tf.feature_column.categorical_column_with_vocabulary_list(
        cat, df[cat].value_counts().index.tolist()
    )
    for cat in CAT
]

cat_var = [
    tf.feature_column.indicator_column(cat)
    for cat in cat_var
]


def MinMax(x, num, df):

    x = tf.cast(x, tf.float32)
    MIN = np.float32(np.min(df[num]))
    MAX = np.float32(np.max(df[num]))
    MEAN = np.float32(np.mean(df[num]))

    x = tf.where(tf.math.is_nan(x), MEAN, x)
    x = (x - MIN) / (MAX - MIN)

    return x


num_var = list()
num_var.append(
    tf.feature_column.numeric_column(
        "Age", normalizer_fn=lambda x: MinMax(x, "Age", x_train))
)
num_var.append(
    tf.feature_column.numeric_column(
        "Fare", normalizer_fn=lambda x: MinMax(x, "Fare", x_train))
)

for var in cat_var:
    features.append(var)

for var in num_var:
    features.append(var)

dense_features = tf.keras.layers.DenseFeatures(features)

tfpl = tfp.layers
layers = tf.keras.layers

inp = {
    "Pclass": tf.keras.layers.Input(shape=(), dtype=tf.int32),
    "Sex": tf.keras.layers.Input(shape=(), dtype=tf.string),
    "Age": tf.keras.layers.Input(shape=(), dtype=tf.float32),
    "Fare": tf.keras.layers.Input(shape=(), dtype=tf.float32)
}

num_class = 2

densef = dense_features(inp)
x = layers.Dense(64, activation="relu")(densef)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(tfpl.OneHotCategorical.params_size(num_class))(x)
out = tfpl.OneHotCategorical(num_class)(x)

model = tf.keras.models.Model(
    inp, out
)


def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)


model.compile(
    optimizer="adam",
    loss=negative_log_likelihood,
    metrics=["acc"]
)

model.summary()

model.load_weights("weights.h5")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Predict_html", methods=["POST"])
def predict_html():

    pclass = np.int32(request.form.get("pclass"))
    sex = str(request.form.get("sex"))
    age = np.float32(request.form.get("age"))
    fare = np.float32(request.form.get("fare"))

    to_pred = {
        "Pclass": np.array([pclass]),
        "Sex": np.array([sex]),
        "Age": np.array([age]),
        "Fare": np.array([fare])
    }

    output = model.predict(to_pred, verbose=1)
    output = np.argmax(output, axis=-1)[0]

    labels = ["Dead", "Alive"]

    return render_template("home.html", pred=labels[output])


if __name__ == "__main__":
    app.run(debug=False)
