import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("Projekt")
st.caption("Możesz wybrać zbiór danych na bocznym pasku.")

def load_df(name):
    if name == "Iris":
        return load_iris(as_frame=True).frame
    if name == "Wine":
        return load_wine(as_frame=True).frame
    if name == "Titanic":
        return sns.load_dataset("titanic")

dataset = st.sidebar.selectbox("Wybierz zbiór danych:", ["Iris","Wine","Titanic"])
df = load_df(dataset)

st.subheader("Podgląd danych:")
st.dataframe(df.head())

st.header("Podstawowe statystyki opisowe")
col1, col2 = st.columns(2)

with col1:
    st.write(f"Liczba obserwacji: {df.shape[0]}")
    st.write(f"Liczba kolumn: {df.shape[1]}")

with col2:
    st.write(df.describe(include="number").T)

num = df.select_dtypes(include="number")

if not num.empty:
    statystyki = pd.DataFrame({
        "srednia": num.mean(),
        "mediana": num.median(),
        "odchylenie": num.std(),
        "min": num.min(),
        "max": num.max()
    })
    st.dataframe(statystyki)
else:
    st.info("Brak kolumn numerycznych.")

st.header("Analiza braków danych")

brak = df.isna().sum()
brak_p = df.isna().mean()*100

brak_df = pd.DataFrame({
    "braki": brak,
    "braki_procentowo": brak_p
}).sort_values("braki_procentowo", ascending=False)

st.subheader("Tabela braków danych")
st.dataframe(brak_df)

st.subheader("Heatmapa braków danych")
fig, ax = plt.subplots(figsize=(10,4))
sns.heatmap(df.isna(), cbar=False, ax=ax)
ax.set_xlabel("Kolumny")
ax.set_ylabel("Wiersze")
st.pyplot(fig)

st.header("Wybór kolumn do analizy")

kolumny = st.multiselect("Wubierz kolumny do analizy:", df.columns.tolist(), default=df.columns.tolist())

if not kolumny:
    st.warning("Wybierz przynajmniej jedną kolumnę.")
    st.stop()

df_wybrane = df[kolumny]

st.write("Wybrane kolumny:")
st.dataframe(df_wybrane.head())

st.header("Wizualizacja")

tab_hist, tab_box, tab_kor, tab_scatter = st.tabs(["Histogramy","Boxploty","Korelacje","Scatterploty"])

num_wybrane = df_wybrane.select_dtypes(include="number").columns

with tab_hist:
    st.subheader("Histogramy i wykresy rozkładu")
    if len(num_wybrane) == 0:
        st.info("Brak kolumn numerycznych")
    else:
        for nw in num_wybrane:
            st.write(nw)
            fig, ax = plt.subplots()
            sns.histplot(df_wybrane[nw].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

with tab_box:
    st.subheader("Boxploty")
    if len(num_wybrane) == 0:
        st.info("Brak kolumn numerycznych")
    else:
        for nw in num_wybrane:
            st.write(nw)
            fig, ax = plt.subplots()
            sns.boxplot(x=df_wybrane[nw].dropna(), ax=ax)
            st.pyplot(fig)

with tab_kor:
    st.subheader("Macierz korelacji")
    if len(num_wybrane) < 2:
        st.info("Za mało kolumn numerycznych")
    else:
        kor = df_wybrane[num_wybrane].corr()
        fig, ax = plt.subplots()
        sns.heatmap(kor, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

with tab_scatter:
    st.subheader("Wykresy rozrzutu")
    if len(num_wybrane) < 2:
        st.info("Za mało kolumn numerycznych")
    else:
        x, y = st.columns(2)
        with x:
            x = st.selectbox("X:", num_wybrane, key="x")
        with y:
            y = st.selectbox("Y:", num_wybrane, key="y")

        hue = st.selectbox("Hue", ["(brak)"] + df_wybrane.select_dtypes(exclude="number").columns.tolist())

        fig, ax = plt.subplots()
        if hue != "(brak)":
            sns.scatterplot(data=df_wybrane, x = x, y = y, hue = hue, ax = ax)
        else:
            sns.scatterplot(data=df_wybrane, x = x, y = y, ax = ax)
        st.pyplot(fig)

st.header("Informacje o typach danych")

numeryczne = df.select_dtypes(include="number").columns.tolist()
kategoryczne = df.select_dtypes(exclude="number").columns.tolist()

kn, kk = st.columns(2)

with kn:
    st.write("Kolumny numeryczne")
    if numeryczne:
        st.write(numeryczne)
    else:
        st.info("Brak kolumn numerycznych.")

with kk:
    st.write("Kolumny kategoryczne")
    if kategoryczne:
        st.write(kategoryczne)
    else:
        st.info("Brak kolumn kategorycznych.")

st.header("Outliery")

num_cols = num.columns

outliery = {}
for nc in num_cols:
    Q1 = df[nc].quantile(0.25)
    Q3 = df[nc].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    mask = (df[nc] < lower) | (df[nc] > upper)
    outliery[nc] = {
        "ile": int(mask.sum())
        ,"procent": round(mask.mean() * 100, 2)
        ,"lower": lower
        ,"upper": upper
    }

st.dataframe(pd.DataFrame(outliery).T)

st.header("Klasyfikacja - Regresja logistyczna")

cel = st.selectbox("Wybierz kolumnę kategoryczną:", df.select_dtypes(exclude="number").columns)
cechy = st.multiselect("Wybierz kolumny numeryczne:", df.select_dtypes(include="number").columns)

if cel and cechy:
    X = df[cechy].dropna()
    y = df[cel].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    skalowanie = StandardScaler()
    X_train_s = skalowanie.fit_transform(X_train)
    X_test_s = skalowanie.transform(X_test)
    C = st.slider("C", 0.01, 0.1, 1.0, 10.0, 100.0)
    penalty = st.selectbox("Penalty:", ["l2"])
    model = LogisticRegression(C=C, penalty=penalty, max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    st.subheader("Metryki klasyfikacji")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision", precision_score(y_test, y_pred, average="weighted"))
    st.write("Recall", recall_score(y_test, y_pred, average="weighted"))
    st.write("F1-score", f1_score(y_test, y_pred, average="weighted"))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

    if len(y.unique()) == 2:
        y_prob = model.predict_proba(X_test_s)[:, 1]
        fbr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fbr, tpr)
        ax.set_title("Krzywa ROC")
        st.pyplot(fig)

st.header("Regresja")

cel_r = st.selectbox("Wybierz kolumnę numeryczną:", df.select_dtypes(include="number").columns)
cechy_r = st.multiselect("Wybierz cechy:", df.select_dtypes(include="number").columns.drop(cel_r))

if cechy_r:
    X = df[cechy_r].dropna()
    y = df[cel_r].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model_r = LinearRegression()
    model_r.fit(X_train, y_train)

    y_pred = model_r.predict(X_test)

    st.subheader("Metryki regresji")
    st.write("R2:", r2_score(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("RMSE:", root_mean_squared_error(y_test, y_pred))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Rzeczywiste")
    ax.set_ylabel("Przewidywane")
    st.pyplot(fig)

    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot((y_test - y_pred), kde=True, ax=ax)
    ax.set_title("Residuals")
    st.pyplot(fig)

st.header("Klasteryzacja KMeans")

klasteryzacja = st.multiselect("Wybierz cechy numeryczne:", df.select_dtypes(include="number").columns)

if klasteryzacja:
    X = df[klasteryzacja].dropna()
    ile_klastrow = st.slider("Liczba klastrów:", 2, 10, 3)

    kmeans = KMeans(n_clusters=ile_klastrow, random_state=42)
    labels = kmeans.fit_predict(X)

    st.write("Score:", silhouette_score(X, labels))

    if ile_klastrow >= 2:
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis")
        ax.set_xlabel(klasteryzacja[0])
        ax.set_ylabel(klasteryzacja[1])
        st.pyplot(fig)