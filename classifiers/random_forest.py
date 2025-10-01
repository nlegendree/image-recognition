from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, cm, y_test, y_pred