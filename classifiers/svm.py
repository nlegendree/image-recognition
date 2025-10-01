from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

def train_svm(X_train, y_train, X_test, y_test):
    model = LinearSVC(random_state=0, max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, cm, y_test, y_pred