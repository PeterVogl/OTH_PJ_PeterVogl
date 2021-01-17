import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


# ----- Einlesen der CSV-Dateien -----
training_data = pd.read_csv("Dateipfad zur Trainingsdatei", sep=";")     # Einlesen des angefertigten Trainings- u. Validierungsdatensatz
evaluation_data = pd.read_csv("Dateipfad zur Evaluationsdatei", sep=";")     # Einlesen des Evaluationsdatensatz
# print(data)
entry_knots = ["Tiefe", "Geschwindigkeit", "Verweildauer", "Laufweg", "Loyalität"]    # Auswahl der Knotendaten
outcome = ["Authentizität"]     # Auswahl der Vergleichsdaten

training_entry_data_values = training_data[entry_knots]      # Erstellen eines Data Frames mit den Inputdaten
training_outcome_data_values = training_data[outcome]        # Erstellen eines Data Frames mit den Vergleichsdaten

evaluation_entry_data_values = evaluation_data[entry_knots]     # Erstellen eines Data Frames mit den Inputdaten der Evaluation
evaluation_outcome_data_values = evaluation_data[outcome]       # Erstellen eines Data Frames mit den Vergleichsdaten der Evaluation

# Normalisieren der Daten
sc = MinMaxScaler()     # Initialisieren des MinMax-Scaler für die Eingabe
sc_out = MinMaxScaler()     # Initialisieren des MinMax-Scaler für die Ausgabe
training_entry_data_values_sc = sc.fit_transform(training_entry_data_values)    # Normalisieren der Trainingsdaten
evaluation_entry_data_values_sc = sc.transform(evaluation_entry_data_values)      # Normalisieren der Evaluationsdaten

# Aufteilen der Daten in Trainings- und Testdaten
training_entry_data_train, training_entry_data_test, training_outcome_data_train, training_outcome_data_test \
    = train_test_split(training_entry_data_values_sc, training_outcome_data_values)

# ----- Erstellen des Neuronalen Netzes -----
knn = Sequential()      # Erstellen eines sequentiellen neuronalen Models
knn.add(Dense(20, input_dim=training_entry_data_train.shape[1], activation="relu", name="Input"))     # Input Layer mit 20 Knoten
knn.add(Dense(120, activation="relu", name="Layer_1"))       # Erste Schicht mit 120 Knoten
# Da eine Abschätzung für 2 Werte erwartet nicht authentisch/authentisch werden zwei Knoten im Output benötigt
knn.add(Dense(2, activation="softmax", name="Output"))
knn.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Kompilieren des KNN
# knn.summary()

# ----- Training und Validierung -----
# Mit Hilfe des aufgeteilten Datensatzes wird das Model über 9 Epochen mit einer Samplesize von 128 trainiert
knn_history = knn.fit(training_entry_data_train, training_outcome_data_train, epochs=9, batch_size=128,
                      validation_data=(training_entry_data_test, training_outcome_data_test), verbose=0)

# Auslesen der Loss- und Metricswerte
data_history = pd.DataFrame.from_dict(knn_history.history)
print(data_history)

# Ausführen der Evaluation des Modells - Loss- und Metricswerte werden zurückgegeben
ergebnis = knn.evaluate(evaluation_entry_data_values_sc, evaluation_outcome_data_values, verbose=0)
print()
print(ergebnis)

# Ausgabe der Loos- und Metricswerte in grafischer Form
"""
fig = plt.figure(figsize=(20, 10), num="Neuronales Netz")
plot_1 = fig.add_subplot(121)
plot_1.plot(data_history.index, data_history.iloc[:, 0], color="blue")
plot_1.plot(data_history.index, data_history.iloc[:, 2], color="red")
plot_1.legend(["Training", "Validierung"])
plot_1.set_xlabel("epoch")
plot_1.set_ylabel(knn.metrics_names[0])
plot_2 = fig.add_subplot(122)
plot_2.plot(data_history.index, data_history.iloc[:, 1], color="blue")
plot_2.plot(data_history.index, data_history.iloc[:, 3], color="red")
plot_2.legend(["Training", "Validierung"])
plot_2.set_xlabel("epoch")
plot_2.set_ylabel(knn.metrics_names[1])
fig.show()
"""