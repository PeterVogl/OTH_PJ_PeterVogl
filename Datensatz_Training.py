import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

target_page_start, target_page_middle, target_page_end = 2, 12, 20
samples_begin, samples_mid, samples_end = 200, 1200, 600
sample_size = samples_begin + samples_mid + samples_end

# np.random.seed(42)
# ----- Scrolltiefe -----
# Die Daten für die Scrolltiefe werden mit drei gaußschen Normalverteilungen erzeugt
depth_start = np.random.normal(15, 8, samples_begin)
depth_middle = np.random.normal(50, 20, samples_mid)
depth_end = np.random.normal(80, 15, samples_end)
depth_final = np.concatenate((depth_start, depth_middle, depth_end), axis=None)

# count, bins, ignored = plt.hist(depth_final, bins="auto", density=False)
# plt.show()

# ----- Scrollgeschwindigkeit -----
# Die Daten für die Scrollgeschwindigkeit werden mit drei gaußschen Normalverteilungen erzeugt
speed_start = np.random.normal(500, 80, samples_begin)
speed_middle = np.random.normal(1000, 500, samples_mid)
speed_end = np.random.normal(3500, 1500, samples_end)
speed_final = np.concatenate((speed_start, speed_middle, speed_end), axis=None)
np.random.shuffle(speed_final)

# count, bins, ignored = plt.hist(speed_final, bins="auto", density=False)
# plt.show()

# ----- Verweildauer -----
# Die Daten für die Verweildauer werden mit drei gaußschen Normalverteilungen erzeugt
time_start = np.random.normal(120, 50, samples_begin)
time_middle = np.random.normal(420, 100, samples_mid)
time_end = np.random.normal(780, 200, samples_end)
time_final = np.concatenate((time_start, time_middle, time_end), axis=None)

# count, bins, ignored = plt.hist(time_final, bins="auto", density=False)
# plt.show()

# ----- Zielseite -----
# Die Daten für die Zielseite werden mit drei gaußschen Normalverteilungen erzeugt
target_start = np.random.normal(target_page_start, 3, samples_begin)
target_middle = np.random.normal(target_page_middle, 4, samples_mid)
target_end = np.random.normal(target_page_end, 3, samples_end)
target_final = np.concatenate((target_start, target_middle, target_end), axis=None)
np.random.shuffle(target_final)

# count, bins, ignored = plt.hist(target_final, bins="auto", density=False)
# plt.show()

# ----- Laufweg -----
# Die Daten für die Laufweg werden mit einer gaußschen Normalverteilung erzeugt
# Im Anschluss werden die Daten auf 0 bzw. 1 gerundet
# 0 - Zugriff über Suchfunktion / 1 - Zugriff durch durchlaufen der Seitenhierarchie
userpath = np.random.normal(0.65, 0.2, sample_size)
userpath = np.around(userpath)

# count, bins, ignored = plt.hist(userpath, bins="auto", density=False)
# plt.show()

# ----- Loyalität -----
# Die Daten für die Loyalität werden mit einer gaußschen Normalverteilung erzeugt
loyalty = np.random.normal(6, 1.8, sample_size)

# count, bins, ignored = plt.hist(loyalty, bins="auto", density=False)
# plt.show()

# ----- Zufriedenheit -----
# Die Daten für die Zufriedenheit werden mit einer gaußschen Normalverteilung erzeugt
# Im Anschluss werden die Daten auf 0 bzw. 1 gerundet
# 0 - zufrieden / 1 - unzufrieden
satisfaction = np.random.normal(0.60, 0.2, sample_size)
satisfaction = np.around(satisfaction)

# count, bins, ignored = plt.hist(satisfaction, bins="auto", density=False)
# plt.show()


data = pd.DataFrame(dict(Tiefe=depth_final, Geschwindigkeit=speed_final, Verweildauer=time_final,
                         Zielseite=target_final, Laufweg=userpath, Loyalität=loyalty,
                         Zufriedenheit=satisfaction)).astype("int32")
data = data[data["Tiefe"] >= 0]             # Tiefendaten unter 0% und 100% werden entfernt
data = data[data["Tiefe"] <= 100]
data = data[data["Geschwindigkeit"] >= 0]   # Geschwindigkeitsdaten unter 0 werden entfernt
data = data[data["Verweildauer"] > 0]       # Verweildauer unter 0 wird entfernt
data = data[data["Zielseite"] >= 0]         # Seitenzahlen unter 0 werden entfernt
data = data[data["Laufweg"] >= 0]           # Laufwege unter 0 und über 1 werden entfernt
data = data[data["Laufweg"] <= 1]
data = data[data["Loyalität"] >= 0]         # Loyalitätsdaten unter 0 und über 10 werden entfernt
data = data[data["Loyalität"] <= 10]
data = data[data["Zufriedenheit"] >= 0]     # Zufriedenheit unter 0 und über 1 wird entfernt
data = data[data["Zufriedenheit"] <= 1]
data["Authentizität"] = pd.Series(index=data.index, dtype="int32")      # Einfügen der Spalte Authentizität

# Gewichtung der Daten zur Berechnung der Authentizität
for index, row in data.iterrows():
    # Anwenden des natürlichen Logarithmus auf die Scrolltiefe
    if row["Tiefe"] == 0:
        row["Authentizität"] = row["Authentizität"] + 0
    else:
        row["Authentizität"] = row["Authentizität"] + 4 * np.log(row["Tiefe"])

    # Nutzung des Geschwindigkeitswertes als Exponent zur Basis 0.9
    row["Authentizität"] = row["Authentizität"] * pow(0.90, row["Geschwindigkeit"]/100)

    # Anwenden des natürlichen Logarithmus auf die Scrolltiefe
    if row["Verweildauer"] == 0:
        row["Authentizität"] = row["Authentizität"] * 0
    else:
        row["Authentizität"] = row["Authentizität"] * np.log(row["Verweildauer"])

    # Wenn der Einstieg z.B. über die Suchfunktion erfolgt sinkt die Authentizität um 50%
    if row["Laufweg"] == 0:
        row["Authentizität"] = row["Authentizität"] * 0.5
    else:
        row["Authentizität"] = row["Authentizität"] * 1

    # Anwenden der Zweierpotenz auf die Loyalität
    row["Authentizität"] = row["Authentizität"] * pow(row["Loyalität"], 2)

quantile = np.quantile(data["Authentizität"], 0.4)      # Berechnung des 40%-Quantiles

# Umwandlung der Authentizitätwerte in binäre Werte
for index, row in data.iterrows():
    if row["Authentizität"] <= quantile:    # Ist die Authentizität kleiner als das 40%-Quantil gilt sie als
        row["Authentizität"] = 0            # nicht vertrauenswürdig und wird mit 0 markiert
    else:
        row["Authentizität"] = 1            # Ist der Wert größer als das 40%-Quantil wird er mit 1 markiert

# Ausgabe des Dataframes als CSV-Datei
# data.to_csv("Dateipfad zur Ausgabe der CSV-Datei", index=False, sep=";")

print(data)
