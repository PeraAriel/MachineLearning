# Titanic Survival Prediction

## Descrizione
Questo progetto ha l'obiettivo di prevedere la sopravvivenza dei passeggeri del Titanic utilizzando dati storici. Viene sviluppato un modello di classificazione semplice (Logistic Regression) per iniziare, con possibilità di estensioni future (Decision Tree, test su dati esterni, ecc.).

---

## Dataset
I dati utilizzati provengono dal file `train.csv`.  
Le colonne principali utilizzate sono:

- **Survived**: target (0 = non sopravvissuto, 1 = sopravvissuto)
- **Pclass**: classe del biglietto
- **Sex**: sesso del passeggero
- **Age**: età
- **SibSp**: numero di fratelli/coniugi a bordo
- **Parch**: numero di genitori/figli a bordo
- **Fare**: tariffa del biglietto
- **Embarked**: porto di imbarco (C, Q, S)

Colonne escluse: `Name`, `Ticket`, `Cabin` (non utili o troppo difficili da elaborare per ora).

---

## Preprocessing
- Conversione delle variabili categoriche (`Sex`, `Embarked`) in numeri
- Gestione dei valori mancanti (`Age`)
- Selezione delle feature utili

---

## Modello
- **Logistic Regression** (massimo 500 iterazioni)
- Divisione del dataset in train (80%) e validation/test (20%)
- Addestramento del modello e valutazione tramite accuratezza
