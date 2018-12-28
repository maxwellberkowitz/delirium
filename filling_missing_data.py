import pandas as pd
def fill_and_average(subject):
    column_labels = list(subject)
    column_labels = column_labels[2:] # Skip time and subject ID columns
    for label in column_labels:
        x = pd.DataFrame(subject[label].rolling(window = 200, min_periods=1).mean()) #Rolling average based on last 200 points, requiring only 1 point in order to compute the average
        subject[label] = x[label].values #Reassign filled values to subject
    return subject