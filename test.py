import pandas as pd
import joblib
from fuzzywuzzy import fuzz
from sqlalchemy import create_engine
# from DBcreation import par_ratio, w_ratio, sort_ratio, avg

# Defining all functions again. if i call it from DBcreation, it runs whole DBcreation again....


def par_ratio(x, y):
    r = fuzz.partial_ratio(x, y)
    return r


def w_ratio(x, y):
    r = fuzz.WRatio(x, y)
    return r


def sort_ratio(x, y):
    r = fuzz.token_sort_ratio(x, y)
    return r


def avg(x, y, z):
    ans = (x + y + z) / 3
    return ans


# Load the pretrained model
model = joblib.load('saved_DTC.joblib')

# Load the dataset
engine = create_engine('postgresql://postgres:aditya@localhost:5432/duplicatefinderDB')
print("Engine created")
dbconn = engine.connect()
print("Database connected")
df1 = pd.read_sql("select * from doctorsdata", dbconn)

# Extract the first row from the dataset as input
input_row = df1.iloc[0].values.reshape(1, -1)

scores1 = []
print('score list created')
sig_col = (1, 2, 3, 5, 6, 7, 9, 11, 12)  # significant columns to be used
for comp_row in range(len(df1)-1):  # select second row to be compared (all rows after 1st comparison row)
    temp = [0 for i in range(14)]  # temp is used to store similarity score of the current rows in comparison
    # total 9 scores will be their as 9 columns will be compared, rest will be 0
    print(comp_row)
    for col_no in sig_col:
        ele = input_row[0, col_no]
        comp_ele = df1.iat[comp_row, col_no]
        # fetch elements form two rows in comparison

        if ele != '' or comp_ele != '' or ele is not pd.NA or comp_ele is not pd.NA:
            # checked if any element is NULL or NA
            r1 = par_ratio(str(ele), str(comp_ele))
            r2 = w_ratio(str(ele), str(comp_ele))
            r3 = sort_ratio(str(ele), str(comp_ele))

            average = avg(r1, r2, r3)
            temp[col_no] = average
        # The average of these three scores is calculated and
        # stored in the temp list at the corresponding column index.

    scores1.append(temp)


# Predict duplicates
predictions = model.predict(scores1)


# Find indices of 'Duplicate' predictions
duplicate_indices = [index for index, prediction in enumerate(predictions) if prediction == 'Duplicate']

print(duplicate_indices)

