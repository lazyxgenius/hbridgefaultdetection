#qwerty

# requirements - pandas, SQLAlchemy, psycopg2, fuzzywuzzy,
import pandas as pd
import psycopg2  # used for insert/update commands...no use here
import sqlalchemy
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
from fuzzywuzzy import process # used for extra matching when we already have a list of potential matches..no use here.
print("Script started")

engine = create_engine('postgresql://postgres:aditya@localhost:5432/duplicatefinderDB')
print("Engine created")

dbconn = engine.connect()
print("Database connected")

df = pd.read_sql("select * from doctorsdata", dbconn)
print("Data fetched")

len_col = len(df.columns)
# print(len_col)


# Replacing NULL and NA with ' '
# only addressline 2 and middle name had null values
df['addressline2'] = df['addressline2'].replace(['NULL'], '')
df['addressline2'] = df['addressline2'].replace([pd.NA], '')
df['MiddleName'] = df['MiddleName'].replace(['NULL'], '')
df['MiddleName'] = df['MiddleName'].replace([pd.NA], '')

# merging addressline1 and addressline2
df['addressline1'] = df['addressline1'] + df['addressline2']
print("Null values changed")

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
print("functions created")

same_rows = []  # list to store pair of similar rows
scores = []  # list to store similarity score between two rows
labels = []  # list to store labels DUPLICATE / NOT_DUPLICATE

sig_col = (1, 2, 3, 5, 6, 7, 9, 11, 12)  # significant columns to be used
print("lists created")
for row_no in range(len(df) - 1):
    print(f"Processing row {row_no}...")# select first row to be compared
    for comp_row in range(row_no + 1,len(df) - 1):  # select second row to be compared (all rows after 1st comparison row)
        temp = [0 for i in range(14)]  # temp is used to store similarity score of the current rows in comparison
                                        # total 9 scores will be their as 9 columns will be compared, rest will be 0

        for col_no in sig_col:
            ele = df.iat[row_no, col_no]
            comp_ele = df.iat[comp_row, col_no]
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

        scores.append(temp)

        # declaring same rows by setting a threshold
        count = 0
        for ele in temp:
            if ele >= 90:
                count += 1
            if count >= 4:
                same_rows.append((row_no, comp_row))

        # making label dataset
        # this dataset will be used as a 'Y' in our ML model.
        # here we check if the rows are actually duplicate or not by
        # matching the NationalProviderIdentifier
        if df.iat[row_no, 4] == df.iat[comp_row, 4]:
            labels.append("Duplicate")
        else:
            labels.append("Not-Duplicate")

print("loop has ended")

# Converts the labels list to a DataFrame df_lab and saves it to a CSV file named labels.csv
# df_lab = pd.DataFrame(labels)
# df_lab.to_csv(r"C:\Users\Aditya PC\PycharmProjects\duplicatefinder\labels.csv", index=False)
#
#
# # Converts the scores list to a DataFrame df_lab and saves it to a CSV file named scores.csv
# df_lab = pd.DataFrame(scores)
# df_lab.to_csv(r"C:\Users\Aditya PC\PycharmProjects\duplicatefinder\scores.csv", index=False)
print("csv created")
print(same_rows)
# print(scores)
# print(labels)
print("code ended")




