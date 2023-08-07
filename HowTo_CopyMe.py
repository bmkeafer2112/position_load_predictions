# DB connection oly occurs once when you declare the object.
# First ARG is denso name as it is called postgres.
# Second ARG is how many rows of data you want when you call.
# Returns Pandas DataFrame [(defined by you as second arg) rows x 21 columns]
# 1000 rows, 100 seconds of data = 6.95 second processing time.
# 900 rows, 96 seconds of data = 6.32 second processing time.
# 800 rows, 85 seconds of data = 5.53 second processing time.
# 700 rows, 75 seconds of data = 4.85 second processing time.
# 600 rows, 63 seconds of data = 4.17 second processing time.
# 500 rows, 56 seconds of data = 3.43 second processing time.


from LabelMotionPath import LabelMotionPath as LMP
getMotionPaths = LMP()
yourVal = getMotionPaths.doWork('denso_04')
print(yourVal)

yourVal.to_csv('denso_04_Sunday.csv')
