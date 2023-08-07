
import psycopg2 as pg
import pandas as pd
import datetime as dt
import os
import numpy as np
import math
# C:/Users/CSWEAT/Desktop/Project Archive/Smart Cell/Software/NUK/DensoDataCleaning
class LabelMotionPath:
	def __init__(self):
		self.conn = pg.connect("dbname=postgres user=postgres password=admin host=192.168.187.7 port=5432")
		self.cur = self.conn.cursor()
		

	def doWork(self, robot):
		whichRobot = robot
		#self.cur.execute("""SELECT * FROM positiondata_allrobots_programdt13ah3 WHERE "Robot_Name""")
		self.cur.execute("""SELECT * FROM positiondata_allrobots_programdt13ah3_4_hours WHERE "Robot_Name" = %s ORDER BY "Time_Stamp" """, [whichRobot])
		#self.cur.execute( """SELECT * FROM positiondata_allrobots_programdt13ah3 WHERE "Robot_Name" = %s ORDER BY "Time_Stamp""", [whichRobot])
      #self.cur.execute("""SELECT * FROM everything WHERE "Robot_Name" = %s AND "Time_Stamp" > Now() - interval '10 minutes' ORDER BY "Time_Stamp""", [whichRobot, limit])
		colnames = [desc[0] for desc in self.cur.description]
		records = self.cur.fetchall()
		dataFrameTP = pd.DataFrame(data = records, columns = colnames)
		dataFrameControllerTP = pd.read_csv(os.getcwd()+"\TP_Data.csv")
		dataFrameTP["Teach_Point"] = 999
		some_Number = 0

		for some_Number in range(400):
			some_Number_X = dataFrameControllerTP["TCP_Current_x"][some_Number]
			some_Number_Y = dataFrameControllerTP["TCP_Current_y"][some_Number]
			some_Number_Z = dataFrameControllerTP["TCP_Current_z"][some_Number]
			some_Number_RX = dataFrameControllerTP["TCP_Current_rx"][some_Number]
			some_Number_RY = dataFrameControllerTP["TCP_Current_ry"][some_Number]
			some_Number_RZ = dataFrameControllerTP["TCP_Current_rz"][some_Number]
	
			for i in range(len(dataFrameTP)):
				if math.isclose(dataFrameTP["TCP_Current_x"][i], some_Number_X, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_y"][i], some_Number_Y, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_z"][i], some_Number_Z, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_rx"][i], some_Number_RX, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_ry"][i], some_Number_RY, rel_tol = 0.1) & math.isclose(dataFrameTP["TCP_Current_rz"][i], some_Number_RZ, rel_tol = 0.1):
					dataFrameTP["Teach_Point"][i] = some_Number

		dataFramePath = dataFrameTP
		PointFrom = 999
		PointTo = 999
		dataFramePath["PointFrom"] = "PointFrom"
		dataFramePath["PointTo"] = "PointTo"
		dataFramePath["PATH"] = "PointFrom_PointTo_Iteration"

		for j in range(len(dataFramePath)):
			if dataFramePath["Teach_Point"][j] != 999:
				PointFrom = dataFramePath["Teach_Point"][j]
			dataFramePath["PointFrom"][j] = PointFrom

		for k in range(len(dataFramePath)-1, 1, -1):
			if dataFramePath["Teach_Point"][k] != 999:
				PointTo = dataFramePath["Teach_Point"][k]
			dataFramePath["PointTo"][k] = PointTo

		for l in range(len(dataFramePath)):
			if dataFramePath["PointFrom"][l] == dataFramePath["PointTo"][l]:
				dataFramePath["PATH"][l] = ("P"+str(dataFramePath["PointFrom"][l]))
			else:
				dataFramePath["PATH"][l] = ("P"+str(dataFramePath["PointFrom"][l])+"->P"+str(dataFramePath["PointTo"][l]))

		dataFramePath = dataFramePath.drop(["Teach_Point", "PointFrom", "PointTo"], axis=1)
		dataFramePath = dataFramePath[~dataFramePath["PATH"].str.contains("PPointTo", na=False)]
		dataFramePath = dataFramePath[~dataFramePath["PATH"].str.contains("P999", na=False)]
		result = dataFramePath
		return result