#!/usr/bin/env python2.7

from sqlalchemy import *

trainfile = "train.txt"

DATABASEURI = "postgresql://localhost/artdb"
engine = create_engine(DATABASEURI)
conn = engine.connect()

if __name__ == "__main__":
	f = open(trainfile, "r") 
	img_id = 0
	for line in f:
		info = line.split(" ")
		img_name = info[0]
		img_style = int(info[1])
		print "id: %d, name: %s, style: %d \n" % (img_id, img_name, img_style)
		conn.execute("INSERT INTO imgs VALUES (%s, %s, %s);" , (img_id, img_style, img_name))
		img_id += 1

	conn.close()		