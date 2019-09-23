from pydub import AudioSegment
import os
import pickle

with open('bangla.pickle', 'rb') as f:
    data = pickle.load(f)

words = [
	'ব্যক্তি',
	'সাইকেল',
	'৪',
	'মোটরসাইকেল',
	'সামনে বর্ণনা এর কিছু নেই।',
	'চুলা',
	'গাড়ী',
	'বিমান'
]

line = AudioSegment.silent(1)
for word in words:
    line += data[word]

# writing mp3 files is a one liner
line.export('out.mp3', format="mp3")

os.system("mpg123 out.mp3")