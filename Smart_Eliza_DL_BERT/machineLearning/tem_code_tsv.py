import json
from pathlib import Path
from goemotions_configuration_constant import ESTIMATOR_PATH, PREDICT_DATA_DIR,VOL_PATH
import csv

sent = input('> ')

chat_file = Path(PREDICT_DATA_DIR, "chat_predict.tsv")
sent_list = sent.split('\n')
with open(chat_file, 'w', encoding='utf8', newline='') as csvfile:
    tsv_writer = csv.writer(csvfile, delimiter='\t',lineterminator='\n')
    tsv_writer.writerow(sent_list)
    csvfile.close()
exit()

# print("save to tsv file")