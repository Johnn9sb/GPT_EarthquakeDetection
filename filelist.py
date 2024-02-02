import os
import sys   

test_path = "/mnt/nas5/johnn9/gptpicker/image/eqsyn_test"
list_path = "./answer/test_list.txt" 
with open(list_path, 'a', encoding='utf-8') as list_txt:
    for testname in os.listdir(test_path):
        test = os.path.join(test_path, testname)
        list_txt.write(test + '\n')