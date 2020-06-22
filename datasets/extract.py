
"""
@author: Bohui Zhang

"""

import os
import sys
import json

def parse(input_dir):

    with open(input_dir, 'r') as f:
        functions = json.load(f)
    
    parsed_code = {}

    # test case
    #code = {c: functions[c] for c in list(functions)[:1]}
    #code = [c for c in list(functions.values())[:1]]
    code = list(functions.values())[0]
    #code = "public class Test {\n" + code + "}"
    count = 0
    
    for key, value in functions.items():
        #if value[:12] != "public class" or value[:21] != "public abstract class":
        
        # test first 100 functions since running the whole preprocessing takes time
        # or one can seperate the json file and running in multiple cmd
        if count == 100:
            break
        count += 1
        if value[0] != "\t" and value[0] != " ":
            code = "public class Train {\n\t" + value + "\n}"
        else:
            code = "public class Train {\n" + value + "\n}"
        parsed_code[key] = code
        file_name = "Train-" + key + ".java"
        java_path = os.path.join(os.getcwd(), "java", file_name)
        with open(java_path, 'x') as f:
            f.write(code)

if __name__ == "__main__":
    #cwd = os.getcwd()  # Get the current working directory (cwd)
    #print(cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))
    #path = "C:\Code\DR\datasets\\funcom_processed\\functions.json"
    path = sys.argv[1]
    parse(path)