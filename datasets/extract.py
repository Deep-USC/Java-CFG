
"""
@author: Bohui Zhang

"""

import os
import sys
import json

def wrapper(input_dir):

    with open(input_dir, 'r') as f:
        functions = json.load(f)
    
    parsed_code = {}

    # test case
    #code = {c: functions[c] for c in list(functions)[:1]}
    #code = [c for c in list(functions.values())[:1]]
    #code = list(functions.values())[0]
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
        output_dir = os.path.join(os.getcwd(), "java", file_name)
        with open(output_dir, 'x') as f:
            f.write(code)

def to_deepwalk_adjlist(input_dir, output_dir): 
    """
    Extract and convert CFG in json to `.adjlist`.
    """

    with open(input_dir, 'r') as f:
        cfg = json.load(f)
    
    node_number = len(cfg['nodes'])
    adjlist = {}
    for i in range(node_number):
        adjlist[i] = [str(i)]
    for edge in cfg['edges']:
        adjlist[edge['source']].append(str(edge['target']))
        adjlist[edge['target']].append(str(edge['source']))

    seperator = " "
    with open(output_dir, 'x') as f:
        for i in range(node_number):
            f.write(seperator.join(adjlist[i]))
            f.write("\n")

def to_graph2seq(input_dir):
    """
    Extract and convert CFG in json to .data`.
    """

    with open(input_dir, 'r') as f:
        cfg = json.load(f)

    ...

if __name__ == "__main__":
    #cwd = os.getcwd()  # Get the current working directory (cwd)
    #print(cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))
    #input_path = "C:\Code\DR\DeepUSC\datasets\Test1-CFG.json"
    #output_path = "C:\Code\DR\DeepUSC\datasets\Test1-CFG.adjlist"
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    #wrapper(input_path)
    to_deepwalk_adjlist(input_path, output_path)