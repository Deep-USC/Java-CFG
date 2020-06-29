
"""
@author: Bohui Zhang

"""

import os
import sys
import json
import argparse

def wrapper(input_dir):
    """
    
    :param input_dir: the input file path
    :return:
    """

    with open(input_dir, 'r') as f:
        functions = json.load(f)
    
    parsed_code = {}

    count = 0
    
    for key, value in functions.items():
        
        # TODO: accelerate processing functions.json
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

def to_adjlist(input_dir, output_dir): 
    """
    Extract and convert CFG in json to `.adjlist`.

    :param input_dir: the input file path
    :param output_dir: the output file path
    :return:
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

def to_edgelist(input_dir, output_dir):
    """
    
    :param input_dir: the input file path
    :param output_dir: the output file path
    :return: 
    """

    with open(input_dir, 'r') as f:
        cfg = json.load(f)

    edgelist = []

    for edge in cfg['edges']:
        edgelist.append([str(edge['source']), str(edge['target'])])
    
    seperator = " "
    with open(output_dir, 'x') as f:
        for edge in edgelist:
            f.write(seperator.join(edge))
            f.write("\n")

def to_graph2seq(input_dir):
    """
    Extract and convert CFG in json to .data`.

    :param input_dir: the input file path
    :return:
    """

    with open(input_dir, 'r') as f:
        cfg = json.load(f)

    ...

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wrapper")
    parser.add_argument("-f", "--format", dest="format", required=True)
    parser.add_argument("-i", "--input", dest="input_dir", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", required=True)

    args = parser.parse_args()

    format_func = {
        "adjlist": to_adjlist,
        "edgelist": to_edgelist
    }

    if args.wrapper:
        wrapper(args.input_dir)
    
    if args.format:
        format_func[args.format](args.input_dir, args.output_dir)
    