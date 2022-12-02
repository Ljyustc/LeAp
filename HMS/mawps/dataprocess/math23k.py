# -*- encoding:utf-8 -*-

import json
import os
import re
from stanfordcorenlp import StanfordCoreNLP

from equ_tools import infix_to_postfix, postfix_to_prefix, post_solver, number_map, eval_num_list

def read_dataset(dataset_path):
    with open(dataset_path, "rt", encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset

def save_dataset(dataset, dataset_path):
    with open(dataset_path, "wt", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
    return

def split_text(text):
    seps = "，。．.；？！!"
    sep_pattern = re.compile(f"([{seps}])", re.S)
    spans = re.split(sep_pattern, text)
    spans = [span.strip() for span in spans if span.strip() != '']
    spans_post = []
    for i, span in enumerate(spans):
        if span in seps:
            if i > 0 and spans[i - 1] not in seps:
                spans_post[-1] += ' ' + span
        else:
            spans_post.append(span)
    return spans_post

def transfer_num(data):
    pattern = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    n_data = list()
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        n_num = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append(f"temp_{chr(n_num + ord('a'))}")
                n_num += 1
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        nums_fraction = []
        for num in nums:
            if re.search(r"\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        # seg the equation and tag the num
        def seg_and_tag(st):
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if n in nums:
                        res.append(f"temp_{chr(nums.index(n) + ord('a'))}")
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search(r"\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if st_num in nums:
                    res.append(f"temp_{chr(nums.index(st_num) + ord('a'))}")
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        postfix = infix_to_postfix(out_seq)
        f_nums = eval_num_list(nums)
        n_d = dict()
        n_d['id'] = d['id']
        n_d['equation'] = d['equation']
        n_d['text'] = ' '.join(input_seq)
        n_d['target_norm_post_template'] = ' '.join(['x', '='] + postfix)
        n_d['num_list'] = f_nums
        n_d["original_text"] = d["original_text"]
        n_data.append(n_d)
        if post_solver(number_map(postfix, f_nums)) is None:
            print(d['id'])
    return n_data

def num_process(in_path, out_path):
    data = read_dataset(in_path)
    data = transfer_num(data)
    save_dataset(data, out_path)
    return

def nlp_process(in_path, out_path):
    dataset = read_dataset(in_path)
    nlp = StanfordCoreNLP('C:/Software/StanfordNLP', lang='zh')
    for item in dataset:
        spans = split_text(item["text"])
        item["spans"] = [' '.join(nlp.word_tokenize(span)) for span in spans]
        item["dependency"] = [json.dumps(nlp.dependency_parse(span)) for span in spans]
    nlp.close()
    save_dataset(dataset, out_path)
    return

def equ_process(in_path, out_path):
    raw_dataset = read_dataset(in_path)
    dataset = list()
    for raw_item in raw_dataset:
        post = raw_item["target_norm_post_template"].split(' ')[2:]
        num_list = raw_item["num_list"]
        pre = postfix_to_prefix(post)
        answer = post_solver(number_map(post, num_list))[1]

        out_item = dict()
        out_item["id"] = raw_item["id"]
        out_item["original_text"] = raw_item["original_text"]
        out_item["equation"] = raw_item["equation"]
        out_item["text"] = raw_item["text"]
        out_item["answer"] = answer
        out_item["num_list"] = num_list
        out_item["target_norm_pre_template"] = ' '.join(["x", "="] + pre)
        out_item["spans"] = raw_item["spans"]
        out_item["dependency"] = raw_item["dependency"]
        dataset.append(out_item)
    save_dataset(dataset, out_path)
    return

if __name__ == "__main__":
    for label in ["train", "test", "valid"]:
        raw_path = os.path.join("data", f"{label}23k.json")
        path = os.path.join("data", f"{label}23k_processed.json")
        num_process(raw_path, path)
        nlp_process(path, path)
        equ_process(path, path)
