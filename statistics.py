import torch
import json
import re
from tqdm import tqdm

def make_dict():
    d = dict()
    for i in range(11):
        d[i] = 0
    return d

def make_statistics(d):
    sum1 = 0.
    cnt = 0.
    for i in range(11):
        sum1 += (i*d[i])
        cnt += d[i]
    d["Avg"] = round(sum1/cnt,3)
    
    sum2 = 0.
    for i in range(11):
        sum2 += pow((i-d["Avg"]),2)*d[i]
    d["Var"] = round(sum2/cnt,3)

model_name_list = ["llava-v1.6-mistral-7b", "llava-v1.6-vicuna-7b", 
                   "instructblip-vicuna-7b", "instructblip-flan-t5-xl",
                   "blip2-flan-t5-xl"]
# model_name_list = ["llava-v1.6-mistral-7b"]

prompt_str_list = ["similar", "shorter", "enlarge"]
img_str_list = ["complex", "blur", "noise"]

out_file = open("./output/total.json", 'w')
tot_data = dict()
for model_name in tqdm(model_name_list):
    model_dict = dict()
    with open("./output/Rating/{}.json".format(model_name), 'r') as file:
        data_rating = json.load(file)

    for i in [0,1,2]:
        prompt_relevance = make_dict()
        prompt_infor = make_dict()
        prompt_self = make_dict()
        img_relevance = make_dict()
        img_infor = make_dict()
        img_self = make_dict()
        for it, _data_rating in enumerate(data_rating):
            prompt_rate = _data_rating["PromptEval"][i]
            img_rate = _data_rating["ImageEval"][i]
            
            prompt_relevance[prompt_rate["Relevance"]] += 1
            prompt_infor[prompt_rate["Informativeness"]] += 1
            prompt_self[prompt_rate["Itself"]] += 1
            
            img_relevance[img_rate["Relevance"]] += 1
            img_infor[img_rate["Informativeness"]] += 1
            img_self[img_rate["Itself"]] += 1
        
        make_statistics(prompt_relevance)
        make_statistics(prompt_infor)
        make_statistics(prompt_self)

        make_statistics(img_relevance)
        make_statistics(img_infor)
        make_statistics(img_self)
        
        model_dict[prompt_str_list[i]] = {"Relevance": prompt_relevance,
                                          "Informativeness": prompt_infor,
                                          "Itself":prompt_self}
        model_dict[img_str_list[i]] = {"Relevance": img_relevance,
                                        "Informativeness": img_infor,
                                        "Itself":img_self}

    tot_data[model_name] = model_dict
        

json.dump(tot_data, out_file)
out_file.close()


