import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

device = "cuda:0" 
model = AutoModelForCausalLM.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2")
# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

with open("../dataset/translated-LLaVA-Instruct-150K/detail_23k.json", 'r') as file:
    data = json.load(file)
# print("json_length: ", len(data))
out_file = open("../dataset/MyDataset/PromptEval.json", 'w')
tot_data = []

for it, _data in tqdm(enumerate(data)):
    
    if it>1000:
        break

    # question = 'Describe the following image.'
    new_data = dict()
    new_data["image"] = _data["image"]
    question = _data["conversations"][0]["value"]
    question_list = []
    question_list.append(question)

    similar_content = "{} \n please give me another single one similar question sentence like the above sentence, and please just print out the question, do not attach unrelative statements".format(question)
    shorten_content = "{} \n please give me a shorter question than the above one, but the new question length should be shorter. And please just print out the question, do not attach unrelative statements".format(question)
    enlarge_content = "{} \n please give me a longer question than the above one without change original meaning, but the new question length should be shorter. And please just print out the question, do not attach unrelative statements".format(question)

    for _content in [similar_content, shorten_content, enlarge_content]:

        messages = [
            {"role": "user", "content": _content},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        
        line = decoded[0]
        _ans = ""
        for i,c in enumerate(line):
            if c!='[':
                continue
            t = line[i:i+7]
            if t=="[/INST]":
                _ans = line[i+7: -5]
                break

        ans = _ans
        for i,c in enumerate(_ans):
            if c=='?':
                ans = _ans[0: i+1]
        # print(ans)
        question_list.append(ans)
    
    new_data["questions"] = question_list
    tot_data.append(new_data)
    # break

json.dump(tot_data, out_file)
out_file.close()
