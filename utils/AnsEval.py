import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

device = "cuda:4" 
model = AutoModelForCausalLM.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2")

with open("../dataset/translated-LLaVA-Instruct-150K/detail_23k.json", 'r') as file:
    data = json.load(file)
# out_file = open("../ouput/AnsEval.json", 'w')
# tot_data = []

for it, _data in tqdm(enumerate(data)):
    
    if it>10: ##
        break

    # judge_prompt = "[Instruction]\nPlease act as an impartial judge and evaluate the relevance of two responses displayed below. Your evaluation should pay more attention to the meaning of response but not the grammar. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the the relevance of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Notice that rating 10 represents that the second response perfectly embodies the meaning of the first response and there is strictly no contradiction between two response. And the more contradictions between two responses, the lower the rating is. \n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
    # judge_prompt_relevance = "[Instruction]\nPlease act as an impartial judge and evaluate the relevance between two responses displayed below. Your should find out the important issues in the first response. And then judge whether the second response embodies these key issues which exist in the first one. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the relevance of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The less key issues two responses both have, the lower the rating is. The more contradictions two responses meet, the lower the rating is.\n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
    # judge_prompt_informativeness = "[Instruction]\nPlease act as an impartial judge and evaluate the informativeness between two responses displayed below. Your should find out important details in the first response. And then judge whether the second response embodies more details than the first one. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the informativeness of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The more details of second responses differ from the first one, the higher the rating is. If count and kind of information in two responses are the same, then rate 1.\n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
    judge_prompt_itself = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response displayed below. Your evaluation should focus on the understandability and inherent logicality of the response. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". \n\n[Response]\n{new_ans}\n\n"

    # original_ans = "The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. The image captures the casual, social atmosphere of a bustling plaza on a nice day."
    # new_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves. A man stands with crossed legs, holding a kite with multiple sections spread out, ready for flight. \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
    # new_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves.  \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
    # new_ans = "The scene shows people walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
    # new_ans = "The scene shows a sleeping dog."

    original_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves. A man stands with crossed legs, holding a kite with multiple sections spread out, ready for flight. \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
    new_ans = "The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. And there is a cute dog eating bones. The image captures the casual, social atmosphere of a bustling plaza on a nice day."
    

    # new_data = dict()
    # new_data["image"] = _data["image"]
    # question = _data["conversations"][0]["value"]
    # question_list = []
    # question_list.append(question)

    # compare_content = "1.{} \n 2.{} \n ".format(original_ans, new_ans)
    # judge_content = judge_prompt_informativeness.format(original_ans=original_ans, new_ans=new_ans)
    judge_content = judge_prompt_itself.format(new_ans=new_ans)
    # print(judge_content)
    messages = [
        {"role": "user", "content": judge_content},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    
    print(decoded[0])
    # line = decoded[0]
    # _ans = ""
    # for i,c in enumerate(line):
    #     if c!='[':
    #         continue
    #     t = line[i:i+7]
    #     if t=="[/INST]":
    #         _ans = line[i+7: -5]
    #         break

    # ans = _ans
    # for i,c in enumerate(_ans):
        # if c=='?':
            # ans = _ans[0: i+1]
    # print(ans)
    # question_list.append(ans)
    
    # new_data["questions"] = question_list
    # tot_data.append(new_data)
    break

# json.dump(tot_data, out_file)
# out_file.close()
