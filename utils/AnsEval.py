import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import re

def SplitQuestion(s):
    for j,c in enumerate(s):
        if c!='[':
            continue
        t = s[j:j+7]
        if t=="[/INST]":
            return s[j+7:]
    return s

model_name_list = ["llava-v1.6-mistral-7b", "llava-v1.6-vicuna-7b", 
                   "instructblip-vicuna-7b", "instructblip-flan-t5-xl",
                   "blip2-flan-t5-xl"]
# model_name_list = ["llava-v1.6-mistral-7b"]

device = "cuda:0" 
model = AutoModelForCausalLM.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("../premodel/Mistral-7B-Instruct-v0.2")

# judge_prompt_relevance = "[Instruction]\nPlease act as an impartial judge and evaluate the similarity of meaning between two responses displayed below. Your should find out the keywords both in the first response and the second response. Notice that in this part you do not need to care about the number of keywords in the first response, for the first response probabaly is not a complete sentence. And then judge whether the second response embodies these keywords which exist in the first one. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the relevance of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The less key issues two responses both have, the lower the rating is. The more contradictions two responses meet, the lower the rating is. If it is so hard for you that can not give specific rating, then give rating of 0 in same format. \n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
judge_prompt_relevance = "[Instruction]\nPlease act as an impartial judge and evaluate the similarity of meaning between two responses displayed below. Your should find out the keywords both in the first response and the second response. And then judge whether the second response embodies keywords in the first response. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the relevance of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The less keywords in the first response the second response simultaneously has, the lower the rating is. The more contradictions of keywords two responses meet, the lower the rating is. If it is so hard for you that can not give specific rating, then give rating of 0 in same format. \n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
# judge_prompt_informativeness = "[Instruction]\nPlease act as an impartial judge and evaluate the difference of information between two responses displayed below. Your should find out keywords and important details in the first response and the second response respectively. And then judge whether the second response embodies more abundant details and keywords than the first one. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the informativeness of these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The more abundant details and keywords of second responses are, the higher the rating is. If the details and keywords in two responses are of the same abundance, then rate 1. If it is so hard for you that can not give specific rating, then give rating of 0 in same format.\n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
judge_prompt_informativeness = "[Instruction]\nPlease act as an impartial judge and evaluate the difference in information abundance between two responses displayed below. Your should find out keywords and details in the first response and the second response respectively. And then judge whether the second response have more abundant details and keywords than the first one. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the the difference in information abundance between these two responses on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". The more abundant the second response's information is, the higher the rating is. If the meaning of the first response is more abundant than the second response, then rate the lowest score 0. If there is no difference in information abundance between two responses displayed below, then rate 1. If it is so hard for you that can not give specific rating, then give rating of 0.\n\n[First response]\n{original_ans}\n\n[Second response]\n{new_ans}\n\n"
judge_prompt_itself = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response displayed below. Your evaluation should focus on the understandability and inherent logicality of the response. Begin your evaluation by providing a short explanation. Be as objective and careful as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". If it is so hard for you that can not give specific rating, then give rating of 0 in same format. \n\n[Response]\n{new_ans}\n\n"
judge_prompt_list = [judge_prompt_relevance, judge_prompt_informativeness, judge_prompt_itself]
prompt_str_list = ["Relevance", "Informativeness", "Itself"]

for model_name in model_name_list:
    with open("../output/prompt/{}.json".format(model_name), 'r') as file:
        data_prompt = json.load(file)
    with open("../output/image/{}.json".format(model_name), 'r') as file:
        data_img = json.load(file)
    # out_file_prompt = open("../ouput/AnsEval_prompt.json", 'w')
    # out_file_img = open("../ouput/AnsEval_image.json", 'w')
    out_file_eval = open("../output/AnsEval/{}.json".format(model_name), 'w')
    out_file_rating = open("../output/Rating/{}.json".format(model_name), 'w')
    tot_data_eval = []
    tot_data_rating = []

    for it, (_data_prompt, _data_img) in tqdm(enumerate(zip(data_prompt, data_img)), total=len(data_img)):
        # if it>10: ##
        #     break
        new_data_eval = dict()
        new_data_eval["image"] = _data_img["image"]
        new_data_eval["PromptEval"] = []
        new_data_eval["ImageEval"] = []

        new_data_rating = dict()
        new_data_rating["image"] = _data_img["image"]
        new_data_rating["PromptEval"] = []
        new_data_rating["ImageEval"] = []
        # original_ans = "The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. The image captures the casual, social atmosphere of a bustling plaza on a nice day."
        # new_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves. A man stands with crossed legs, holding a kite with multiple sections spread out, ready for flight. \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
        # new_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves.  \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
        # new_ans = "The scene shows people walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
        # new_ans = "The scene shows a sleeping dog."
        # original_ans = "The scene shows a vibrant plaza with people walking and enjoying themselves. A man stands with crossed legs, holding a kite with multiple sections spread out, ready for flight. \n\nPeople walk and interact with each other, some carrying handbags and others with backpacks. The image captures the lively, social atmosphere of a bustling plaza on a sunny day."
        # new_ans = "The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. And there is a cute dog eating bones. The image captures the casual, social atmosphere of a bustling plaza on a nice day."
        
        original_ans =_data_img["answers"][0]
        for i in [1, 2, 3]: # evaluate image test
            new_ans = _data_img["answers"][i]
            if model_name=="llava-v1.6-mistral-7b" or model_name == "llava-v1.6-vicuna-7b":
                new_ans = SplitQuestion(new_ans)
            eval_dict = dict()
            rating_dict = dict()
            for (judge_prompt, prompt_str) in zip(judge_prompt_list, prompt_str_list):
                if judge_prompt==judge_prompt_itself:
                    judge_content = judge_prompt.format(new_ans=new_ans)
                else:
                    judge_content = judge_prompt.format(original_ans=original_ans, new_ans=new_ans)
                
                messages = [
                    {"role": "user", "content": judge_content},
                ]
                
                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to(device)
                model.to(device)
                generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                eval_dict[prompt_str] = decoded[0]
                
                _ans = SplitQuestion(decoded[0])

                _rating = 0
                pattern = r"Rating:"
                matches = re.finditer(pattern, _ans)
                positions = [match.start() for match in matches]
                for pos in positions:
                    if _rating>0:
                        break
                    for ic, c in enumerate(_ans[pos+7: min(len(_ans)+1, pos+14)]):
                        if c>='0' and c<='9':
                            if c=='1' and _ans[pos+8+ic]=='0':
                                _rating = 10
                            else:
                                _rating = int(c)
                            break
                rating_dict[prompt_str] = _rating
            
            new_data_eval["ImageEval"].append(eval_dict)
            new_data_rating["ImageEval"].append(rating_dict)
        

        for i in [0, 1, 2]: # evaluate prompt test
            new_ans = _data_prompt["answers"][i]
            if model_name=="llava-v1.6-mistral-7b" or model_name == "llava-v1.6-vicuna-7b":
                new_ans = SplitQuestion(new_ans)
            eval_dict = dict()
            rating_dict = dict()
            for (judge_prompt, prompt_str) in zip(judge_prompt_list, prompt_str_list):
                if judge_prompt==judge_prompt_itself:
                    judge_content = judge_prompt.format(new_ans=new_ans)
                else:
                    judge_content = judge_prompt.format(original_ans=original_ans, new_ans=new_ans)
                
                messages = [
                    {"role": "user", "content": judge_content},
                ]

                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to(device)
                model.to(device)
                generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                eval_dict[prompt_str] = decoded[0]
                
                _ans = SplitQuestion(decoded[0])
                
                _rating = 0
                pattern = r"Rating:"
                matches = re.finditer(pattern, _ans)
                positions = [match.start() for match in matches]
                for pos in positions:
                    if _rating>0:
                        break
                    for ic, c in enumerate(_ans[pos+7: min(len(_ans)+1, pos+14)]):
                        if c>='0' and c<='9':
                            if c=='1' and _ans[pos+8+ic]=='0':
                                _rating = 10
                            else:
                                _rating = int(c)
                            break
                rating_dict[prompt_str] = _rating
            
            new_data_eval["PromptEval"].append(eval_dict)
            new_data_rating["PromptEval"].append(rating_dict)
                
        tot_data_eval.append(new_data_eval)
        tot_data_rating.append(new_data_rating)
        
        # break ##

    json.dump(tot_data_eval, out_file_eval)
    out_file_eval.close()
    json.dump(tot_data_rating, out_file_rating)
    out_file_rating.close()
