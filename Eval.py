from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import os
import json
from tqdm import tqdm

model_name_list = ["llava-v1.6-mistral-7b", "llava-v1.6-vicuna-7b", 
                   "instructblip-vicuna-7b", "instructblip-flan-t5-xl",
                   "blip2-flan-t5-xl"]
for model_name in model_name_list:
    # model_name = "llava-v1.6-mistral-7b"
    if model_name == "llava-v1.6-mistral-7b": ###
        continue

    img_path_pre = "./dataset/translated-LLaVA-Instruct-150K/llava-imgs/filtered-llava-images"
    with open("./dataset/MyDataset/PromptEval.json", 'r') as file:
        data0 = json.load(file)
    out_file = open("./output/prompt/{}.json".format(model_name), 'w')
    tot_data = []

    model_path = os.path.join("./premodel", model_name)
    if model_name=="llava-v1.6-mistral-7b":
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda")
        
        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            new_data = dict()
            new_data["image"] = _data["image"]
            ans_list = []

            img_path = os.path.join(img_path_pre, _data["image"])
            image = Image.open(img_path)
            for i in [1,2,3]:
                prompt = "[INST] <image>\n{} [/INST]".format(_data["questions"][i])
                inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
                output = model.generate(**inputs, max_new_tokens=100)

                ans = processor.decode(output[0], skip_special_tokens=True)
                # print(i, " >> ", ans)
                ans_list.append(ans)
            new_data["answers"] = ans_list
            tot_data.append(new_data)
            # break

    elif model_name == "llava-v1.6-vicuna-7b":
        continue ##
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda")

        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            new_data = dict()
            new_data["image"] = _data["image"]
            ans_list = []

            img_path = os.path.join(img_path_pre, _data["image"])
            image = Image.open(img_path)
            for i in [1,2,3]:
                # prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{} ASSISTANT:".format(_data["questions"][i])
                prompt = "USER: <image>\n{} ASSISTANT:".format(_data["questions"][i])
                inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
                output = model.generate(**inputs, max_new_tokens=100)

                ans = processor.decode(output[0], skip_special_tokens=True)
                ans_list.append(ans)

            new_data["answers"] = ans_list
            tot_data.append(new_data)
            # break
    
    elif model_name == "instructblip-vicuna-7b" or model_name == "instructblip-flan-t5-xl":
        # continue ##
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = InstructBlipProcessor.from_pretrained(model_path)
        model.to("cuda")

        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            new_data = dict()
            new_data["image"] = _data["image"]
            ans_list = []

            img_path = os.path.join(img_path_pre, _data["image"])
            image = Image.open(img_path)
            for i in [1,2,3]:
                # prompt = "USER: <image>\n{} ASSISTANT:".format(_data["questions"][i])
                prompt = _data["questions"][i]
                if len(prompt) > 1000:
                    prompt = prompt[:1000]
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
                ans = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                ans_list.append(ans)

            new_data["answers"] = ans_list
            tot_data.append(new_data)
            # break
    
    elif model_name == "blip2-flan-t5-xl":
        # continue ##
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to("cuda")

        for it, _data in tqdm(enumerate(data0), total=len(data0)):
            new_data = dict()
            new_data["image"] = _data["image"]
            ans_list = []

            img_path = os.path.join(img_path_pre, _data["image"])
            image = Image.open(img_path)
            for i in [1,2,3]:
                prompt = _data["questions"][i]
                # print(" >> ", len(prompt))
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
                generated_ids = model.generate(**inputs)
                ans = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                ans_list.append(ans)
            new_data["answers"] = ans_list
            tot_data.append(new_data)
            # break

    json.dump(tot_data, out_file)
    out_file.close()
    # break
