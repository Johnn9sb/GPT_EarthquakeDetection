import os
import sys
from openai import OpenAI
import base64
import requests
from datetime import datetime
from apikey import get_api

key = get_api()

client = OpenAI(
    api_key = key
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_upload(
        prompt_eq1,
        prompt_eq2,
        prompt_no1,
        prompt_no2,
        test,
):
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text": "Among the 5 waveform pictures I will give you next, the 1st and 2nd waveform pictures belong to the first category, and the 3rd and 4th waveform pictures belong to the second category. Please help me classify the 5th waveform based on the above pictures. Which category does the picture belong to? And when answering, you only need to answer 1 or 2, which means belonging to the first category or the second category respectively."
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{prompt_eq1}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{prompt_eq2}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{prompt_no1}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{prompt_no2}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{test}",
                            "detail": "high"
                        }
                    },
                ],
            },
        ],
        
    )
    print(completion)
    return completion.choices[0].message.content

test_type = 'eq'  # eq, noise
test_num = '3rd'


eq1_path = "/mnt/nas5/johnn9/gptpicker/image/eqsyn_prom/3rd/wav_227.jpg"
eq2_path = "/mnt/nas5/johnn9/gptpicker/image/eqsyn_prom/3rd/wav_228.jpg"
no1_path = "/mnt/nas5/johnn9/gptpicker/image/eqno_prom/3rd/wav_111.jpg"
no2_path = "/mnt/nas5/johnn9/gptpicker/image/eqno_prom/3rd/wav_117.jpg"
eq1 = encode_image(eq1_path)
eq2 = encode_image(eq2_path)
no1 = encode_image(no1_path)
no2 = encode_image(no2_path)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if test_type == 'eq':
    test_path = "/mnt/nas5/johnn9/gptpicker/image/eqsyn_test"
    answer_path = f"./answer/test_eq_{current_time}.txt" 
elif test_type == 'noise':
    test_path = "/mnt/nas5/johnn9/gptpicker/image/eqno_test"
    answer_path = f"./answer/{test_num}_test_noise_{current_time}.txt" 

with open(answer_path, 'a', encoding='utf-8') as answer_txt:

    for testname in os.listdir(test_path):
        test_name = os.path.join(test_path, testname)
        test = encode_image(test_name)
        answer = chat_upload(eq1, eq2, no1, no2, test)
        answer_txt.write(test_name + '\n' + answer + '\n')

print("Succesfully!!!")

sys.exit()