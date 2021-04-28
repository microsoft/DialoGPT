# A 3rd party demo contributed by Github user AK391 (https://github.com/AK391). This is not implemented by Microsoft and Microsoft do not own any IP with this implementation and associated demo. 
# Microsoft has not tested the generation of this demo and is not responsible for any offensive or biased generation from this demo. 
# Please contact the creator AK391 (https://github.com/AK391) for any potential issue. 


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def dialogpt(text):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    for step in range(50000):

        new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

inputs = gr.inputs.Textbox(lines=1, label="Input Text")
outputs =  gr.outputs.Textbox(label="DialoGPT")

title = "DialoGPT"
description = "demo for Microsoft DialoGPT with Hugging Face transformers. To use it, simply input text or click one of the examples text to load them. Read more at the links below. *This is not a Microsoft product and is developed for Gradio*"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1911.00536'>DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation</a> | <a href='https://github.com/microsoft/DialoGPT'>Github Repo</a> | <a href='https://huggingface.co/microsoft/DialoGPT-large'>Hugging Face DialoGPT-large</a></p>"
examples = [
            ["Hi, how are you?"],
            ["How far away is the moon?"],
]

gr.Interface(dialogpt, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)
