import os

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

import getpass
import os


from diffusers import AutoPipelineForText2Image
import torch

YAML_CONTENT = """
models:
  - type: main
    engine: openai
    model: gpt-4o
"""

nsfw_file = open("data/nsfw_prompts.txt", "r")
nsfw_txt = ""
for idx, nsfw in enumerate(nsfw_file):
    nsfw_txt += f'''  "{nsfw.strip()}"\n'''


COLANG_CONTENT = f"""
define user express bad prompt
{nsfw_txt}

define bot express error message
  "Error: Bad Input Theme!"

define flow
  user express bad prompt
  bot express error message
"""



def prompt_generator(model, theme):
    rails_config = RailsConfig.from_content(
        yaml_content=YAML_CONTENT, colang_content=COLANG_CONTENT
    )
    guardrails = RunnableRails(config=rails_config)
    model_with_rails = guardrails | model

    # Invoke the chain using the model with rails.
    prompt = ChatPromptTemplate.from_template("a sentence about {theme}.")
    output_parser = StrOutputParser()
    chain = prompt | model_with_rails | output_parser

    output = chain.invoke({"theme": theme})
    return output

def sdxl_text_to_image(prompt):
  pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
  ).to("cuda:4")
  image = pipeline_text2image(prompt=prompt).images[0]
  return image



if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    theme = getpass.getpass("Enter the theme of the image you want to create: ")
    print("-"*50)
    model = ChatOpenAI()
    prompt = prompt_generator(model, theme)
    print("-"*50)
    if prompt=="Error: Bad Input Theme!":
      print(prompt)
    else:
      print("Text Prompt for SDXL:",prompt)
      output_dir = "output_image"
      os.makedirs(output_dir, exist_ok=True)

      output_image = sdxl_text_to_image(prompt)
      image_name = theme+".png"
      output_path = os.path.join(output_dir,image_name)
      output_image.save(output_path)