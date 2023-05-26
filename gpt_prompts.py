import openai
import os

# Set up your OpenAI API credentials
openai.api_key = os.getenv('OPEN_AI_KEY')

# Define the prompt
PROMPT = f'''Generate a prompt using the following template:
"a photo of a bluhrboi wearing [attire]"
Where [attire] can be replaced with a comma denoted list of "[color] [item]", where "[item]" is a piece of attire:'''

class SingleImagePipeline():
    def __init__():


# Call the OpenAI API to generate responses
def generate_prompts(N: int) -> list[str]:
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100,
        n=N,
        stop=None,
        temperature=0.7
    )

    prompts = [r.text.strip() for r in response.choices]

    return prompts
    # Extract and print the generated prompts

def 
