from openai import OpenAI

# Here was my first test of the API call, calling a haiku for fun
# client = OpenAI()
#
# response = client.responses.create(
#   model="gpt-5-nano",
#   input="write a haiku about the power of AI to change a small company",
#   store=True,
# )
#
# print(response.output_text);

# Now for the actual program code

client = OpenAI() # This is basically saying to the AI “Look in the environment for the key.” to the api

response = client.responses.create(
  model="gpt-5-mini",
  input="Read the text and createa a part description and json format containing the full specs of this hardware part.",
  store=True,
)

print(response.output_text);