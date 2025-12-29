from langchain_openai.llms import OpenAI

model = OpenAI(model="gpt-4o-mini")

result = model.invoke("What is the capital of Japan?")
print(result)
