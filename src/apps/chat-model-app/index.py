from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

model = ChatOpenAI(model="gpt-4o-mini")
prompt = [HumanMessage("What is the capital of France?")]

result = model.invoke(prompt)
print(json.dumps(result.dict(), indent=2, ensure_ascii=False))
