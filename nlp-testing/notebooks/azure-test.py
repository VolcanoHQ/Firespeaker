import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai-3dtechsolutionsai004073129672.openai.azure.com/openai/deployments/gpt-35-turbo"
model_name = "gpt-35-turbo"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential("567cdfbc4f3a4b5089fcd43d3096c194"),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?")
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=model_name
)

print(response.choices[0].message.content)