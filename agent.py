from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, WikipediaSearchTool

model = LiteLLMModel(
    model_id="lm_studio/qwen2.5-1.5b-instruct",
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    num_ctx=8192,
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool(), WikipediaSearchTool()], model=model)

result = agent.run("What is the cockfight find through wikipedia?")
print(result)
