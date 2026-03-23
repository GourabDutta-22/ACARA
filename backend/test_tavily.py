from agent import search_tool
print("Invoking search...")
res = search_tool.invoke({"query": "What is Apple?"})
print("Result:", res)
