# from call_gpt import send_chat_request
# import time
#
# # 定义要测试的引擎列表
# engines = ["AT:70b-at", "GPT4"]
#
#
# temperatures = [0.7] * len(engines)
#
# # 共用的系统消息和问题
# system_message = "You are a helpful assistant."
# examples = []
# question = "帮我计算88*89"
#
# # 遍历每个引擎和温度组合并执行测试
# for engine, temperature in zip(engines, temperatures):
#     # 记录开始时间
#     start_time = time.time()
#
#     response = send_chat_request(
#         system_message,
#         examples,
#         question,
#         engine=engine,
#         temperature=temperature,
#         max_retry=1,
#         at_url='https://api.openai.com/v1/embeddings'
#     )
#
#     # 记录结束时间
#     end_time = time.time()
#
#     # 计算执行时间
#     execution_time = end_time - start_time
#
#     print(f"{engine} response:", response["response"])
#     print(f"Execution time for {engine}: {execution_time} seconds")
#     print("----------")
print(30 / 36 * 100)
