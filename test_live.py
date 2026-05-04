import sys
sys.path.append(".")
try:
    from src.retrieval.live_web_search import LiveWebSearchRetriever
    lws = LiveWebSearchRetriever()
    res, _ = lws.multi_retrieve(["varicose veins treatments"])
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
