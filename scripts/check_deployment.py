"""Check HTTPS and the real Streamlit WebSocket session after deployment."""
import asyncio
import sys
import urllib.request

from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.asyncio.client import connect

url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(url + "/_stcore/health", timeout=30) as response:
    assert response.status == 200
    print("HTTPS health:", response.read().decode())
with urllib.request.urlopen(url, timeout=30) as response:
    assert response.status == 200
    assert b"streamlit" in response.read().lower()


async def main():
    address = url.replace("https://", "wss://").replace("http://", "ws://") + "/_stcore/stream"
    async with connect(address, subprotocols=["streamlit"], origin=url, open_timeout=60, max_size=8_000_000) as socket:
        request = BackMsg()
        request.rerun_script.query_string = ""
        await socket.send(request.SerializeToString())
        metrics = []
        async with asyncio.timeout(120):
            async for data in socket:
                if isinstance(data, str):
                    continue
                message = ForwardMsg()
                message.ParseFromString(data)
                kind = message.WhichOneof("type")
                if kind == "delta" and message.delta.HasField("new_element"):
                    element = message.delta.new_element
                    if element.HasField("exception"):
                        raise AssertionError(str(element.exception))
                    if element.HasField("metric"):
                        metrics.append(element.metric.body)
                if kind == "script_finished":
                    assert metrics, "No model prediction rendered in the live session"
                    print("WebSocket and live model prediction:", metrics)
                    return
        raise AssertionError("Streamlit session did not finish")


asyncio.run(main())
