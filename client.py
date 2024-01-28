import asyncio
import websockets
import json, io
from PIL import Image

def compress_image(image_path, quality=85):
    image = Image.open(image_path)
    compress_image_stream = io.BytesIO()
    image.save(compress_image_stream, format="JPEG", quality=quality)
    compress_image_bytes = compress_image_stream.getvalue()
    return compress_image_bytes


async def recognize_face():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websockets:
        compressed_image_bytes = compress_image("unknown.jpeg")
        await websockets.send(compressed_image_bytes)
        response = await websockets.recv()
        return json.loads(response)


async def main():
    try:
        response = await recognize_face()
        print(response)
    except Exception as err:
        print('Error: '+str(err))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

