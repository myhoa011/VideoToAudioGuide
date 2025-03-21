import asyncio
async def call_api(openai_client, output_path, model, voice, text):
    response = await asyncio.to_thread(
        openai_client.audio.speech.create,
        model=model,
        voice=voice,
        input=text,
        response_format="wav"
    )
    
    await asyncio.to_thread(response.stream_to_file, output_path)
