
'''
Requirements
- support image, text, functools etc as input

Questions
- Do we expose incoming audio as a track or queue, or just a callback?

Audio is forwarded like this atm in the kickboxing_example

@ai_connection.on("audio")
async def on_audio(pcm: PcmData, user):
    if user.user_id == player_user_id and g_session:
        await g_session.send_realtime_input(
            audio=types.Blob(
                data=pcm.samples.astype(np.int16).tobytes(),
                mime_type="audio/pcm;rate=48000"
            )
        )

In the kickboxing example it does this for playout

audio_in_queue.put_nowait(data)

asyncio.create_task(play_audio(audio_in_queue, audio_track))
async def play_audio(audio_in_queue, audio_track):
    """Play audio responses from Gemini Live"""
    while True:
        bytestream = await audio_in_queue.get()
        await audio_track.write(bytestream)

'''

class LLM:
    sts: bool = False

    def create_response(self, *args, **kwargs):
        # Follow openAI style response?
        pass

class RealtimeLLM(LLM):
    '''
    We need to standardize connect and attaching audio in/ audio out and video in
    '''
    sts : bool = True

    def connect(self):
        pass

    def attach_incoming_audio(self, track):
        pass

    def attach_incoming_video(self, track):
        pass

    def attach_outgoing_audio(self, track):
        pass

