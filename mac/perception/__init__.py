"""VoiceLink perception layer (Mac).

Connects to the wearable's WebSocket as a client, runs Whisper on the audio
stream and a gated VLM on the frame stream concurrently, and writes everything
into the shared ContextState. It never calls rank_vocabulary or the UI.
"""
