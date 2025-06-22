import json
import os
import asyncio
import edge_tts
from pydub import AudioSegment

# --- Voice Configuration for Edge-TTS ---
# High-quality, distinct voices from Microsoft's service.
ALEX_VOICE = "en-US-JennyNeural"  # A standard female voice
JAMIE_VOICE = "en-US-GuyNeural"    # A standard male voice
# For other voice options, see: edge-tts --list-voices

async def communicate_tps(text, voice, path):
    """
    Generates audio for a given text and voice using edge-tts.
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

async def generate_audio_from_dialog(json_path, output_dir, final_filename="final_conversation.mp3"):
    """
    Reads a conversation JSON, generates audio for each utterance using
    edge-tts for distinct male/female voices, and combines them into a single MP3 file.
    """
    # Ensure the output directory and a temp directory for clips exist
    temp_dir = os.path.join(output_dir, "temp_clips")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Audio files will be saved in: {output_dir}")

    # Load the conversation data
    try:
        with open(json_path, 'r') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}.")
        return

    print("\nGenerating individual audio clips using Edge-TTS...")
    tasks = []
    clip_paths = []

    # Create all generation tasks first
    for convo in conversations:
        frame_index = convo.get("frame_index")
        dialog = convo.get("dialog", [])

        if not isinstance(dialog, list):
            continue

        for i, turn in enumerate(dialog):
            speaker = turn.get("speaker", "Unknown").strip()
            utterance = turn.get("utterance", "").strip()

            if not utterance:
                continue

            voice = ALEX_VOICE if speaker == "Alex" else JAMIE_VOICE
            temp_path = os.path.join(temp_dir, f"frame_{frame_index}_turn_{i}_{speaker}.mp3")
            
            # Add the generation task to the list
            tasks.append(communicate_tps(utterance, voice, temp_path))
            clip_paths.append(temp_path)
            print(f"  - Queued: {os.path.basename(temp_path)}")

    # Run all audio generation tasks concurrently
    await asyncio.gather(*tasks)
    print("\nAll clips generated.")

    # Combine all audio clips using pydub
    print("\nCombining audio clips into a single file...")
    pause = AudioSegment.silent(duration=500) # 0.5-second pause
    final_audio = AudioSegment.empty()

    for path in clip_paths:
        try:
            clip = AudioSegment.from_mp3(path)
            final_audio += clip + pause
        except Exception as e:
            print(f"Could not process clip {path}: {e}")

    if not final_audio:
        print("No audio clips were successfully combined. Exiting.")
        return

    # Export the final combined audio file
    final_path = os.path.join(output_dir, final_filename)
    final_audio.export(final_path, format="mp3")
    print(f"\n✅ Final conversation audio saved to: {final_path}")

    # Clean up temporary files
    print("Cleaning up temporary clips...")
    for file_name in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file_name))
    os.rmdir(temp_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
    # Define the path to your conversations file and the desired output directory
    conversation_file = "./video_frames_output/all_conversations.json"
    audio_output_folder = "./generated_audio"
    
    # Run the asynchronous function
    asyncio.run(generate_audio_from_dialog(conversation_file, audio_output_folder)) 