import json
import os
import shutil
import asyncio
import edge_tts

# --- Voice Configuration ---
# Using the same high-quality voices as before for consistency.
ALEX_VOICE = "en-US-GuyNeural"
JAMIE_VOICE = "en-US-JennyNeural"

async def _generate_clip(text, voice, path):
    """Helper to generate a single audio clip."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)
    except Exception as e:
        print(f"Failed to generate clip for text '{text[:30]}...'. Error: {e}")

async def create_video_package(
    conversations_path,
    avatar_dir,
    output_package_dir,
    alex_avatar_name="alex_avatar.jpg",
    jamie_avatar_name="jamie_avatar.jpg",
):
    """
    Organizes all necessary assets into a single package for video generation.

    This function reads a conversation JSON, re-generates the individual audio clips,
    pairs them with the correct avatar image, and copies all files into a
    self-contained 'package' directory. It also creates a manifest.json
    to orchestrate the video generation process.
    """
    print(f"Starting video package creation...")

    # --- 1. Setup Directories ---
    audio_output_dir = os.path.join(output_package_dir, "audio_clips")
    os.makedirs(audio_output_dir, exist_ok=True)
    print(f"Package will be created at: {output_package_dir}")

    # --- 2. Verify and Copy Avatars ---
    alex_avatar_path = os.path.join(avatar_dir, alex_avatar_name)
    jamie_avatar_path = os.path.join(avatar_dir, jamie_avatar_name)

    if not os.path.exists(alex_avatar_path) or not os.path.exists(jamie_avatar_path):
        print(f"\n--- ACTION REQUIRED ---")
        print(f"Error: Avatar images not found.")
        print(f"Please create two files named '{alex_avatar_name}' and '{jamie_avatar_name}'")
        print(f"and place them in the '{avatar_dir}' directory before running this script.")
        print(f"-----------------------\n")
        return

    shutil.copy(alex_avatar_path, os.path.join(output_package_dir, alex_avatar_name))
    shutil.copy(jamie_avatar_path, os.path.join(output_package_dir, jamie_avatar_name))
    print("Avatars verified and copied.")

    # --- 3. Load Conversation Data ---
    try:
        with open(conversations_path, 'r') as f:
            data = json.load(f)

        # --- Data Transformation ---
        # The script expects a list of conversations, but the JSON is a single object.
        # We'll extract the scene analysis and transform it into the expected format.
        scene_analysis = data.get("scene_analysis", "")
        dialog = []
        for line in scene_analysis.strip().split('\n\n'):
            if ':' in line:
                speaker, utterance = line.split(':', 1)
                dialog.append({"speaker": speaker.strip('**'), "utterance": utterance.strip()})

        # The script expects a list of conversation objects. We'll create one.
        conversations = [{
            "frame_index": 0, # Since we have only one analysis, use a default index.
            "dialog": dialog
        }]

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing the conversations file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data transformation: {e}")
        return

    # --- 4. Generate Audio and Create Manifest ---
    manifest = []
    generation_tasks = []
    print("Processing conversations and queueing audio generation...")

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

            # Determine avatar and voice
            avatar_filename = alex_avatar_name if speaker == "Alex" else jamie_avatar_name
            voice = ALEX_VOICE if speaker == "Alex" else JAMIE_VOICE
            
            # Define paths
            audio_filename = f"frame_{frame_index}_turn_{i}_{speaker}.mp3"
            audio_path_in_package = os.path.join(audio_output_dir, audio_filename)

            # Add entry to manifest
            manifest.append({
                "image": avatar_filename,
                "audio": os.path.join("audio_clips", audio_filename)
            })

            # Add the generation task
            generation_tasks.append(_generate_clip(utterance, voice, audio_path_in_package))
            print(f"  - Queued: {audio_filename} for {speaker}")

    # --- 5. Run Audio Generation ---
    print("\nGenerating all audio clips... (This may take a moment)")
    await asyncio.gather(*generation_tasks)
    print("All audio clips generated successfully.")

    # --- 6. Save the Manifest ---
    manifest_path = os.path.join(output_package_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest file created at: {manifest_path}")

    print("\n✅ Video package created successfully!")
    print("You can now upload the entire 'video_package' folder to a service like Google Colab.")

async def main():
    # Define paths
    CONVERSATION_FILE = "analysis_results.json"
    AVATAR_DIR = "."
    OUTPUT_PACKAGE_DIR = "video_package"

    await create_video_package(
        conversations_path=CONVERSATION_FILE,
        avatar_dir=AVATAR_DIR,
        output_package_dir=OUTPUT_PACKAGE_DIR,
    )

if __name__ == "__main__":
    asyncio.run(main()) 