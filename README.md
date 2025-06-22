# Sport Vision Language

This project analyzes sports-related images and videos to extract insights. It uses various AI and computer vision techniques to process visual data.

## How It Works

The project is broken down into three main stages:

1.  **Image Analysis (`Image_Analyzer_AI.py`)**: This is the core analysis script.
    *   It uses the `insightface` library for high-accuracy face detection and recognition.
    *   A local database of player images is used to create face embeddings, allowing the system to recognize known players in new images.
    *   `EasyOCR` and `spaCy` (for Named Entity Recognition) are used to extract and understand text from the image, such as player jerseys, scores, and team names.
    *   All this extracted information (recognized players, text, etc.) is fed into the Gemini, which generates a conversational dialogue between two AI agents ("Alex" and "Jamie") discussing the scene.

2.  **Text-to-Speech (`dialog_to_audio.py`)**: This script converts the generated conversation into audio.
    *   It takes the dialogue from the analysis step.
    *   Using Microsoft's Edge-TTS service, it generates natural-sounding speech for each line of dialogue, assigning distinct voices to Alex and Jamie.
    *   The individual audio clips are then stitched together into a single audio file representing the full conversation.

3.  **Video Creation (`create_video_package.py`)**: This script prepares all the assets needed for the final video.
    *   It takes the generated audio clips and pairs them with static avatar images for Alex and Jamie.
    *   It organizes these assets (audio, images) into a "video package" directory.
    *   A `manifest.json` file is created to orchestrate the final video generation, specifying the sequence of avatars and audio clips. The final output is designed to resemble a sports broadcast or highlight report.

## Getting Started

To get started with this project, clone the repository and install the dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
python Image_Analyzer_AI.py
``` 