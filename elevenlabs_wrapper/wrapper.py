import os
import math
import json
import logging
from typing import List, Optional
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ElevenLabsManager:
    def __init__(self, api_key: str, default_model: str = "eleven_turbo_v2_5",base_url : str = "https://api.elevenlabs.io/v1"):
        logger.info("Initializing ElevenLabsManager...")
        try:
            self.client = ElevenLabs(api_key=api_key,base_url=base_url)
            logger.info("Successfully connected to ElevenLabs API")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {str(e)}")
            raise

        self.default_model = default_model
        self._voices_cache = None
        self.stability = 0.5
        self.similarity_boost = 0.75
        logger.info(f"Initialized with default model: {default_model}")
        logger.info(f"Initial stability: {self.stability}")
        logger.info(f"Initial similarity boost: {self.similarity_boost}")

    def set_model(self, model_name: str):
        """Set the default model for generation."""
        logger.info(f"Changing model from {self.default_model} to {model_name}")
        self.default_model = model_name
        logger.info("Model successfully changed")

    def set_stability(self, value: float):
        """Set the stability parameter."""
        if not 0 <= value <= 1:
            logger.warning(f"Invalid stability value {value}. Must be between 0 and 1")
            raise ValueError("Stability must be between 0 and 1")
        logger.info(f"Changing stability from {self.stability} to {value}")
        self.stability = value

    def set_similarity_boost(self, value: float):
        """Set the similarity boost parameter."""
        if not 0 <= value <= 1:
            logger.warning(f"Invalid similarity boost value {value}. Must be between 0 and 1")
            raise ValueError("Similarity boost must be between 0 and 1")
        logger.info(f"Changing similarity boost from {self.similarity_boost} to {value}")
        self.similarity_boost = value

    def _fetch_voices(self):
        """Internal method to fetch and cache voices."""
        logger.info("Fetching available voices from API...")
        try:
            if self._voices_cache is None:
                response = self.client.voices.get_all()
                self._voices_cache = response.voices
                logger.info(f"Successfully cached {len(self._voices_cache)} voices")
            return self._voices_cache
        except Exception as e:
            logger.error(f"Failed to fetch voices: {str(e)}")
            raise

    def get_voices(self, filter_by: str = 'all'):
        """Get filtered list of voices."""
        logger.info(f"Getting voices with filter: {filter_by}")
        try:
            voices = self._fetch_voices()
            filtered_voices = []
            for voice in voices:
                if filter_by == 'all':
                    filtered_voices.append(voice)
                elif filter_by == 'cloned' and voice.category == 'cloned':
                    filtered_voices.append(voice)
                elif filter_by == 'non-cloned' and voice.category != 'cloned':
                    filtered_voices.append(voice)
            
            logger.info(f"Found {len(filtered_voices)} voices matching filter '{filter_by}'")
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": v.category,
                    "description": v.description,
                    "labels": v.labels,
                    "preview_url": v.preview_url
                }
                for v in filtered_voices
            ]
        except Exception as e:
            logger.error(f"Error while filtering voices: {str(e)}")
            raise

    def split_files_if_needed(self, file_paths: List[str], max_size_mb: float = 10.0) -> List[str]:
        """
        Checks the size of files and splits them if they exceed max_size_mb.
        Returns a list of paths to the final files.
        """
        logger.info(f"Checking and splitting files if needed (max size: {max_size_mb}MB)")
        final_files = []
        max_bytes = max_size_mb * 1024 * 1024
        safe_size = 9 * 1024 * 1024  # 9MB для безопасной загрузки

        for fpath in file_paths:
            try:
                if not os.path.exists(fpath):
                    logger.error(f"File not found: {fpath}")
                    raise FileNotFoundError(f"File not found: {fpath}")

                size = os.path.getsize(fpath)
                logger.info(f"Processing file: {fpath} (size: {size/1024/1024:.2f}MB)")

                if size <= max_bytes:
                    logger.info(f"File {fpath} is within size limit, using as is")
                    final_files.append(fpath)
                else:
                    logger.warning(f"File {fpath} exceeds {max_size_mb}MB, splitting required")
                    parts_count = math.ceil(size / safe_size)
                    logger.info(f"Will split into {parts_count} parts")

                    base_name, ext = os.path.splitext(fpath)

                    with open(fpath, 'rb') as infile:
                        for i in range(parts_count):
                            part_data = infile.read(int(safe_size))
                            part_file = f"{base_name}_part{i+1}{ext}"

                            with open(part_file, 'wb') as outfile:
                                outfile.write(part_data)

                            part_size = os.path.getsize(part_file)
                            logger.info(f"Created part {i+1}/{parts_count}: {part_file} "
                                      f"(size: {part_size/1024/1024:.2f}MB)")
                            final_files.append(part_file)

                    logger.info(f"Successfully split {fpath} into {parts_count} parts")

            except Exception as e:
                logger.error(f"Error processing file {fpath}: {str(e)}")
                raise

        logger.info(f"File processing completed. Total files for upload: {len(final_files)}")

        # Проверка итоговых файлов
        for f in final_files:
            size = os.path.getsize(f)
            if size > max_bytes:
                logger.error(f"Final file {f} is still too large: {size/1024/1024:.2f}MB")
                raise ValueError(f"Failed to properly split file {f}")
            logger.debug(f"Final file {f} size: {size/1024/1024:.2f}MB")

        return final_files
    
    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice by its ID.

        Args:
            voice_id (str): The ID of the voice to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        logger.info(f"Attempting to delete voice with ID: {voice_id}")

        try:
            # First check if voice exists and is cloned
            voices = self._fetch_voices()
            voice_exists = False
            is_cloned = False

            for voice in voices:
                if voice.voice_id == voice_id:
                    voice_exists = True
                    is_cloned = voice.category == "cloned"
                    voice_name = voice.name
                    break
                
            if not voice_exists:
                logger.warning(f"Voice with ID {voice_id} not found")
                return False

            if not is_cloned:
                logger.warning(f"Voice {voice_id} is not a cloned voice and cannot be deleted")
                return False

            logger.info(f"Deleting cloned voice: {voice_name} (ID: {voice_id})")
            self.client.voices.delete(voice_id)

            # Reset cache to reflect changes
            self._voices_cache = None
            logger.info(f"Successfully deleted voice {voice_name} (ID: {voice_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {str(e)}")
            raise

    
    def clone_voice(self, name: str, files: List[str], description: Optional[str] = None) -> str:
        """Clone a voice from audio files."""
        logger.info(f"Starting voice cloning process for '{name}'")
        logger.info(f"Input files: {files}")
    
        try:
            prepared_files = self.split_files_if_needed(files, max_size_mb=10.0)
            logger.info(f"Prepared {len(prepared_files)} files for upload")

            voice = self.client.clone(
                name=name,
                description=description,
                files=prepared_files
            )
            
            logger.info(f"Successfully cloned voice. New voice ID: {voice.voice_id}")
            self._voices_cache = None
            return voice.voice_id

        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            raise

    def generate_audio(self, text: str, voice: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Generate audio from text."""
        logger.info("Starting audio generation...")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Requested voice: {voice}")
        logger.info(f"Using model: {model or self.default_model}")

        try:
            if model is None:
                model = self.default_model

            voice_id = voice
            if voice:
                found_id = self.find_voice_by_name(voice)
                voice_id = found_id if found_id else voice
                logger.info(f"Resolved voice ID: {voice_id}")

            logger.info("Generating audio...")
            audio = self.client.generate(
                text=text,
                voice=voice_id,
                model=model,
                voice_settings={
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                },
                **kwargs
            )
            logger.info("Audio generation completed successfully")
            return audio

        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise

    def save_audio(self, audio: bytes, filename: str):
        """Save generated audio to file."""
        logger.info(f"Saving audio to file: {filename}")
        try:
            save(audio, filename)
            logger.info("Audio file saved successfully")
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            raise

    def find_voice_by_name(self, voice_name: str) -> Optional[str]:
            """
            Find a voice_id by its name. Returns None if not found.

            Args:
                voice_name (str): The name of the voice to search for

            Returns:
                Optional[str]: The voice_id if found, None otherwise
            """
            logger.info(f"Searching for voice with name: '{voice_name}'")

            try:
                voices = self._fetch_voices()
                logger.debug(f"Searching through {len(voices)} available voices")

                for voice in voices:
                    if voice.name.lower() == voice_name.lower():
                        logger.info(f"Voice found! Name: '{voice.name}', ID: {voice.voice_id}")
                        return voice.voice_id

                logger.warning(f"No voice found with name '{voice_name}'")
                return None

            except Exception as e:
                logger.error(f"Error while searching for voice: {str(e)}")
                raise


if __name__ == "__main__":
    try:
        logger.info("Starting ElevenLabs TTS service...")
        
        api_key = "your_api"
        manager = ElevenLabsManager(api_key=api_key)
        
        logger.info("Configuring voice settings...")
        manager.set_model("eleven_turbo_v2_5")
        manager.set_stability(0.5)
        manager.set_similarity_boost(0.8)
        
        voices = manager.get_voices("cloned")
        print(voices)

        logger.info("Starting voice cloning process...")
        new_voice_id = manager.clone_voice(
            name="TestVoice",
            files=["sample_0.mp3"],
            description="Test voice description"
        )
        logger.info(f"Voice cloning completed. ID: {new_voice_id}")

        logger.info("Generating test audio...")
        audio_data = manager.generate_audio(
            text="This is a test of the cloned voice.",
            voice=new_voice_id
        )

        logger.info("Saving generated audio...")
        manager.save_audio(audio_data, "test_output.mp3")
        logger.info("Process completed successfully")
        
        logger.info("Deleting custom voice")
        manager.delete_voice(new_voice_id)
        logger.info("Custom voice deleted successfully")



    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise