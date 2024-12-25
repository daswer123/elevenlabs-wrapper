import os
import math
import json
import logging
import requests
from typing import List, Optional
from elevenlabs.client import ElevenLabs
from elevenlabs import save

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ElevenLabsManager:
    def __init__(
        self,
        api_key: str,
        default_model: str = "eleven_turbo_v2_5",
        base_url: str = "https://api.elevenlabs.io/v1"
    ):
        logger.info("Initializing ElevenLabsManager...")
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        
        # Флаги доступности
        self._is_available = False
        self._can_generate = False
        
        # Итоговая информация о подписке
        self._subscription_info = {
            "total_symbols": None,
            "used_symbols": None,
            "remain_symbols": None,
            "can_generate": None
        }

        # Причина недоступности
        self._unavailability_reason = ""

        # Настройки генерации
        self._voices_cache = None
        self.stability = 0.5
        self.similarity_boost = 0.75

        # Инициализация ElevenLabs клиентом
        try:
            self.client = ElevenLabs(api_key=self.api_key, base_url=self.base_url)
            logger.info("Successfully connected to ElevenLabs API")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {str(e)}")
            raise
        
        logger.info(f"Initialized with default model: {default_model}")
        logger.info(f"Initial stability: {self.stability}")
        logger.info(f"Initial similarity boost: {self.similarity_boost}")

        # После инициализации сразу проверим здоровье
        self.check_api_health_update_flags()

    # -------------------------------------------------------------------------
    # Методы для установки новых значений ключа и URL
    # -------------------------------------------------------------------------
    def set_api_key(self, new_api_key: str):
        """
        Устанавливает новый API-ключ и переинициализирует клиент,
        а затем заново проверяет доступность API и обновляет внутренние флаги.
        """
        logger.info(f"Setting a new API key: {new_api_key}")
        self.api_key = new_api_key
        self._voices_cache = None  # сбрасываем кэш, если был

        try:
            self.client = ElevenLabs(api_key=self.api_key, base_url=self.base_url)
            logger.info("Client re-initialized with the new API key.")
        except Exception as e:
            logger.error(f"Failed to re-initialize client with new API key: {str(e)}")
            raise

        # Проверим здоровье только если это не часть совместной установки
        if not hasattr(self, '_skip_health_check'):
            self.check_api_health_update_flags()

    def set_base_url(self, new_base_url: str):
        """
        Устанавливает новый base_url и переинициализирует клиент,
        а затем заново проверяет доступность API и обновляет внутренние флаги.
        """
        logger.info(f"Setting a new base_url: {new_base_url}")
        self.base_url = new_base_url
        self._voices_cache = None  # сбрасываем кэш, если был

        try:
            self.client = ElevenLabs(api_key=self.api_key, base_url=self.base_url)
            logger.info("Client re-initialized with the new base_url.")
        except Exception as e:
            logger.error(f"Failed to re-initialize client with new base_url: {str(e)}")
            raise

        # Проверим здоровье только если это не часть совместной установки
        if not hasattr(self, '_skip_health_check'):
            self.check_api_health_update_flags()
    
    def set_api_and_base_url(self, api_key: str, base_url: str):
        # Устанавливаем флаг для пропуска промежуточных проверок
        self._skip_health_check = True
        
        try:
            self.set_api_key(api_key)
            self.set_base_url(base_url)
            # Делаем одну финальную проверку
            self.check_api_health_update_flags()
        finally:
            # Удаляем флаг в любом случае
            delattr(self, '_skip_health_check')

    # -------------------------------------------------------------------------
    # Внутренний метод для обновления внутренних флагов по результатам check_api_health
    # -------------------------------------------------------------------------
    def check_api_health_update_flags(self):
        """
        Вызывает check_api_health(), обновляет _is_available, _can_generate,
        а также _unavailability_reason в случае проблем.
        """
        health = self.check_api_health()
        self._is_available = health["api_available"]
        self._can_generate = health["subscription"]["can_generate"]
        self._unavailability_reason = health["reason"]

        logger.info(
            f"Updated internal flags: _is_available={self._is_available}, "
            f"_can_generate={self._can_generate}, reason='{self._unavailability_reason}'"
        )

    # =========================================================================
    # Стандартные "геттеры" для _is_available и _can_generate, если нужно
    # =========================================================================
    @property
    def is_available(self):
        return self._is_available

    @property
    def can_generate(self):
        return self._can_generate
    
    @property
    def subscription_info(self):
        return self._subscription_info

    # -------------------------------------------------------------------------
    # Методы управления параметрами TTS
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Работа с голосами
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Обработка файлов (сплит)
    # -------------------------------------------------------------------------
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
                            logger.info(
                                f"Created part {i+1}/{parts_count}: {part_file} "
                                f"(size: {part_size/1024/1024:.2f}MB)"
                            )
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

    # -------------------------------------------------------------------------
    # Удаление, клонирование голосов
    # -------------------------------------------------------------------------
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
            voices = self._fetch_voices()
            voice_exists = False
            is_cloned = False

            for voice in voices:
                if voice.voice_id == voice_id:
                    voice_exists = True
                    is_cloned = (voice.category == "cloned")
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
            self._voices_cache = None
            logger.info(f"Successfully deleted voice {voice_name} (ID: {voice_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {str(e)}")
            raise

    def clone_voice(self, name: str, files: List[str], description: Optional[str] = None) -> str:
        """
        Clone a voice from audio files.
        """
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

    # -------------------------------------------------------------------------
    # Генерация и сохранение аудио
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Поиск голоса
    # -------------------------------------------------------------------------
    def find_voice_by_name(self, voice_name: str) -> Optional[str]:
        """
        Find a voice_id by its name. Returns None if not found.
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

    # -------------------------------------------------------------------------
    # Проверка доступности API и состояния подписки
    # -------------------------------------------------------------------------
    def check_api_availability(self) -> bool:
        """
        Проверяем, что API-ключ рабочий, пытаясь получить список голосов.
        Возвращает:
            True, если ключ валиден (получили голоса без ошибок и список не пуст).
            False в противном случае.
        """
        logger.info("Checking if ElevenLabs API is available with the provided API key...")
        try:
            voices = self.get_voices(filter_by="all")
            if voices and len(voices) > 0:
                logger.info("API key is valid, voices retrieved successfully.")
                return True
            else:
                logger.warning("API key might be invalid or no voices returned.")
                return False
        except Exception as e:
            logger.error(f"Error checking API availability: {e}")
            return False

    def check_subscription_info(self) -> dict:
        """
        Запрос информации о подписке через /v1/user/subscription.
        Возвращает словарь со структурой:
            {
                "total_symbols": int,
                "used_symbols": int,
                "remain_symbols": int,
                "can_generate": bool
            }

        В случае ошибки выбрасывает исключение или заполняет поля None.
        """
        logger.info("Checking subscription limits from ElevenLabs...")
        subscription_url = f"{self.base_url}/v1/user/subscription"
        headers = {
            "xi-api-key": self.api_key
        }

        try:
            response = requests.get(subscription_url, headers=headers)
            logger.debug(f"Subscription response JSON: {response.json()}")
            if response.status_code != 200:
                logger.warning(f"Failed to fetch subscription info. Status code: {response.status_code}")
                response.raise_for_status()

            data = response.json()

            # Предполагаем, что в ответе есть эти поля:
            total_symbols = data.get("character_limit", 0)
            used_symbols = data.get("character_count", 0)
            remain_symbols = total_symbols - used_symbols
            can_generate = (remain_symbols > 0)

            self._subscription_info = {
                "total_symbols": total_symbols,
                "used_symbols": used_symbols,
                "remain_symbols": max(remain_symbols, 0),
                "can_generate": can_generate
            }
            logger.info(
                f"Subscription info: "
                f"total={total_symbols}, used={used_symbols}, remain={remain_symbols}, can_generate={can_generate}"
            )
            return self._subscription_info

        except Exception as e:
            logger.error(f"Error while checking subscription info: {e}")
            raise

    def check_api_health(self) -> dict:
        """
        Общая проверка работоспособности API.
        1) Проверяет валидность API-ключа (check_api_availability).
        2) Если ключ валиден, запрашивает информацию о лимитах (check_subscription_info).

        Возвращает словарь формата:
            {
                "api_available": bool,
                "reason": str,
                "subscription": {
                    "total_symbols": int/None,
                    "used_symbols": int/None,
                    "remain_symbols": int/None,
                    "can_generate": bool/None
                }
            }
        """
        logger.info("Performing overall API health check...")

        result = {
            "api_available": False,
            "reason": "",
            "subscription": {
                "total_symbols": None,
                "used_symbols": None,
                "remain_symbols": None,
                "can_generate": None
            }
        }

        # 1. Проверяем доступность API
        api_ok = self.check_api_availability()
        if not api_ok:
            reason = "Invalid API key or cannot fetch voices."
            logger.warning(reason)
            result["api_available"] = False
            result["reason"] = reason
            return result

        # 2. Если ключ валиден, проверяем лимиты
        try:
            sub_info = self.check_subscription_info()
            result["api_available"] = True
            result["subscription"] = sub_info

            # Если can_generate=False, значит лимит символов исчерпан
            if not sub_info["can_generate"]:
                result["reason"] = "No symbols left for generation."
            else:
                result["reason"] = "All good. You can generate audio."
        except Exception as e:
            reason = f"Failed to get subscription info: {e}"
            logger.warning(reason)
            result["api_available"] = False
            result["reason"] = reason

        return result


# -----------------------------------------------------------------------------
# Пример использования
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        api_key = "-"
        manager = ElevenLabsManager(api_key=api_key, base_url="https://api.elevenlabs.io")

        # Посмотрим, что у нас в health
        health = manager.check_api_health()
        print("API Health:", health)
        print("Is available?", manager.is_available)
        print("Can generate?", manager.can_generate)

        # Попробуем сменить API-ключ (фиктивный пример):
        manager.set_api_key("NEW_KEY_123")
        # И посмотрим заново, что вернёт:
        print("After setting new API key:", manager.check_api_health())
        print("Is available?", manager.is_available)
        print("Can generate?", manager.can_generate)

        # Сменим base_url (опять же условно, если бы работали с иным URL)
        manager.set_base_url("https://api.elevenlabs.io/v1")
        print("After setting new base url:", manager.check_api_health())
        print("Is available?", manager.is_available)
        print("Can generate?", manager.can_generate)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
