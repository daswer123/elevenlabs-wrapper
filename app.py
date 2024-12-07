import time
import gradio as gr
import os
from typing import Optional
from dotenv import load_dotenv
from elevenlabs_wrapper import ElevenLabsManager

class ElevenLabsApp:
    def __init__(self):
        self.manager: Optional[ElevenLabsManager] = None
        # Load API key from .env
        load_dotenv()
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        
        # Try to validate API key on initialization
        self.show_interface = False
        self.initial_voices = []
        if self.api_key:
            try:
                self.manager = ElevenLabsManager(api_key=self.api_key)
                self.initial_voices = self.get_filtered_voices("all")
                self.show_interface = True
            except Exception as e:
                print(f"Failed to validate initial API key: {str(e)}")
        
        self.create_interface()

    def validate_api_key(self, api_key: str):
        try:
            self.manager = ElevenLabsManager(api_key=api_key)
            voices = self.get_filtered_voices("all")
            initial_voice = voices[0] if voices else None
            return "API key validated successfully!", gr.update(visible=True), gr.update(visible=False), gr.update(choices=voices, value=initial_voice)
        except Exception as e:
            return f"Error: {str(e)}", gr.update(visible=False), gr.update(visible=True), gr.update(choices=[], value=None)

    def get_filtered_voices(self, filter_type: str):
        if not self.manager:
            return []
        voices = self.manager.get_voices(filter_type)
        voice_list = [f"{voice['name']} ({voice['category']})" for voice in voices]
        return voice_list

    def update_voice_list(self, filter_type: str):
        voices = self.get_filtered_voices(filter_type)
        initial_voice = voices[0] if voices else None
        return gr.update(choices=voices, value=initial_voice)

    def clone_voice(self, name: str, description: str, files):
        if not self.manager:
            return "Please validate API key first", None
        try:
            file_paths = [f.name for f in files]
            voice_id = self.manager.clone_voice(name, file_paths, description)
            voices = self.get_filtered_voices("all")
            return f"Voice cloned successfully! Voice ID: {voice_id}", None
        except Exception as e:
            return f"Error cloning voice: {str(e)}", None

    def delete_voice(self, voice_name: str):
        if not self.manager or not voice_name:
            return "Please select a voice to delete"
        try:
            voice_id = self.manager.find_voice_by_name(voice_name.split(" (")[0])
            if voice_id:
                success = self.manager.delete_voice(voice_id)
                if success:
                    return "Voice deleted successfully!"
                return "Failed to delete voice"
            return "Voice not found"
        except Exception as e:
            return f"Error deleting voice: {str(e)}"

    def set_model(self, model_name: str):
        if not self.manager:
            return "Please validate API key first"
        try:
            self.manager.set_model(model_name)
            return f"Model changed to {model_name}"
        except Exception as e:
            return f"Error changing model: {str(e)}"

    def generate_speech(self, text: str, voice_name: str, stability: float, similarity_boost: float, model: str):
        if not self.manager:
            return "Please validate API key first", None
        try:
            if not text.strip():
                return "Please enter some text", None
            if not voice_name:
                return "Please select a voice", None
            
            self.manager.set_stability(stability)
            self.manager.set_similarity_boost(similarity_boost)
            self.manager.set_model(model)
            voice_id = self.manager.find_voice_by_name(voice_name.split(" (")[0])
            audio = self.manager.generate_audio(text, voice_id)
            
            filename = str(time.time()) + ".wav"
            os.makedirs("temp", exist_ok=True)
            output_path = os.path.join("temp", filename)
            self.manager.save_audio(audio, output_path)
            
            return "Speech generated successfully!", output_path
        except Exception as e:
            return f"Error generating speech: {str(e)}", None

    def create_interface(self):
        with gr.Blocks() as app:
            gr.Markdown("# ElevenLabs Text-to-Speech Interface")
            
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Enter your ElevenLabs API key",
                    type="password",
                    placeholder="Enter API key here...",
                    value=self.api_key if self.api_key else ""
                )
                validate_btn = gr.Button("Validate API Key")
            api_status = gr.Textbox(label="Status", interactive=False)
            
            main_interface = gr.Column(visible=self.show_interface)
            with main_interface:
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion(label="Voice", open=True):
                            voice_filter = gr.Dropdown(
                                choices=["all", "cloned", "non-cloned"],
                                label="Filter Voices",
                                value="all"
                            )
                            with gr.Row():
                                voice_dropdown = gr.Dropdown(
                                    label="Select Voice",
                                    interactive=True,
                                    choices=self.initial_voices,
                                    value=self.initial_voices[0] if self.initial_voices else None
                                )
                            refresh_btn = gr.Button("Refresh Voices")
                            delete_btn = gr.Button("Delete Selected Voice")
                            delete_status = gr.Textbox(
                                label="Deletion Status",
                                interactive=False
                            )

                        with gr.Accordion("Voice Settings", open=False):
                            model_dropdown = gr.Dropdown(
                                choices=["eleven_turbo_v2_5", "eleven_multilingual_v2"],
                                value="eleven_turbo_v2_5",
                                label="Model"
                            )
                            stability = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                label="Stability",
                                info="Higher values make the voice more consistent"
                            )
                            similarity_boost = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.75,
                                label="Similarity Boost",
                                info="Higher values make the voice more similar to the original"
                            )

                        with gr.Accordion("Clone New Voice", open=False):
                            clone_name = gr.Textbox(
                                label="Voice Name",
                                placeholder="Enter a name for your cloned voice"
                            )
                            clone_description = gr.Textbox(
                                label="Voice Description",
                                placeholder="Enter a description for your voice"
                            )
                            clone_files = gr.File(
                                label="Upload Audio Files",
                                file_count="multiple"
                            )
                            clone_btn = gr.Button("Clone Voice")
                            clone_status = gr.Textbox(
                                label="Cloning Status",
                                interactive=False
                            )

                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Text to Convert",
                            placeholder="Enter text here...",
                            lines=5
                        )
                        generate_btn = gr.Button("Generate Speech", variant="primary")
                        with gr.Row():
                            status_box = gr.Textbox(
                                label="Generation Status",
                                interactive=False
                            )
                            audio_output = gr.Audio(
                                label="Generated Speech",
                                type="filepath"
                            )

            api_key_view = gr.Column(visible=not self.show_interface)
            with api_key_view:
                gr.Markdown("Please enter your API key to continue")

            validate_btn.click(
                fn=self.validate_api_key,
                inputs=[api_key_input],
                outputs=[api_status, main_interface, api_key_view, voice_dropdown]
            )
            
            voice_filter.change(
                fn=self.update_voice_list,
                inputs=[voice_filter],
                outputs=[voice_dropdown]
            )
            
            refresh_btn.click(
                fn=self.update_voice_list,
                inputs=[voice_filter],
                outputs=[voice_dropdown]
            )
            
            clone_btn.click(
                fn=self.clone_voice,
                inputs=[clone_name, clone_description, clone_files],
                outputs=[clone_status, audio_output]
            ).then(
                fn=self.update_voice_list,
                inputs=[voice_filter],
                outputs=[voice_dropdown]
            )
            
            delete_btn.click(
                fn=self.delete_voice,
                inputs=[voice_dropdown],
                outputs=[delete_status]
            ).then(
                fn=self.update_voice_list,
                inputs=[voice_filter],
                outputs=[voice_dropdown]
            )
            
            generate_btn.click(
                fn=self.generate_speech,
                inputs=[text_input, voice_dropdown, stability, similarity_boost, model_dropdown],
                outputs=[status_box, audio_output]
            )

        return app

if __name__ == "__main__":
    app = ElevenLabsApp()
    app.create_interface().launch(
        inbrowser=True
    )