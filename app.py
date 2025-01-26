import sys
import os
import re
import logging
import mimetypes
import tempfile
import uuid
import time
from dataclasses import dataclass, field 
from typing import List, Dict, Optional
import threading
from queue import Queue
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from bs4 import BeautifulSoup
import requests

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QMessageBox,
    QSystemTrayIcon,
    QMenu,
    QFileDialog,
    QLabel
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QSettings, QTimer, Qt

# Logging configuration
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    GEMINI_API_KEY: str = "Tu api key de ai studio"
    GEMINI_API_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-exp-1206:generateContent"
    )
    GEMINI_MODEL: str = "gemini-exp-1206"
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 2.0
    MAX_HISTORY: int = 100
    TIMEOUT: int = 60
    MAX_WORKERS: int = 4
    MAX_FILE_SIZE: int = 30 * 1024 * 1024  # 30 MB
    SUPPORTED_MIME_TYPES: list = field(default_factory=lambda: [ # Usa field y default_factory
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ])
    CHUNK_SIZE: int = 30000  # Tamaño de chunk en caracteres
    NOTION_TOKEN: str = ""
    NOTION_DATABASE_ID: str = ""
    NOTION_SYSTEM_PROMPT: str = "UTILIZA EL MÉTODO Zettelkasten para tomar notas de este contenido."

class DocumentManager:
    def __init__(self):
        self.active_documents = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.current_doc_id = None

    def _extract_text(self, file_path: str, mime_type: str) -> str:
        try:
            text = ""
            if mime_type == 'application/pdf':
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() for page in reader.pages])
            elif mime_type == 'text/plain':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        return [text[i:i+Config.CHUNK_SIZE] for i in range(0, len(text), Config.CHUNK_SIZE)]

    def store_document(self, file_data: bytes, file_name: str, mime_type: str) -> str:
        try:
            doc_id = str(uuid.uuid4())
            file_path = os.path.join(self.temp_dir.name, f"{doc_id}_{file_name}")
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            full_text = self._extract_text(file_path, mime_type)
            chunks = self._chunk_text(full_text)
            
            self.active_documents[doc_id] = {
                'name': file_name,
                'chunks': chunks,
                'current_chunk': 0,
                'mime_type': mime_type,
                'full_text': full_text
            }
            
            return doc_id
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def get_next_chunk(self, doc_id: str) -> Optional[str]:
        doc = self.active_documents.get(doc_id)
        if not doc:
            return None
        
        if doc['current_chunk'] >= len(doc['chunks']):
            return None
        
        chunk = doc['chunks'][doc['current_chunk']]
        doc['current_chunk'] += 1
        return chunk

    def reset_chunk_counter(self, doc_id: str):
        if doc_id in self.active_documents:
            self.active_documents[doc_id]['current_chunk'] = 0

class APIHandler:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    async def send_message(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a message to the Gemini API and returns the response.

        Args:
            messages: A list of dictionaries representing the conversation history,
                      where each dictionary has "role" and "content" keys.

        Returns:
            The response text from the Gemini API.

        Raises:
            requests.exceptions.RequestException: If there is an error with the HTTP request.
            ValueError: If the Gemini API response is unexpected or blocked.
        """
        for attempt in range(self.config.MAX_RETRIES):
            try:
                headers = {"Content-Type": "application/json"}
                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append(
                        {"role": role, "parts": [{"text": msg["content"]}]}
                    )

                payload = {"contents": gemini_messages}
                params = {"key": self.config.GEMINI_API_KEY}

                response = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.session.post(
                        url=self.config.GEMINI_API_URL,
                        headers=headers,
                        json=payload,
                        params=params,
                        timeout=self.config.TIMEOUT,
                    ),
                )
                response.raise_for_status()
                gemini_response = response.json()
                if gemini_response.get("candidates"):
                    return gemini_response["candidates"][0]["content"]["parts"][0].get(
                        "text", ""
                    )
                else:
                    logger.error(
                        f"Gemini API returned an unexpected response: {gemini_response}"
                    )
                    if (
                        "promptFeedback" in gemini_response
                        and "blockReason" in gemini_response["promptFeedback"]
                    ):
                        block_reason = gemini_response["promptFeedback"][
                            "blockReason"
                        ]
                        logger.error(
                            f"Gemini API blocked the request. Block reason: {block_reason}"
                        )
                        raise ValueError(
                            f"Gemini API blocked the request. Reason: {block_reason}"
                        )
                    raise ValueError("Unexpected response from Gemini API")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self.config.RETRY_DELAY * (attempt + 1))
            except ValueError as e:
                logger.error(f"Error processing Gemini API response: {str(e)}")
                if attempt == self.config.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self.config.RETRY_DELAY * (attempt + 1))

    def close(self):
        self._executor.shutdown(wait=True)
        self.session.close()
class NotionHandler:
    """
    Handles communication with the Notion API.
    """

    def __init__(self, config: Config):
        """
        Initializes the NotionHandler with configuration settings.

        Args:
            config: An instance of the Config class containing Notion API token and database ID.
        """
        self.config = config
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.config.NOTION_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def create_page(self, title: str, content: str) -> bool:
        """
        Creates a new page in Notion with the given title and content.

        Handles the 100 blocks per request limit by combining text into larger blocks.

        Args:
            title: The title of the new page.
            content: The content of the new page.

        Returns:
            True if the page was created successfully, False otherwise.
        """
        try:
            # Divide content into paragraphs
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

            # Combine paragraphs to keep fewer than 100 blocks
            combined_paragraphs = []
            current_block = []
            current_length = 0

            for paragraph in paragraphs:
                if current_length + len(paragraph) > 2000:  # Notion block character limit
                    if current_block:
                        combined_paragraphs.append("\n".join(current_block))
                        current_block = []
                        current_length = 0

                current_block.append(paragraph)
                current_length += len(paragraph) + 1  # +1 for newline character

            if current_block:
                combined_paragraphs.append("\n".join(current_block))

            # Limit to 100 blocks
            if len(combined_paragraphs) > 100:
                combined_paragraphs = combined_paragraphs[:99]
                combined_paragraphs.append(
                    "... (Content truncated due to Notion's block limit)"
                )

            # Create content blocks for Notion
            content_blocks = []
            for paragraph in combined_paragraphs:
                content_blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": paragraph},
                                }
                            ]
                        },
                    }
                )

            data = {
                "parent": {"database_id": self.config.NOTION_DATABASE_ID},
                "properties": {
                    "Title": {
                        "title": [{"text": {"content": title}}]
                    }
                },
                "children": content_blocks,
            }

            response = requests.post(
                f"{self.base_url}/pages", headers=self.headers, json=data
            )

            if response.status_code != 200:
                logger.error(
                    f"Notion API error: {response.status_code} - {response.text}"
                )
                return False

            response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating Notion page: {str(e)}")
            return False

class ConversationHistory:
    """
    Manages the conversation history.
    """

    def __init__(self, max_size: int = Config.MAX_HISTORY):
        """
        Initializes the ConversationHistory with a maximum size.

        Args:
            max_size: The maximum number of messages to store in the history.
        """
        self.history: List[Dict[str, str]] = []
        self.max_size = max_size
        self._lock = threading.Lock()

    def add(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role: The role of the message sender ("user" or "assistant").
            content: The content of the message.
        """
        with self._lock:
            if len(self.history) >= self.max_size:
                self.history.pop(0)
            self.history.append({"role": role, "content": content})

    def clear(self) -> None:
        """
        Clears the conversation history.
        """
        with self._lock:
            self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns a copy of the conversation history.

        Returns:
            A list of dictionaries representing the conversation history.
        """
        with self._lock:
            return self.history.copy()
class SignalHandler(QObject):
    response_received = pyqtSignal(str, float, int)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    ask_url_action = pyqtSignal(str)

class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.api_handler = APIHandler(self.config)
        self.document_manager = DocumentManager()
        self.conversation_history = ConversationHistory()
        self.signal_handler = SignalHandler()
        self.settings = QSettings("MyCompany", "Chatbot")
        self.message_queue = Queue()
        self.pending_url = None  # Initialize pending_url <-------------------- THIS LINE IS IMPORTANT
        self.pending_content = None # Initialize pending_content (if not already)
        self.waiting_for_notion_confirmation = False  # Initialize waiting_for_notion_confirmation
        self.notion_handler = NotionHandler(self.config) # Initialize notion_handler (if not already)
        self.init_ui()
        self.setup_signals()
        self.start_message_processing()
        self.setup_system_tray()

    def init_ui(self):
        self.setWindowTitle("Asistente Inteligencia Artificial")
        self.setFixedSize(1000, 800)
        self.setup_styles()
        self.create_widgets()
        self.create_layouts()
        self.setup_auto_scroll()

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #282c34;
                color: white;
            }
            QTextEdit, QLineEdit {
                background-color: #21252B;
                border: 2px solid #3d4148;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a4e58;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a5e68;
            }
            QPushButton:pressed {
                background-color: #3a3e48;
            }
            QLabel {
                font-size: 12px;
                color: #abb2bf;
            }
        """)

    def create_widgets(self):
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("Segoe UI", 10))

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Escribe tu mensaje o pregunta aquí...")
        self.input_field.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Enviar")
        self.clear_button = QPushButton("Limpiar Chat")
        self.load_button = QPushButton("Cargar Documento")
        self.doc_label = QLabel("Documento activo: Ninguno")

    def create_layouts(self):
        main_layout = QVBoxLayout(self)
        
        # Sección de documento
        doc_layout = QHBoxLayout()
        doc_layout.addWidget(self.doc_label)
        doc_layout.addWidget(self.load_button)
        
        # Área de chat
        main_layout.addLayout(doc_layout)
        main_layout.addWidget(self.chat_area)
        
        # Controles de entrada
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field, 4)
        input_layout.addWidget(self.send_button, 1)
        input_layout.addWidget(self.clear_button, 1)
        
        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

    def setup_signals(self):
        self.signal_handler.response_received.connect(self.update_chat_area)
        self.signal_handler.error_occurred.connect(self.show_error)
        self.signal_handler.ask_url_action.connect(self.ask_ai_about_url_action)
        self.send_button.clicked.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.load_button.clicked.connect(self.load_document)
    def setup_auto_scroll(self):
        """
        Sets up the auto-scroll feature for the chat area.
        """
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self.scroll_to_bottom_if_needed)
        self.scroll_timer.start(100)

    def scroll_to_bottom_if_needed(self):
        """
        Scrolls the chat area to the bottom if needed.
        """
        scroll_bar = self.chat_area.verticalScrollBar()
        if scroll_bar.value() >= scroll_bar.maximum() - 50:
            scroll_bar.setValue(scroll_bar.maximum())
    def start_message_processing(self):
        """
        Starts the message processing thread.
        """
        self.processing_thread = threading.Thread(
            target=self.process_message_queue, daemon=True
        )
        self.processing_thread.start()
    def process_message_queue(self):
        """
        Processes messages from the message queue.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            message = self.message_queue.get()
            if message is None:
                break
            loop.run_until_complete(self.process_message(message))
            self.message_queue.task_done()
        loop.close()

    def load_document(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Seleccionar Documento",
            "",
            "Documentos (*.pdf *.docx *.doc *.txt)"
        )
        
        if file_path:
            try:
                file_size = os.path.getsize(file_path)
                if file_size > self.config.MAX_FILE_SIZE:
                    raise ValueError(f"El archivo excede el tamaño máximo de {self.config.MAX_FILE_SIZE//1024//1024}MB")
                
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type not in self.config.SUPPORTED_MIME_TYPES:
                    raise ValueError("Tipo de archivo no soportado")
                
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                doc_id = self.document_manager.store_document(file_data, os.path.basename(file_path), mime_type)
                self.document_manager.current_doc_id = doc_id
                self.doc_label.setText(f"Documento activo: {os.path.basename(file_path)}")
                self.chat_area.append(f"\nSistema: Documento cargado - {os.path.basename(file_path)}")
                
            except Exception as e:
                self.signal_handler.error_occurred.emit(str(e))

    async def process_document_query(self, message: str, doc_id: str):
        try:
            doc = self.document_manager.active_documents[doc_id]
            full_response = []

            for chunk_index, chunk in enumerate(doc['chunks'], 1):
                prompt = f"""
                Documento: {doc['name']} (Parte {chunk_index} de {len(doc['chunks'])})
                Contenido: {chunk}

                Pregunta: {message}

                Instrucciones:
                1. Responde basándote exclusivamente en el contenido proporcionado
                2. Si la información no está presente, indica claramente que no hay datos
                3. Incluye referencias a la sección del documento cuando sea posible
                """

                messages = [{"role": "user", "content": prompt}]
                response = await self.api_handler.send_message(messages)
                full_response.append(response.strip())

                # Actualizar el historial parcialmente
                self.signal_handler.response_received.emit(
                    f"\n[Progreso: Parte {chunk_index}/{len(doc['chunks'])} procesada]",
                    0, 0
                )

            # **Pre-format the joined responses outside the f-string:**
            joined_responses = '\n'.join(full_response)

            # Procesar respuestas final
            summary_prompt = f"""
            Sintetiza una respuesta coherente basada en estas respuestas parciales:

            Pregunta original: {message}

            Respuestas parciales:
            {joined_responses}  # Use the pre-formatted string here

            Instrucciones:
            1. Combina la información de forma lógica
            2. Elimina repeticiones
            3. Mantén las referencias a secciones del documento
            4. Formatea la respuesta claramente
            """

            final_response = await self.api_handler.send_message([{"role": "user", "content": summary_prompt}])
            return final_response

        except Exception as e:
            logger.error(f"Error procesando documento: {str(e)}")
            raise
    async def get_youtube_transcript(self, url: str) -> Optional[str]:
        """
        Retrieves the transcript of a YouTube video.

        Args:
            url: The URL of the YouTube video.

        Returns:
            The transcript of the video, or None if no transcript is found.
        """
        try:
            video_id = url.split("v=")[1]
            if "&" in video_id:
                video_id = video_id.split("&")[0]
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en"]
            )
            transcript = " ".join([entry["text"] for entry in transcript_list])
            return transcript
        except NoTranscriptFound:
            try:
                available_languages = YouTubeTranscriptApi.list_transcripts(video_id)
                lang_codes = []
                for transcript in available_languages:
                    if transcript.language_code in ["es", "de", "fr"]:
                        transcript_text = " ".join(
                            [entry["text"] for entry in transcript.fetch()]
                        )
                        return transcript_text
                    else:
                        lang_codes.append(
                            f"- {transcript.language} ({transcript.language_code}) [{'auto-generated' if transcript.is_generated else 'manual'}]"
                        )

                error_message = f"No English, Spanish, German, or French transcripts found for this video.\nAvailable transcripts:\n" + "\n".join(lang_codes)
                logger.error(f"Error getting YouTube transcript: {error_message}")
                self.signal_handler.error_occurred.emit(error_message)
                return None
            except Exception as e:
                logger.error(f"Error getting YouTube transcript: {e}")
                self.signal_handler.error_occurred.emit(
                    f"Could not fetch YouTube transcript: {e}"
                )
                return None
        except Exception as e:
            logger.error(f"Error getting YouTube transcript: {e}")
            self.signal_handler.error_occurred.emit(
                f"Could not fetch YouTube transcript: {e}"
            )
            return None

    async def get_website_content(self, url: str) -> Optional[str]:
        """
        Retrieves the content of a website.

        Args:
            url: The URL of the website.

        Returns:
            The content of the website, or None if an error occurs.
        """
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text_content = " ".join(soup.stripped_strings)
            return text_content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting website content: {e}")
            self.signal_handler.error_occurred.emit(
                f"Could not fetch website content: {e}"
            )
            return None

    def detect_url(self, text: str) -> Optional[str]:
        """
        Detects a URL in a given text.

        Args:
            text: The text to search for a URL.

        Returns:
            The detected URL, or None if no URL is found.
        """
        url_match = re.search(r"https?://\S+", text)
        return url_match.group(0) if url_match else None

    def detect_url_type(self, url: str) -> Optional[str]:
        """
        Detects the type of a given URL (YouTube or website).

        Args:
            url: The URL to check.

        Returns:
            "youtube" if the URL is a YouTube URL, "website" otherwise.
        """
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        return "website"

    @pyqtSlot(str)
    def ask_ai_about_url_action(self, url):
        """
        Asks the user what to do with a detected URL.

        Args:
            url: The detected URL.
        """
        self.pending_url = url
        self.chat_area.append(
            f"Assistant: I detected a URL: {url}. What would you like me to do with it? You can also ask me to take notes and save them to Notion."
        )
        self.conversation_history.add(
            "assistant",
            f"I detected a URL: {url}. What would you like me to do with it? You can also ask me to take notes and save them to Notion.",
        )

    async def process_message(self, message: str):
        try:
            start_time = time.time()
            
            # Detectar URL primero
            detected_url = self.detect_url(message)
            if detected_url:
                self.signal_handler.ask_url_action.emit(detected_url)
                return

            # Manejar documentos
            if self.document_manager.current_doc_id:
                final_response = await self.process_document_query(
                    message, 
                    self.document_manager.current_doc_id
                )
                
                self.conversation_history.add("assistant", final_response)
                self.signal_handler.response_received.emit(
                    final_response, 
                    time.time() - start_time, 
                    len(final_response.split())
                )
                return

            # Resto de lógica para otras funcionalidades...
            # ... (código existente para manejo de URLs y conversación normal)
            if self.waiting_for_notion_confirmation:
                if message.lower() == "yes":
                    notes = self.conversation_history.get_history()[-2]["content"]
                    url = self.pending_url
                    page_title = f"Notes from {url}"
                    if self.notion_handler.create_page(page_title, notes):
                        self.chat_area.append(
                            "Assistant: Notes saved to Notion successfully!"
                        )
                        self.conversation_history.add(
                            "assistant", "Notes saved to Notion successfully!"
                        )
                    else:
                        self.signal_handler.error_occurred.emit(
                            "Failed to save notes to Notion."
                        )
                    self.pending_content = None
                    self.pending_url = None
                    self.waiting_for_notion_confirmation = False
                    return
                elif message.lower() == "no":
                    self.chat_area.append("Assistant: Okay, not saving to Notion.")
                    self.conversation_history.add(
                        "assistant", "Okay, not saving to Notion."
                    )
                    self.pending_content = None
                    self.pending_url = None
                    self.waiting_for_notion_confirmation = False
                    return
                else:
                    self.chat_area.append(
                        "Assistant: Invalid response. Please answer 'yes' or 'no'."
                    )
                    return

            if self.pending_url:
                url = self.pending_url
                url_type = self.detect_url_type(url)
                self.conversation_history.add("user", message)

                content = None
                if url_type == "youtube":
                    self.signal_handler.status_changed.emit(
                        f"Fetching YouTube transcript..."
                    )
                    content = await self.get_youtube_transcript(url)
                elif url_type == "website":
                    self.signal_handler.status_changed.emit(
                        f"Fetching website content..."
                    )
                    content = await self.get_website_content(url)

                if not content:
                    self.pending_url = None
                    return

                if any(
                    keyword in message.lower()
                    for keyword in ["summarize", "key points", "extract"]
                ):
                    if "summarize" in message.lower():
                        system_prompt = f"Please summarize the following {'transcript' if url_type == 'youtube' else 'content'} from a {url_type}:\n\n{content}"
                    else:
                        system_prompt = f"Please extract the key points from the following {'transcript' if url_type == 'youtube' else 'content'} of a {url_type}:\n\n{content}"

                    messages = [{"role": "user", "content": system_prompt}]
                    response = await self.api_handler.send_message(messages)
                    self.conversation_history.add("assistant", response)
                    self.signal_handler.response_received.emit(
                        response, time.time() - start_time, len(response.split())
                    )
                    self.pending_url = None
                elif (
                    "notion" in message.lower() or "take notes" in message.lower()
                ):
                    self.pending_content = content
                    self.pending_url = url

                    system_prompt = (
                        self.config.NOTION_SYSTEM_PROMPT + f"\n\nContent:\n{content}"
                    )
                    messages = [{"role": "user", "content": system_prompt}]
                    response = await self.api_handler.send_message(messages)
                    self.conversation_history.add("assistant", response)
                    self.signal_handler.response_received.emit(
                        response, time.time() - start_time, len(response.split())
                    )

                    self.chat_area.append(
                        "Assistant: Notes generated. Do you want to save them to Notion? (yes/no)"
                    )
                    self.conversation_history.add(
                        "assistant",
                        "Notes generated. Do you want to save them to Notion? (yes/no)",
                    )
                    self.waiting_for_notion_confirmation = True
                    return
                else:
                    system_prompt = f"Based on the following {url_type} {'transcript' if url_type == 'youtube' else 'content'}, answer the user's question:\n\n{content}\n\nUser's question: {message}"
                    messages = [{"role": "user", "content": system_prompt}]
                    response = await self.api_handler.send_message(messages)
                    self.conversation_history.add("assistant", response)
                    self.signal_handler.response_received.emit(
                        response, time.time() - start_time, len(response.split())
                    )
                    self.pending_url = None

                return

            gemini_history = []
            for msg in self.conversation_history.get_history():
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "content": msg["content"]})

            try:
                response = await self.api_handler.send_message(
                    gemini_history + [{"role": "user", "content": message}]
                )

                time_taken = time.time() - start_time
                tokens = len(response.split())

                self.conversation_history.add("assistant", response)
                self.signal_handler.response_received.emit(
                    response, time_taken, tokens
                )
            except ValueError as e:
                logger.error(f"Error processing message with Gemini: {e}")
                self.signal_handler.error_occurred.emit(str(e))

        except Exception as e:
            logger.error(f"Error procesando mensaje: {str(e)}")
            self.signal_handler.error_occurred.emit(str(e))

    # ... (mantener el resto de métodos existentes sin cambios)
    @pyqtSlot()
    def send_message(self):
        """
        Sends a message from the user to the chatbot.
        """
        message = self.input_field.text().strip()
        if not message:
            return

        self.input_field.clear()
        self.chat_area.append(f"You: {message}")
        self.message_queue.put(message)

    @pyqtSlot(str, float, int)
    def update_chat_area(self, response: str, time_taken: float, tokens: int):
        """
        Updates the chat area with a response from the chatbot.

        Args:
            response: The response text.
            time_taken: The time taken to generate the response.
            tokens: The number of tokens in the response.
        """
        self.chat_area.append(
            f"\nAssistant: {response}\n"
            f"Model: {self.config.GEMINI_MODEL} | "
            f"Time: {time_taken:.2f}s | "
            f"Tokens: {tokens}\n"
        )

    @pyqtSlot(str)
    def show_error(self, error_message: str):
        """
        Displays an error message.

        Args:
            error_message: The error message to display.
        """
        QMessageBox.critical(self, "Error", f"Gemini API Error: {error_message}")
        logger.error(f"Application error: {error_message}")

    @pyqtSlot(str)
    def update_status(self, status_message: str):
        """
        Updates the status message in the chat area.

        Args:
            status_message: The status message to display.
        """
        self.chat_area.append(f"Status: {status_message}")

    def clear_chat(self):
        """
        Clears the chat area and conversation history.
        """
        self.chat_area.clear()
        self.conversation_history.clear()
        self.chat_area.append("Chat cleared.")

    def setup_system_tray(self):
        """
        Sets up the system tray icon.
        """
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = "iconchatbot.ico"
        self.tray_icon.setIcon(
            QIcon(icon_path)
            if os.path.exists(icon_path)
            else QIcon.fromTheme("application-x-executable")
        )

        tray_menu = QMenu()
        restore_action = tray_menu.addAction("Restore")
        minimize_action = tray_menu.addAction("Minimize")
        quit_action = tray_menu.addAction("Quit")

        restore_action.triggered.connect(self.restore_window)
        minimize_action.triggered.connect(self.minimize_to_tray)
        quit_action.triggered.connect(self.quit_application)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()

    def tray_icon_activated(self, reason):
        """
        Handles activation of the system tray icon.

        Args:
            reason: The reason for activation.
        """
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isHidden() or self.isMinimized():
                self.restore_window()
            else:
                self.minimize_to_tray()

    def restore_window(self):
        """
        Restores the application window from the system tray.
        """
        self.showNormal()
        self.activateWindow()
        self.minimized_to_tray = False

    def minimize_to_tray(self):
        """
        Minimizes the application window to the system tray.
        """
        self.hide()
        self.minimized_to_tray = True
        if self.tray_icon.isVisible():
            self.tray_icon.showMessage(
                "Chatbot",
                "Application minimized to tray",
                QSystemTrayIcon.Information,
                2000,
            )

    def changeEvent(self, event):
        """
        Handles window state change events.

        Args:
            event: The event object.
        """
        if event.type() == event.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                event.accept()
                self.minimize_to_tray()
            else:
                super().changeEvent(event)

    def quit_application(self):
        """
        Quits the application.
        """
        reply = QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.tray_icon.hide()
            self.api_handler.close()
            QApplication.quit()

    def closeEvent(self, event):
        if hasattr(self.document_manager, 'temp_dir'):
            self.document_manager.temp_dir.cleanup()
        self.api_handler.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = ChatbotApp()
    window.show()
    sys.exit(app.exec_())
