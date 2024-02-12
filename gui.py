from run import MistralChatWrapper, WrapperLLM, mistral_api_key, FileScrapper, split_text_into_chunks
import sys
import re
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QApplication, QFileDialog, QComboBox, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QTextEdit, QScrollArea, QMessageBox

class MistralChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.llm = None
        self.setup()

    def setup(self):
        self.setup_llm()
        self.init_ui()
        self.update_chat_history()
        self.files_to_embed = []

    def setup_llm(self):
        mistralLLM = MistralChatWrapper(
            api_key=mistral_api_key, temperature_insead_top_p=True,
            model="mistral-tiny", max_tokens=21000,
        )
        mistralLLM.setup()
        self.llm = WrapperLLM(mistralLLM)

    def init_ui(self):
        self.setWindowTitle("Mistral Chat")
        self.setGeometry(100, 100, 400, 600)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowStaysOnTopHint)

        # Layouts
        layout = QVBoxLayout()

        # Outputs
        self.chat_log = QTextEdit()
        self.chat_log.setPlaceholderText("Here will be conversation log")
        self.chat_log.setWordWrapMode(True)
        layout.addWidget(self.chat_log)

        upper_part_layout = QHBoxLayout()
        layout.addLayout(upper_part_layout)
        prompt_layout = QVBoxLayout()
        upper_part_layout.addLayout(prompt_layout)
        button_layout = QVBoxLayout()
        upper_part_layout.addLayout(button_layout)

        # System prompt
        self.system_prompt_editor = QTextEdit()
        self.system_prompt_editor.setPlaceholderText(" Your System Prompt here. ")
        self.system_prompt_editor.font = QFont('Monospace')
        self.system_prompt_editor.setFixedHeight(70)
        prompt_layout.addWidget(self.system_prompt_editor)
        self.system_prompt_editor.setText(self.llm.history[0]['content'])

        # User input
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText(" Your Query here. \n include SRCH_EMBD to append found embeddings, if files to embed added.  ")
        self.user_input.font = QFont('Monospace')
        self.user_input.installEventFilter(self)
        self.user_input.setFixedHeight(140)
        prompt_layout.addWidget(self.user_input)
        button_layout.addSpacing(1)

        # Model Combo Box
        self.model_combo = QComboBox()
        for model in ['mistral-tiny', 'mistral-small', 'mistral-medium']:
            self.model_combo.addItem(model)
        self.model_combo.setFixedWidth(25)
        button_layout.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.model_name_changed)

        # Delete button
        self.delete_button = QPushButton("X")
        button_layout.addWidget(self.delete_button)
        self.delete_button.clicked.connect(self.delete_message)
        self.delete_button.setFixedHeight(25)
        self.delete_button.setFixedWidth(25)

        # Send button
        self.send_button = QPushButton(">")
        button_layout.addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedHeight(25)
        self.send_button.setFixedWidth(25)
        self.setLayout(layout)

        self.add_embedding_button = QPushButton("E(+)")
        self.add_embedding_button.setFixedSize(25, 25)
        button_layout.addWidget(self.add_embedding_button)

        # Add embedings
        self.add_embedding_box = QMessageBox(QMessageBox.Question, "Add Files", "Pres yes to add file, press no to output in console current files to embed.", QMessageBox.Yes | QMessageBox.No, self)
        self.add_embedding_button.clicked.connect(self.add_embedding_box.exec_)
        self.add_embedding_box.buttonClicked.connect(self.handle_response)

        # Delete embeddings
        self.delete_embedding = QPushButton("E(X)")
        self.delete_embedding.clicked.connect(self.delete_last_embedding)
        self.delete_embedding.setFixedSize(25, 25)
        button_layout.addWidget(self.delete_embedding)

    def delete_last_embedding(self):
        from colorama import Fore, init, Style
        self.files_to_embed = self.files_to_embed[:-1]
        print(f"{Style.BRIGHT}{Fore.RED} Embeddings: {self.files_to_embed} {Style.RESET_ALL}")

    def handle_response(self, button):
        from colorama import Fore, init, Style
        print(f"{Style.BRIGHT}{Fore.GREEN} Embeddings: {self.files_to_embed} {Style.RESET_ALL}")
        if button.text() == "&Yes":
            self.files_to_embed += self.open_file_dialog()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*);;Text Files (*.txt)", options=options)
        return file_names

    def model_name_changed(self, model:str):
        self.llm.llm.model = model
        print(self.llm.llm.model)

    def send_message(self):
        system_prompt = self.system_prompt_editor.toPlainText()
        user_message = self.user_input.toPlainText()

        if "SRCH_EMBD" in user_message:
            user_message += "Embeddings: " +self.get_embeddings(user_message)
            user_message = user_message.replace("SRCH_EMBD", "")
        self.user_input.clear()
        self.llm.set_system_prompt(system_prompt)

        self.llm.invoke(user_message)
        self.update_chat_history()

    def get_embeddings(self, query):
        if not self.files_to_embed:
            return

        file_path = self.files_to_embed[-1] #FileScrapper().find_file('', )
        file_content = FileScrapper().scrap_file(file_path)
        print(f" [ -- ] File content lenght: {len(file_content)}")
        file_content = split_text_into_chunks(file_content, 256)
        file_embeddings = self.llm.get_embeddings(file_content)
        search = self.llm.embedding_search(
            file_content, file_embeddings, query
        )
        print(search)
        return "\n ".join(search)
        #query = query.replace("$embd", str(search), 1)
        #print(f" [ -- ] Found similarities: {len(search)}")

    def delete_message(self):
        self.llm.set_history(self.llm.history[:-1])
        self.update_chat_history()

    def update_chat_history(self):
        self.chat_log.clear()

        html_content = '<div style="white-space: pre;" class="tip" markdown="1">'
        for entry in self.llm.history:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                html_content += f'<p"><b style="color:#dbb54d;">User:</b> {content}</p>'
            elif role == "assistant":
                html_content += f'<p><b style="color:#674ddb;">Assistant:</b> {content}</p>'
            html_content += '<br>'
        html_content += '</div>'
        self.chat_log.setHtml(html_content)
        print("updated")

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Enter and event.modifiers() != Qt.ShiftModifier:
                self.send_message()
                return True
            elif event.key() == Qt.Key_Escape:  # Handle Esc key to close the window
                self.close()
                return True
        return super().eventFilter(source, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MistralChatApp()
    window.show()
    sys.exit(app.exec())
