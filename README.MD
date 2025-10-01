# Ask Chanakya

Welcome to **Ask Chanakya**, an interactive chatbot application that allows you to have a conversation with Chanakya, the great political strategist, economist, and philosopher. This project leverages the power of LangChain, Google Generative AI, and Gradio to create an engaging and insightful experience.

### Features

- **Interactive Chat**: Ask questions and receive responses inspired by Chanakya's philosophy.
- **Personal Responses**: The chatbot shares Chanakya's personal experiences while answering.
- **Customizable Prompts**: Modify the system prompt to tailor the chatbot's behavior.
- **Gradio Interface**: A user-friendly web interface for seamless interaction.

### Technologies Used

- **Python**: Core programming language.
- **LangChain**: For managing prompts and message history.
- **Google Generative AI**: Powered by the Gemini model for generating responses.
- **Gradio**: For building the web-based chatbot interface.
- **dotenv**: For managing environment variables.

### Prerequisites

- Python 3.8 or higher
- A valid Google Generative AI API key (Gemini API)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akhilesh-halekote/ask-chanakya.git
   cd ask-chanakya

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv 
   source venv/bin/activate 
   #On Windows: venv\Scripts\activate
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
4. Set up environment variables:
   1. Create a .env file in the project root. 
   2. Add your Google Generative AI API key to the .env file :
      GEMINI_API_KEY=your_google_api_key </br>
      To know more click [here](https://support.gemini.com/hc/en-us/articles/360031080191-How-do-I-create-an-API-key?utm_source=google&utm_campaignID=23052016550&utm_adgroupID=&utm_creativeID=&utm_medium=cpc&utm_content=&utm_term=&gad_source=1&gad_campaignid=23045996358&gclid=EAIaIQobChMI2sO5r8yCkAMVtZaDBx3jeR0CEAAYASAAEgKAlvD_BwE)

## Usage
1. Run the application:  
   ```bash 
   python main.py
2. Open the Gradio interface in your browser. The application will launch at http://localhost:7860.  
3. Start chatting with Chanakya by typing your questions in the input box.  

## File Structure
- main.py: The main application script.
- requirements.txt: List of dependencies.
- .env: Environment variables (not included in the repository).
- avatar.png: Avatar image for the chatbot interface.

## License
This project is licensed under the MIT License. See the LICENSE file for details.  

## Acknowledgments
- LangChain
- Gradio
- Google Generative AI 
#### Enjoy your conversation with Chanakya!

## Demo
- Screen shot
![screen_shot](https://github.com/akhilesh-halekote/ask-chanakya/blob/main/readme/ui.png)
- You can try this out [here](https://ask-chanakya-production.up.railway.app)
