from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr
#load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


system_prompt = """
                You are Chanakya.
                You are a great political strategist, economist and philosopher.
                Answer the questions through the lens of Chanakya's philosophy.
                You will share personal things from your life even when user's don't ask for it.
                For example, if the user asks about the war strategies, you will share your
                personal experience with it and not only explain the theory.
                Please explain in 2-5 sentences. You should have sense of humour.
                """

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()


def chat(user_input, hist):
    # Convert Gradio history into LangChain message history
    langchain_history = []
    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))

    # Generate response
    response = chain.invoke({"input": user_input, "history": langchain_history})

    # Return updated history (Gradio now expects dicts with role/content)
    return "", hist + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ]


page = gr.Blocks(
    title="Ask Chanakya",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
        """
        # Ask Chanakya 
        ![](file/avatar.png)
        ## The great economist | The Royal Advisor & Teacher
        Welcome to your personal conversation with Chanakya!
        """
    )

    chatbot = gr.Chatbot(
        type='messages',
        avatar_images=(None, 'avatar.png'),  # optional avatar for assistant
        show_label=False
    )

    msg = gr.Textbox(show_label=False, placeholder="Ask Chanakya anything....")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat")
    clear.click(lambda: (None, []), None, [msg, chatbot])

page.launch()
