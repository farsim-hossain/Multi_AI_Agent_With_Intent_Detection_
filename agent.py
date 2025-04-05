# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from tools.search_tool import SearchTool
# from tools.document_processing_tool import DocumentProcessingTool
# from langchain_core.messages import HumanMessage
# from tools.email_tool import EmailTool
# from utils.env_loader import load_env

# class AIAgent:
#     def __init__(self):
#         groq_api_key = load_env()[0]
#         self.llm = ChatGroq(model="llama-3.2-90b-vision-preview", api_key=groq_api_key)
#         self.search_tool = SearchTool()
#         self.document_tool = DocumentProcessingTool()
#         self.email_tool = EmailTool(llm=self.llm)  # Pass the LLM to the EmailTool
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self.vector_store = None
#         self.is_document_context_active = False
#         self.last_read_emails = []  # Store the last read emails for context

#     def generate_response(self, query):
#         # Fetch the latest news using DuckDuckGo
#         news_results = self.search_tool.search_latest_news(query)
        
#         # Format the news results into a readable summary
#         news_summary = "\n".join([f"- {item}" for item in news_results[:5]])
        
#         # Generate a natural language response using the LLM
#         prompt = f"Here is the latest news about {query}:\n{news_summary}\n\nCan you summarize this and provide insights?"
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         return response.content

#     def process_and_chat_documents(self, query):
#         if not self.vector_store:
#             return "No documents processed yet. Please process documents first."
        
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=self.vector_store.as_retriever(),
#             memory=self.memory
#         )
#         result = qa_chain.invoke({"question": query})
#         return result["answer"]

#     def send_email_manually(self, user_input):
#         if "send email" in user_input.lower():
#             to = input("Enter recipient's email address: ").strip()
#             subject = input("Enter email subject: ").strip()
#             body = input("Enter email body: ").strip()
#             return self.email_tool.send_email(to, subject, body)
#         return "Please specify 'send email' in your request."

#     def read_emails(self, user_input):
#         if "read emails" in user_input.lower():
#             num_emails = int(input("How many emails would you like to read? ").strip())
#             self.last_read_emails = self.email_tool.read_emails(num_emails=num_emails)  # Store emails for later use
#             if isinstance(self.last_read_emails, str):
#                 return self.last_read_emails
            
#             # Display emails
#             email_list = []
#             for i, email in enumerate(self.last_read_emails):
#                 email_list.append(f"\nEmail {i + 1}:\nFrom: {email['from']}\nSubject: {email['subject']}\nBody: {email['body'][:100]}...")
#             return "\n".join(email_list)
#         return "Please specify 'read emails' in your request."

#     def respond_to_email(self, user_input):
#         if "respond to email" in user_input.lower():
#             if not self.last_read_emails:
#                 return "No emails have been read yet. Use 'read emails' first."
            
#             email_index = int(input("Which email would you like to respond to? (Enter the number): ").strip()) - 1
#             if email_index < 0 or email_index >= len(self.last_read_emails):
#                 return "Invalid email selection."
            
#             selected_email = self.last_read_emails[email_index]
#             context = input("Provide context for the response: ").strip()
#             return self.email_tool.generate_and_send_response(selected_email, context)
#         return "Please specify 'respond to email [X]' in your request."

#     def chat(self, user_input):
#         if "search" in user_input.lower():
#             query = user_input.replace("search", "").strip()
#             return self.generate_response(query)
#         elif "process" in user_input.lower():
#             directory_path = input("Enter the directory path containing PDF files: ").strip()
#             self.vector_store = self.document_tool.process_documents(directory_path)
#             self.is_document_context_active = True
#             return "Documents processed and stored in vector store."
#         elif "compare" in user_input.lower():
#             query = input("Enter your query to compare documents: ").strip()
#             if not self.vector_store:
#                 return "No documents processed yet. Please process documents first."
#             results = self.document_tool.compare_documents(self.vector_store, query)
#             return "\n".join(results)
#         elif "send email" in user_input.lower():
#             return self.send_email_manually(user_input)
#         elif "read emails" in user_input.lower():
#             return self.read_emails(user_input)
#         elif "respond to email" in user_input.lower():
#             return self.respond_to_email(user_input)
#         elif self.is_document_context_active or any(keyword in user_input.lower() for keyword in ["document", "pdf", "report"]):
#             return self.process_and_chat_documents(user_input)
#         else:
#             return "I can help you search, process documents, read emails, or respond to emails."


# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_core.runnables.base import RunnableSequence
# from langchain.memory import ConversationBufferMemory
# from tools.search_tool import SearchTool
# from tools.document_processing_tool import DocumentProcessingTool
# from tools.email_tool import EmailTool
# from utils.env_loader import load_env
# import uuid
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph




# class AIAgent:
#     def __init__(self):
#         groq_api_key = load_env()[0]
#         self.llm = ChatGroq(model="llama-3.2-90b-vision-preview", api_key=groq_api_key)
#         self.search_tool = SearchTool()
#         self.document_tool = DocumentProcessingTool()
#         self.email_tool = EmailTool(llm=self.llm)
#         self.vector_store = None
#         self.last_search_results = None
#         self.last_read_emails = []
#         self.is_document_context_active = False  # Initialize flag
#         self.doc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         # Initialize LangGraph memory
#         self.thread_id = uuid.uuid4()
#         self.memory = MemorySaver()

#         # Define intent detection prompt with examples
#         intent_prompt = ChatPromptTemplate(
#             [
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 HumanMessagePromptTemplate.from_template(
#                     """
#                     Instructions: Classify the user's input into one of the following intents based on the context:
#                     - search: The user wants to search for information (e.g., "find me news about AI", "search nvidia").
#                     - process_documents: The user wants to process documents (e.g., "process PDFs", "analyze documents").
#                     - read_emails: The user wants to read emails (e.g., "check emails", "show my inbox").
#                     - send_email: The user wants to send an email (e.g., "send a message", "email support").
#                     - respond_to_email: The user wants to reply to an email (e.g., "reply to email 1").
#                     - general_chat: The user is asking a general question or making a statement (e.g., "hello", "what do you think about this?").

#                     User Input: {input}
#                     Intent (only the intent name, nothing else):
#                     """
#                 ),
#             ]
#         )
#         self.intent_chain = intent_prompt | self.llm
#             # Initialize LangChain memory for document discussions
#         self.doc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         # Define the chat model function
#         def call_model(state: MessagesState):
#             response = self.llm.invoke(state["messages"])
#             return {"messages": response}

#         # Define the workflow graph
#         self.workflow = StateGraph(state_schema=MessagesState)
#         self.workflow.add_edge(START, "model")
#         self.workflow.add_node("model", call_model)
#         self.app = self.workflow.compile(checkpointer=self.memory)

#         # Initialize chat history
#         self.chat_history = []

#     def detect_intent(self, user_input):
#         # Add user input to chat history
#         self.chat_history.append(HumanMessage(content=user_input))
        
#         # Detect intent
#         try:
#             intent_response = self.intent_chain.invoke({
#                 "chat_history": self.chat_history,
#                 "input": user_input
#             })
#             intent = intent_response.content.strip().lower()
#         except Exception as e:
#             intent = "general_chat"
#             print(f"Intent detection error: {str(e)}")
        
#         # Add AI response to chat history
#         ai_response = AIMessage(content=intent)
#         self.chat_history.append(ai_response)
        
#         return intent

#     def generate_response(self, query):
#         try:
#             news_results = self.search_tool.search_latest_news(query)
#             news_summary = "\n".join([f"- {item}" for item in news_results[:5]])
#             response = self.llm.invoke(f"Summarize and provide insights on the following news:\n{news_summary}").content
#             self.last_search_results = news_summary
#             return response
#         except Exception as e:
#             return f"Error: {str(e)}. Please try again later."


#     def process_and_chat_documents(self, query):
#         if not self.vector_store:
#             return "No documents processed yet. Please process documents first."
        
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
#             memory=self.doc_memory,  
#             get_chat_history=lambda h: h.chat_history,  # Correctly references chat_history
#             verbose=True
#         )
        
#         result = qa_chain({"question": query})
#         return result["answer"]
    
    

#     def send_email_manually(self):
#         to = input("Enter recipient's email address: ").strip()
#         subject = input("Enter email subject: ").strip()
#         body = input("Enter email body: ").strip()
#         return self.email_tool.send_email(to, subject, body)

#     def read_emails(self):
#         num_emails = int(input("How many emails would you like to read? ").strip())
#         self.last_read_emails = self.email_tool.read_emails(num_emails=num_emails)
#         if isinstance(self.last_read_emails, str):
#             return self.last_read_emails
        
#         email_list = []
#         for i, email in enumerate(self.last_read_emails):
#             email_list.append(f"\nEmail {i + 1}:\nFrom: {email['from']}\nSubject: {email['subject']}\nBody: {email['body'][:100]}...")
#         return "\n".join(email_list)

#     def respond_to_email(self):
#         if not self.last_read_emails:
#             return "No emails have been read yet. Use 'read emails' first."
        
#         email_index = int(input("Which email would you like to respond to? (Enter the number): ").strip()) - 1
#         if email_index < 0 or email_index >= len(self.last_read_emails):
#             return "Invalid email selection."
        
#         selected_email = self.last_read_emails[email_index]
#         context = input("Provide context for the response: ").strip()
#         return self.email_tool.generate_and_send_response(selected_email, context)

#     def handle_general_chat(self, user_input):
#         # Use LLM with chat history
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 *self.chat_history,
#                 HumanMessage(content=user_input),
#             ]
#         )
#         response = self.llm.invoke(prompt.format_prompt()).content
#         return response

#     def chat(self, user_input):
#             # Intent detection first
#         intent = self.detect_intent(user_input)
        
#         # Document context handling
#         if self.vector_store and (self.is_document_context_active or 
#                                  any(kw in user_input.lower() for kw in ["document", "pdf", "report", "processed"])):
#             return self.process_and_chat_documents(user_input)
        
#         if intent == "search":
#             query = user_input
#             response = self.generate_response(query)
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "process_documents":
#             directory_path = input("Enter the directory path containing PDF files: ").strip()
#             self.vector_store = self.document_tool.process_documents(directory_path)
#             self.is_document_context_active = True  # THIS IS CORRECT
#             return "Documents processed and stored in vector store."
#         elif intent == "read_emails":
#             response = self.read_emails()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "send_email":
#             response = self.send_email_manually()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "respond_to_email":
#             response = self.respond_to_email()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "general_chat":
#             return self.handle_general_chat(user_input)
#         else:
#             return "I'm here to help! Try asking me something."

#     def run_chat(self, user_input):
#         human_message = HumanMessage(content=user_input)
#         config = {"configurable": {"thread_id": self.thread_id}}
        
#         try:
#             for event in self.app.stream({"messages": [human_message]}, config, stream_mode="values"):
#                 ai_message = event["messages"][-1]
#                 return ai_message.content
#         except Exception as e:
#             return f"Error: {str(e)}"


# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_core.runnables.base import RunnableSequence
# from langchain.memory import ConversationBufferMemory
# from tools.search_tool import SearchTool
# from tools.document_processing_tool import DocumentProcessingTool
# from tools.email_tool import EmailTool
# from utils.env_loader import load_env
# import uuid
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph


# class AIAgent:
#     def __init__(self):
#         groq_api_key = load_env()[0]
#         self.llm = ChatGroq(model="llama-3.2-90b-vision-preview", api_key=groq_api_key)
#         self.search_tool = SearchTool()
#         self.document_tool = DocumentProcessingTool()
#         self.email_tool = EmailTool(llm=self.llm)
#         self.vector_store = None
#         self.last_search_results = None
#         self.last_read_emails = []
#         self.is_document_context_active = False  # Initialize flag

#         # Initialize LangChain memory for document discussions
#         self.doc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         # Initialize LangChain memory for general conversation
#         self.general_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#         # Initialize LangGraph memory
#         self.thread_id = uuid.uuid4()
#         self.memory = MemorySaver()

#         # Define intent detection prompt with examples
#         intent_prompt = ChatPromptTemplate(
#             [
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 HumanMessagePromptTemplate.from_template(
#                     """
#                     Instructions: Classify the user's input into one of the following intents based on the context:
#                     - search: The user wants to search for information (e.g., "find me news about AI", "search nvidia").
#                     - process_documents: The user wants to process documents (e.g., "process PDFs", "analyze documents").
#                     - read_emails: The user wants to read emails (e.g., "check emails", "show my inbox").
#                     - send_email: The user wants to send an email (e.g., "send a message", "email support").
#                     - respond_to_email: The user wants to reply to an email (e.g., "reply to email 1").
#                     - general_chat: The user is asking a general question or making a statement (e.g., "hello", "what do you think about this?").

#                     User Input: {input}
#                     Intent (only the intent name, nothing else):
#                     """
#                 ),
#             ]
#         )
#         self.intent_chain = intent_prompt | self.llm

#         # Define the chat model function
#         def call_model(state: MessagesState):
#             response = self.llm.invoke(state["messages"])
#             return {"messages": response}

#         # Define the workflow graph
#         self.workflow = StateGraph(state_schema=MessagesState)
#         self.workflow.add_edge(START, "model")
#         self.workflow.add_node("model", call_model)
#         self.app = self.workflow.compile(checkpointer=self.memory)

#         # Initialize chat history
#         self.chat_history = []

#     def detect_intent(self, user_input):
#         # Add user input to chat history
#         self.chat_history.append(HumanMessage(content=user_input))
        
#         # Detect intent
#         try:
#             intent_response = self.intent_chain.invoke({
#                 "chat_history": self.chat_history,
#                 "input": user_input
#             })
#             intent = intent_response.content.strip().lower()
#         except Exception as e:
#             intent = "general_chat"
#             print(f"Intent detection error: {str(e)}")
        
#         # Add AI response to chat history
#         ai_response = AIMessage(content=intent)
#         self.chat_history.append(ai_response)
        
#         return intent

#     def generate_response(self, query):
#         try:
#             news_results = self.search_tool.search_latest_news(query)
#             news_summary = "\n".join([f"- {item}" for item in news_results[:5]])
#             response = self.llm.invoke(f"Summarize and provide insights on the following news:\n{news_summary}").content
#             self.last_search_results = news_summary
#             self.general_memory.save_context({"input": query}, {"output": response})
#             return response
#         except Exception as e:
#             return f"Error: {str(e)}. Please try again later."

#     def process_and_chat_documents(self, query):
#         if not self.vector_store:
#             return "No documents processed yet. Please process documents first."
        
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
#             memory=self.doc_memory,  
#             get_chat_history=lambda h: h,  # Corrected to return the list directly
#             verbose=True
#         )
        
#         result = qa_chain.invoke({"question": query})
#         return result["answer"]
    
#     def read_emails(self):
#         num_emails = int(input("How many emails would you like to read? ").strip())
#         self.last_read_emails = self.email_tool.read_emails(num_emails=num_emails)
#         if isinstance(self.last_read_emails, str):
#             response = self.last_read_emails
#         else:
#             email_list = []
#             for i, email in enumerate(self.last_read_emails):
#                 email_list.append(f"\nEmail {i + 1}:\nFrom: {email['from']}\nSubject: {email['subject']}\nBody: {email['body'][:100]}...")
#             response = "\n".join(email_list)
        
#         # Save context with separate inputs and outputs
#         self.general_memory.save_context(
#             inputs={"input": f"read {num_emails} emails"},
#             outputs={"output": response}
#         )
#         return response

#     def send_email_manually(self):
#         to = input("Enter recipient's email address: ").strip()
#         subject = input("Enter email subject: ").strip()
#         body = input("Enter email body: ").strip()
#         response = self.email_tool.send_email(to, subject, body)
        
#         # Save context with separate inputs and outputs
#         self.general_memory.save_context(
#             inputs={"input": f"send email to {to} with subject '{subject}'"},
#             outputs={"output": response}
#         )
#         return response

#     def respond_to_email(self):
#         if not self.last_read_emails:
#             return "No emails have been read yet. Use 'read emails' first."
        
#         email_index = int(input("Which email would you like to respond to? (Enter the number): ").strip()) - 1
#         if email_index < 0 or email_index >= len(self.last_read_emails):
#             return "Invalid email selection."
        
#         selected_email = self.last_read_emails[email_index]
#         context = input("Provide context for the response: ").strip()
#         response = self.email_tool.generate_and_send_response(selected_email, context)
        
#         # Save context with separate inputs and outputs
#         self.general_memory.save_context(
#             inputs={"input": f"respond to email {email_index + 1}"},
#             outputs={"output": response}
#         )
#         return response

#     def handle_general_chat(self, user_input):
#         # Use LLM with chat history
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 *self.chat_history,
#                 HumanMessage(content=user_input),
#             ]
#         )
#         response = self.llm.invoke(prompt.format_prompt()).content
#         self.general_memory.save_context({"input": user_input}, {"output": response})
#         return response

#     def chat(self, user_input):
#         # Intent detection first
#         intent = self.detect_intent(user_input)
        
#         # Document context handling
#         if self.vector_store and (self.is_document_context_active or 
#                                  any(kw in user_input.lower() for kw in ["document", "pdf", "report", "processed"])):
#             return self.process_and_chat_documents(user_input)
        
#         if intent == "search":
#             query = user_input.replace("search", "").strip()
#             response = self.generate_response(query)
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "process_documents":
#             directory_path = input("Enter the directory path containing PDF files: ").strip()
#             self.vector_store = self.document_tool.process_documents(directory_path)
#             response = "Documents processed and stored in vector store."
#             self.chat_history.append(AIMessage(content=response))
#             self.is_document_context_active = True
#             return response
#         elif intent == "read_emails":
#             response = self.read_emails()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "send_email":
#             response = self.send_email_manually()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "respond_to_email":
#             response = self.respond_to_email()
#             self.chat_history.append(AIMessage(content=response))
#             return response
#         elif intent == "general_chat":
#             return self.handle_general_chat(user_input)
#         else:
#             return "I'm here to help! Try asking me something."

#     def run_chat(self, user_input):
#         human_message = HumanMessage(content=user_input)
#         config = {"configurable": {"thread_id": self.thread_id}}
        
#         try:
#             for event in self.app.stream({"messages": [human_message]}, config, stream_mode="values"):
#                 ai_message = event["messages"][-1]
#                 return ai_message.content
#         except Exception as e:
#             return f"Error: {str(e)}"



from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.runnables.base import RunnableSequence
from langchain.memory import ConversationBufferMemory
from tools.search_tool import SearchTool
from tools.document_processing_tool import DocumentProcessingTool
from tools.email_tool import EmailTool
from utils.env_loader import load_env
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


class AIAgent:
    def __init__(self):
        groq_api_key = load_env()[0]
        self.llm = ChatGroq(model="llama-3.2-90b-vision-preview", api_key=groq_api_key)
        self.search_tool = SearchTool()
        self.document_tool = DocumentProcessingTool()
        self.email_tool = EmailTool(llm=self.llm)
        self.vector_store = None
        self.last_search_results = None
        self.last_read_emails = []
        self.is_document_context_active = False  # Initialize flag

        # Initialize memory for document discussions
        self.doc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize memory for general conversation
        self.general_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        
        

        # Initialize LangGraph memory
        self.thread_id = uuid.uuid4()
        self.memory = MemorySaver()

        # Define intent detection prompt with examples
# In __init__ method, update the intent_prompt:
# In __init__ method, update the intent_prompt:
        intent_prompt = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    """
                    Instructions: Classify the user's input into one of the following intents based on the context:
                    - search: The user wants to search for information (e.g., "find me news about AI", "search nvidia", "what's the latest on quantum computing").
                    - process_documents: The user wants to process documents (e.g., "process PDFs", "analyze documents", "read the report").
                    - read_emails: The user wants to read emails (e.g., "check emails", "show my inbox", "can you read emails?").
                    - send_email: The user wants to send an email (e.g., "send a message", "email support", "compose an email").
                    - respond_to_email: The user wants to reply to an email (e.g., "reply to email 1", "respond to the last email").
                    - general_chat: The user is asking a general question or making a statement (e.g., "hello", "what do you think about this?").

                    User Input: {input}
                    Intent (only the intent name, nothing else):
                    """
                ),
            ]
        )
        self.intent_chain = intent_prompt | self.llm

        # Define the chat model function
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}

        # Define the workflow graph
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Initialize chat history
        self.chat_history = []

    def detect_intent(self, user_input):
        # Add user input to chat history
        self.chat_history.append(HumanMessage(content=user_input))
        
        # Detect intent
        try:
            intent_response = self.intent_chain.invoke({
                "chat_history": self.chat_history,
                "input": user_input
            })
            intent = intent_response.content.strip().lower()
        except Exception as e:
            intent = "general_chat"
            print(f"Intent detection error: {str(e)}")
        
        # Add AI response to chat history
        ai_response = AIMessage(content=intent)
        self.chat_history.append(ai_response)
        
        return intent

    def generate_response(self, query):
        try:
            news_results = self.search_tool.search_latest_news(query)
            news_summary = "\n".join([f"- {item}" for item in news_results[:5]])
            response = self.llm.invoke(f"Summarize and provide insights on the following news:\n{news_summary}").content
            self.last_search_results = news_summary
            # Save to general_memory instead of doc_memory
            self.general_memory.save_context(
                inputs={"input": query},
                outputs={"output": response}
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}. Please try again later."

    def process_and_chat_documents(self, query):
        if not self.vector_store:
            return "No documents processed yet. Please process documents first."
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.doc_memory,  
            get_chat_history=lambda h: h,  # Return list directly
            verbose=True
        )
        
        result = qa_chain.invoke({"question": query})
        return result["answer"]
    
    def read_emails(self):
        num_emails = int(input("How many emails would you like to read? ").strip())
        self.last_read_emails = self.email_tool.read_emails(num_emails=num_emails)
        if isinstance(self.last_read_emails, str):
            response = self.last_read_emails
        else:
            email_list = []
            for i, email in enumerate(self.last_read_emails):
                email_list.append(f"\nEmail {i + 1}:\nFrom: {email['from']}\nSubject: {email['subject']}\nBody: {email['body'][:100]}...")
            response = "\n".join(email_list)
        
        # Save context with separate inputs and outputs
        self.general_memory.save_context(
            inputs={"input": f"read {num_emails} emails"},
            outputs={"output": response}
        )
        return response

    def send_email_manually(self):
        to = input("Enter recipient's email address: ").strip()
        subject = input("Enter email subject: ").strip()
        body = input("Enter email body: ").strip()
        response = self.email_tool.send_email(to, subject, body)
        
        # Save context with separate inputs and outputs
        self.general_memory.save_context(
            inputs={"input": f"send email to {to} with subject '{subject}'"},
            outputs={"output": response}
        )
        return response

    def respond_to_email(self):
        if not self.last_read_emails:
            return "No emails have been read yet. Use 'read emails' first."
        
        email_index = int(input("Which email would you like to respond to? (Enter the number): ").strip()) - 1
        if email_index < 0 or email_index >= len(self.last_read_emails):
            return "Invalid email selection."
        
        selected_email = self.last_read_emails[email_index]
        context = input("Provide context for the response: ").strip()
        response = self.email_tool.generate_and_send_response(selected_email, context)
        
        # Save context with separate inputs and outputs
        self.general_memory.save_context(
            inputs={"input": f"respond to email {email_index + 1}"},
            outputs={"output": response}
        )
        return response

    def handle_general_chat(self, user_input):
        # Use LLM with general_memory
        prompt = ChatPromptTemplate.from_messages(
            [
                *self.general_memory.chat_memory.messages,  # Corrected attribute
                HumanMessage(content=user_input),
            ]
        )
        response = self.llm.invoke(prompt.format_prompt()).content
        
        # Save context to general_memory
        self.general_memory.save_context(
            inputs={"input": user_input},
            outputs={"output": response}
        )
        return response

    # def chat(self, user_input):
    #     intent = self.detect_intent(user_input)
        
    #     # Handle explicit intents first but don't return immediately
    #     response = None
    #     if intent == "search":
    #         query = user_input
    #         response = self.generate_response(query)
    #         self.chat_history.append(AIMessage(content=response))
    #     elif intent == "process_documents":
    #         directory_path = input("Enter the directory path containing PDF files: ").strip()
    #         self.vector_store = self.document_tool.process_documents(directory_path)
    #         response = "Documents processed and stored in vector store."
    #         self.chat_history.append(AIMessage(content=response))
    #         self.is_document_context_active = True
    #     elif intent == "read_emails":
    #         response = self.read_emails()
    #         self.chat_history.append(AIMessage(content=response))
    #     elif intent == "send_email":
    #         response = self.send_email_manually()
    #         self.chat_history.append(AIMessage(content=response))
    #     elif intent == "respond_to_email":
    #         response = self.respond_to_email()
    #         self.chat_history.append(AIMessage(content=response))
    #     elif intent == "general_chat":
    #         response = self.handle_general_chat(user_input)
    #     else:
    #         response = "I'm here to help! Try asking me something."

    #     # Reset document context if not explicitly document-related
    #     if intent not in ["process_documents"]:
    #         self.is_document_context_active = False

    #     # Document context check as a fallback
    #     if self.vector_store and (self.is_document_context_active or 
    #                             any(kw in user_input.lower() for kw in ["document", "pdf", "report", "processed"])):
    #         return self.process_and_chat_documents(user_input)
        
    #     # Return the response after all checks
    #     return response if response is not None else "I'm here to help! Try asking me something."

    def chat(self, user_input):
        intent = self.detect_intent(user_input)
        
        # Reset document context for non-document intents
        if intent not in ["process_documents"]:
            self.is_document_context_active = False
            
        # Document context check as fallback
        if self.vector_store and (self.is_document_context_active or 
                                any(kw in user_input.lower() for kw in ["document", "pdf", "report", "processed"])):
            return self.process_and_chat_documents(user_input)

        # Handle explicit intents
        if intent == "search":
            query = user_input
            response = self.generate_response(query)
            self.chat_history.append(AIMessage(content=response))
            return response
        elif intent == "process_documents":
            directory_path = input("Enter the directory path containing PDF files: ").strip()
            self.vector_store = self.document_tool.process_documents(directory_path)
            response = "Documents processed and stored in vector store."
            self.chat_history.append(AIMessage(content=response))
            self.is_document_context_active = True
            return response
        elif intent == "read_emails":
            response = self.read_emails()
            self.chat_history.append(AIMessage(content=response))
            return response
        elif intent == "send_email":
            response = self.send_email_manually()
            self.chat_history.append(AIMessage(content=response))
            return response
        elif intent == "respond_to_email":
            response = self.respond_to_email()
            self.chat_history.append(AIMessage(content=response))
            return response
        elif intent == "general_chat":
            return self.handle_general_chat(user_input)
        
        return "I'm here to help! Try asking me something."


    def run_chat(self, user_input):
        human_message = HumanMessage(content=user_input)
        config = {"configurable": {"thread_id": self.thread_id}}
        
        try:
            for event in self.app.stream({"messages": [human_message]}, config, stream_mode="values"):
                ai_message = event["messages"][-1]
                return ai_message.content
        except Exception as e:
            return f"Error: {str(e)}"