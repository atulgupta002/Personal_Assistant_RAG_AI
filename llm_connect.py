# Optional file for initial testing and later langchain implementation

def get_ai_response(user_input, conversation_history):
    if 'hello' in user_input:
        return "Hello! How can I assist you today?"
    elif 'help' in user_input:
        return "I'm here to help! What specific questions do you have?"
    elif 'bye' in user_input:
        return "Goodbye! Feel free to reach out if you have more questions."
    else:
        return "Interesting! Can you tell me more about that?"