# app.py
from flask import Flask, render_template, request, jsonify, session

# Here I am using AWS connector file. To change it, we can use the llm_connector instead and define our own
# model endpoint.
from aws_connect import get_ai_response

application = Flask(__name__)
application.secret_key = ''

# Function to get response from AI with optional conversation history.
# However we are operating with low contect windows so conversation history is not implemented yet.
def ai_response(user_input, conversation_history):
    lower_input = user_input.lower()
    response = get_ai_response(user_input,conversation_history)
    return response


@application.route('/')
def home():
    # Initialize conversation history
    session['conversation'] = []  
    return render_template('index.html')

@application.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    print(user_message)
    conversation = session.get('conversation', [])
    
    # Get response from llm
    ai_reply = ai_response(user_message, conversation)
    
    # Update conversation history
    conversation.append({'sender': 'user', 'text': user_message})
    conversation.append({'sender': 'ai', 'text': ai_reply})
    session['conversation'] = conversation
    
    return jsonify({
        'ai_response': ai_reply,
        'conversation': conversation
    })

if __name__ == '__main__':
    application.run(host="",port=)
