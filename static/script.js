const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const loading = document.getElementById('loading');

function addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    console.log(message);
    if (!message) return;

    // loading.style.display = 'block';
    chatContainer.style.display = "block";
    
    addMessage('user', message);
    userInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        addMessage('ai', data.ai_response);
        
    } catch (error) {
        addMessage('ai', 'Sorry, I encountered an error. Please try again.');
    } finally {
        loading.style.display = 'none';
    }
}

// Handle Enter key
userInput.addEventListener('keypress', (e) => {
    // console.log("Event listener triggered");
    if (e.key === 'Enter' && !e.shiftKey) {
        sendMessage();
    }
});